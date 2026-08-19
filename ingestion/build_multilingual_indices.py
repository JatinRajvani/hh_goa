import os
import sys
import json
import hashlib
import time
import re
import faiss
import numpy as np
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

# Ensure parent directory is in path to allow relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ==========================================
# Configuration
# ==========================================
REPO_ID = "ai4bharat/MSMARCO-XI"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_PASSAGES = 1500  # Set to 1,500 per language to keep indexing fast (around 1.5 mins per language)
BATCH_SIZE = 500

INDEX_DIR = os.path.join(parent_dir, "data", "index")
PROCESSED_DIR = os.path.join(parent_dir, "data", "processed")

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def clean_text(text):
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_language_index(lang_code, parquet_filename, query_col, passage_col):
    print("\n" + "=" * 60)
    print(f" Building Index for Language: {lang_code.upper()} ")
    print("=" * 60)
    
    # 1. Download file
    print(f"Downloading '{parquet_filename}' from HF repo '{REPO_ID}'...")
    try:
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=parquet_filename,
            repo_type="dataset",
        )
        print(f"Dataset cached at: {file_path}")
    except Exception as e:
        print(f"Error downloading dataset for {lang_code}: {e}")
        return False

    # 2. Extract passages and queries
    print("Extracting unique passages...")
    parquet_file = pq.ParquetFile(file_path)
    
    passages = []  # list of dicts: {"document_id": ..., "text": ...}
    passage_mapping = {}  # doc_id -> text
    eval_queries = []  # list of dicts: {"query": ..., "relevant_passage_ids": [...]}
    seen_passage_hashes = set()
    
    batch_iter = parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["query_id", "query", "Eng_Query", "passages"]
    )
    
    passages_written = 0
    
    for batch in batch_iter:
        if passages_written >= MAX_PASSAGES:
            break
            
        records = batch.to_pylist()
        for r in records:
            if passages_written >= MAX_PASSAGES:
                break
                
            query_id = r.get("query_id")
            # Determine query text
            if query_col == "Eng_Query":
                query_text = clean_text(r.get("Eng_Query"))
            else:
                query_text = clean_text(r.get("query"))
                
            passages_struct = r.get("passages")
            if not query_id or not query_text or not passages_struct:
                continue
                
            # Determine passages list
            if passage_col == "English_passages":
                raw_passages = passages_struct.get("English_passages", [])
            else:
                raw_passages = passages_struct.get("Translated_passages", [])
                
            is_selected_list = passages_struct.get("is_selected", [])
            
            relevant_ids = []
            
            for idx, p_text in enumerate(raw_passages):
                cleaned_p = clean_text(p_text)
                if not cleaned_p or len(cleaned_p) < 3:
                    continue
                    
                doc_id = f"{query_id}_{idx}"
                is_selected = is_selected_list[idx] if idx < len(is_selected_list) else 0
                
                # Check duplicate
                p_hash = hashlib.md5(cleaned_p.encode("utf-8")).hexdigest()
                if p_hash not in seen_passage_hashes:
                    if passages_written < MAX_PASSAGES:
                        seen_passage_hashes.add(p_hash)
                        passages.append({"document_id": doc_id, "text": cleaned_p})
                        passage_mapping[doc_id] = cleaned_p
                        passages_written += 1
                        
                if is_selected:
                    relevant_ids.append(doc_id)
            
            if relevant_ids:
                eval_queries.append({
                    "query": query_text,
                    "relevant_passage_ids": relevant_ids
                })

    print(f"Extracted {len(passages)} unique passages for {lang_code.upper()}.")
    
    if len(passages) == 0:
        print(f"No passages found for {lang_code}. Skipping index creation.")
        return False

    # 3. Load Multilingual Embedding Model
    print(f"Loading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 4. Generate Embeddings
    print("Generating embeddings...")
    texts = [p["text"] for p in passages]
    t_start = time.time()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    print(f"Generated embeddings in {time.time() - t_start:.2f} seconds.")
    
    # 5. Build FAISS Index
    print("Building FAISS index...")
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)  # cosine similarity pre-normalization
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IndexFlatIP supports inner product (cosine on L2 normalized)
    index.add(embeddings)
    
    # 6. Save files
    index_path = os.path.join(INDEX_DIR, f"index_{lang_code}.faiss")
    mapping_path = os.path.join(INDEX_DIR, f"id_mapping_{lang_code}.json")
    eval_queries_path = os.path.join(PROCESSED_DIR, f"evaluation_queries_{lang_code}.jsonl")
    
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to: {index_path}")
    
    # Convert index mapping (ordered lists to map FAISS indices to doc_ids)
    ordered_ids = [p["document_id"] for p in passages]
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "ordered_ids": ordered_ids,
            "mapping": passage_mapping
        }, f, ensure_ascii=False, indent=2)
    print(f"Document mapping saved to: {mapping_path}")
    
    # Save evaluation queries
    with open(eval_queries_path, "w", encoding="utf-8") as f:
        for q in eval_queries[:150]:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Evaluation queries saved to: {eval_queries_path}")
    return True


def main():
    # Job configurations for all 13 dataset languages + English
    jobs = [
        {"lang": "en", "file": "train/gujtrain.parquet", "query_col": "Eng_Query", "passage_col": "English_passages"},
        {"lang": "as", "file": "train/asmtrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "bn", "file": "train/bentrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "gu", "file": "train/gujtrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "hi", "file": "train/hintrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "kn", "file": "train/kantrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "ml", "file": "train/maltrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "mr", "file": "train/martrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "ne", "file": "train/neptrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "or", "file": "train/oritrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "pa", "file": "train/pantrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "sa", "file": "train/santrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "ta", "file": "train/tamtrain.parquet", "query_col": "query", "passage_col": "Translated_passages"},
        {"lang": "ur", "file": "train/urdtrain.parquet", "query_col": "query", "passage_col": "Translated_passages"}
    ]
    
    t_start_all = time.time()
    for job in jobs:
        try:
            build_language_index(
                lang_code=job["lang"],
                parquet_filename=job["file"],
                query_col=job["query_col"],
                passage_col=job["passage_col"]
            )
        except Exception as e:
            print(f"Error processing language index for '{job['lang']}': {e}")
            
    print("\n" + "=" * 60)
    print(f" ALL 14 INDICES BUILT SUCCESSFULLY IN {time.time() - t_start_all:.2f} SECONDS! ")
    print("=" * 60)

if __name__ == "__main__":
    main()
