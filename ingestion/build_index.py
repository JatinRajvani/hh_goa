import os
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ==========================================
# Configuration
# ==========================================
CORPUS_FILE = "data/processed/english_passages.jsonl"
INDEX_DIR = "data/index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
MAP_FILE = os.path.join(INDEX_DIR, "id_mapping.json")

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256  # Batches for sentence encoder


def build_index():
    print("========== Starting Index Building ==========")
    
    if not os.path.exists(CORPUS_FILE):
        print(f"Error: Corpus file '{CORPUS_FILE}' not found. Please run Phase 2 extraction first.")
        return False
        
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # 1. Load passages
    print(f"Loading English passages from {CORPUS_FILE}...")
    start_time = time.time()
    passages = []
    
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                passages.append(json.loads(line))
                
    load_time = time.time() - start_time
    total_passages = len(passages)
    print(f"Loaded {total_passages:,} passages in {load_time:.2f} seconds.")
    
    if total_passages == 0:
        print("Error: No passages found in corpus file.")
        return False
        
    # 2. Initialize embedding model
    print(f"Loading embedding model '{MODEL_NAME}'...")
    model_start = time.time()
    model = SentenceTransformer(MODEL_NAME)
    
    # Determine device
    device = "cuda" if model.device.type == "cuda" else "cpu"
    print(f"Model loaded in {time.time() - model_start:.2f} seconds on device: {device}")
    
    # 3. Generate embeddings
    print(f"Generating embeddings for {total_passages:,} passages (Batch Size: {BATCH_SIZE})...")
    texts = [p["text"] for p in passages]
    
    embed_start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    embed_time = time.time() - embed_start
    print(f"Generated embeddings in {embed_time:.2f} seconds. Average speed: {total_passages / embed_time:.1f} passages/sec.")
    
    # Cast to float32 (required by FAISS)
    embeddings = embeddings.astype('float32')
    
    # 4. Normalize embeddings (L2 normalization turns Inner Product into Cosine Similarity)
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    # 5. Build FAISS Index
    dimension = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP (Dimension: {dimension})...")
    index_start = time.time()
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    index_time = time.time() - index_start
    print(f"FAISS index built with {index.ntotal:,} vectors in {index_time:.4f} seconds.")
    
    # 6. Save Index & Document Mapping
    print(f"Saving FAISS index to {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    
    print(f"Saving document ID mapping to {MAP_FILE}...")
    # Prepare mapping: FAISS index offset matches the list index exactly
    id_mapping = []
    for p in passages:
        id_mapping.append({
            "document_id": p["document_id"],
            "text": p["text"],
            "metadata": p["metadata"]
        })
        
    with open(MAP_FILE, "w", encoding="utf-8") as f_out:
        json.dump(id_mapping, f_out, ensure_ascii=False)
        
    print("\n========== Indexing Stats ==========")
    print(f"Total passages indexed: {index.ntotal:,}")
    print(f"Index Flat IP Size   : {os.path.getsize(INDEX_FILE) / (1024 * 1024):.2f} MB")
    print(f"ID Mapping File Size : {os.path.getsize(MAP_FILE) / (1024 * 1024):.2f} MB")
    print(f"Total indexing time  : {time.time() - start_time:.2f} seconds")
    print("Index build successful!\n")
    return True


if __name__ == "__main__":
    build_index()
