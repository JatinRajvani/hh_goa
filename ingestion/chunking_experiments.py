import os
import sys
import json
import time
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Ensure parent directory is in path to allow relative imports when run as script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

load_dotenv()

CORPUS_FILE = "data/processed/english_passages.jsonl"
EVAL_FILE = "data/processed/evaluation_queries.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"
SUBSET_LIMIT = 5000

# ----------------------------------------------------
# 1. Chunkers Implementation
# ----------------------------------------------------

def chunk_fixed_size(passages, chunk_size=100, overlap=20):
    """Chunks text into fixed word counts with overlap."""
    chunked_docs = []
    
    for p in passages:
        text = p["text"]
        doc_id = p["document_id"]
        words = text.split()
        
        if len(words) <= chunk_size:
            # Document is small enough, no splitting needed
            chunked_docs.append({
                "document_id": f"{doc_id}_fixed_0",
                "text": text,
                "metadata": {**p["metadata"], "original_document_id": doc_id}
            })
            continue
            
        # Perform sliding window chunking
        start = 0
        chunk_idx = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            chunked_docs.append({
                "document_id": f"{doc_id}_fixed_{chunk_idx}",
                "text": chunk_text,
                "metadata": {**p["metadata"], "original_document_id": doc_id}
            })
            
            start += (chunk_size - overlap)
            chunk_idx += 1
            
    return chunked_docs

def chunk_sentence_aware(passages, target_words=120):
    """Chunks text by grouping whole sentences to avoid mid-sentence splits."""
    chunked_docs = []
    sentence_end_regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    
    for p in passages:
        text = p["text"]
        doc_id = p["document_id"]
        
        # Split into sentences
        sentences = sentence_end_regex.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            continue
            
        current_chunk = []
        current_words = 0
        chunk_idx = 0
        
        for sent in sentences:
            sent_words = len(sent.split())
            if current_words + sent_words > target_words and current_chunk:
                # Save current chunk and start a new one
                chunked_docs.append({
                    "document_id": f"{doc_id}_sent_{chunk_idx}",
                    "text": " ".join(current_chunk),
                    "metadata": {**p["metadata"], "original_document_id": doc_id}
                })
                current_chunk = [sent]
                current_words = sent_words
                chunk_idx += 1
            else:
                current_chunk.append(sent)
                current_words += sent_words
                
        # Append final chunk
        if current_chunk:
            chunked_docs.append({
                "document_id": f"{doc_id}_sent_{chunk_idx}",
                "text": " ".join(current_chunk),
                "metadata": {**p["metadata"], "original_document_id": doc_id}
            })
            
    return chunked_docs

# ----------------------------------------------------
# 2. Main Benchmarking Logic
# ----------------------------------------------------

def run_experiments():
    print("==================================================")
    print("           Starting Chunking Experiments          ")
    print("==================================================")
    
    # 1. Load subset of passages
    if not os.path.exists(CORPUS_FILE):
        print(f"Corpus file '{CORPUS_FILE}' not found. Run extract_english_passages.py first.")
        return
        
    print(f"Loading first {SUBSET_LIMIT} documents from corpus...")
    subset_passages = []
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                subset_passages.append(json.loads(line))
            if len(subset_passages) >= SUBSET_LIMIT:
                break
                
    subset_doc_ids = set([d["document_id"] for d in subset_passages])
    print(f"Loaded {len(subset_passages)} baseline passages.")
    
    # 2. Load evaluation queries
    if not os.path.exists(EVAL_FILE):
        print(f"Evaluation queries file '{EVAL_FILE}' not found. Run Phase 2 extraction first.")
        return
        
    print("Loading evaluation queries...")
    eval_queries = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eq = json.loads(line)
                # Find queries whose ground-truth relevant passage is in our subset
                gt_ids = set(eq.get("relevant_passage_ids", []))
                if gt_ids and gt_ids.issubset(subset_doc_ids):
                    eval_queries.append(eq)
                    
    print(f"Found {len(eval_queries)} evaluation queries mapped to this subset.")
    if not eval_queries:
        print("Error: No evaluation queries are fully represented in this subset. Try increasing SUBSET_LIMIT.")
        return
        
    # Limit eval queries for speed
    test_queries = eval_queries[:150]
    print(f"Using {len(test_queries)} queries for benchmarking.")
    
    # 3. Generate Chunks for each Strategy
    print("\nGenerating chunks...")
    strategies = {
        "Baseline": [
            # Wrap baseline passages with metadata field for uniform recall calculation
            {**p, "metadata": {**p["metadata"], "original_document_id": p["document_id"]}}
            for p in subset_passages
        ],
        "Fixed-Size": chunk_fixed_size(subset_passages, chunk_size=100, overlap=20),
        "Sentence-Aware": chunk_sentence_aware(subset_passages, target_words=120)
    }
    
    # Print chunk sizes
    for name, docs in strategies.items():
        print(f"  - {name}: {len(docs)} chunks")
        
    # 4. Load Embedding Model
    print(f"\nLoading embedding model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    results = {}
    
    # 5. Build FAISS Index and Evaluate each strategy
    for name, docs in strategies.items():
        print(f"\nEvaluating strategy: {name}...")
        
        # Extract text and metadata mappings
        texts = [d["text"] for d in docs]
        id_mapping = {idx: d for idx, d in enumerate(docs)}
        
        # Generate embeddings
        t0 = time.time()
        embeddings = model.encode(texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        embed_time = time.time() - t0
        print(f"  - Generated embeddings in {embed_time:.2f} seconds.")
        
        # Build Index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Benchmark queries
        latencies = []
        hits = 0
        k = 5
        
        # Run test queries
        for eq in test_queries:
            q_text = eq["query"]
            gt_ids = set(eq["relevant_passage_ids"])
            
            t_start = time.time()
            # Encode query
            q_vec = model.encode(q_text, convert_to_numpy=True).astype("float32").reshape(1, -1)
            faiss.normalize_L2(q_vec)
            
            # FAISS Search
            scores, indices = index.search(q_vec, k)
            latencies.append((time.time() - t_start) * 1000)
            
            # Map back to original document IDs to compute Recall
            retrieved_original_ids = []
            for offset in indices[0]:
                if offset != -1:
                    chunk_doc = id_mapping[offset]
                    retrieved_original_ids.append(chunk_doc["metadata"]["original_document_id"])
                    
            if gt_ids.intersection(retrieved_original_ids):
                hits += 1
                
        # Calculate stats
        recall_pct = (hits / len(test_queries)) * 100
        p50_latency = np.percentile(latencies, 50)
        p90_latency = np.percentile(latencies, 90)
        
        results[name] = {
            "recall": recall_pct,
            "p50_ms": p50_latency,
            "p90_ms": p90_latency,
            "total_chunks": len(docs)
        }
        
    # 6. Print Comparison Table
    print("\n" + "="*70)
    print("                   CHUNKING COMPARISON REPORT                 ")
    print("="*70)
    print(f"{'Strategy':<20} | {'Recall@5':<10} | {'P50 Latency':<12} | {'P90 Latency':<12} | {'Total Chunks':<12}")
    print("-" * 70)
    for name, stats in results.items():
        print(f"{name:<20} | {stats['recall']:>8.2f}% | {stats['p50_ms']:>9.2f} ms | {stats['p90_ms']:>9.2f} ms | {stats['total_chunks']:>12,}")
    print("="*70)
    
    # Save results to local json for reference
    out_file = "data/processed/chunking_experiments.json"
    with open(out_file, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2)
    print(f"\nExperiment results saved to: {out_file}\n")

if __name__ == "__main__":
    run_experiments()
