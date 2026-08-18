import os
import sys
import json
import time
import argparse
import numpy as np
from dotenv import load_dotenv

# Ensure parent directory is in path to allow relative imports when run as script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from retrieval.search import RetrievalService
from retrieval.rag_orchestrator import RAGOrchestrator

# Load environment variables
load_dotenv()

def load_eval_queries(file_path, limit=None):
    """Loads evaluation queries from jsonl file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation queries file not found: {file_path}")
        
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
            if limit and len(queries) >= limit:
                break
    return queries

def run_evaluation(num_retrieval=100, num_rag=15, k=5):
    print("==================================================")
    print("          Starting RAG Pipeline Evaluation        ")
    print("==================================================")
    
    # Paths
    eval_queries_file = "data/processed/evaluation_queries.jsonl"
    results_output_file = "data/processed/evaluation_results.json"
    
    # Load queries
    print(f"Loading evaluation queries from {eval_queries_file}...")
    max_to_load = max(num_retrieval, num_rag)
    all_queries = load_eval_queries(eval_queries_file, limit=max_to_load)
    print(f"Loaded {len(all_queries)} queries.")
    
    if not all_queries:
        print("No queries found for evaluation.")
        return
        
    # Initialize services
    print("\nInitializing Retrieval Service...")
    retriever = RetrievalService()
    retriever.load()
    
    print("\nInitializing RAG Orchestrator...")
    orchestrator = RAGOrchestrator()
    
    # ----------------------------------------------------
    # 1. Retrieval Benchmarking (Recall & Latency)
    # ----------------------------------------------------
    print(f"\n--- 1. Running Retrieval Benchmark ({num_retrieval} queries, K={k}) ---")
    retrieval_latencies = []
    recall_hits = 0
    valid_recall_queries = 0
    
    for idx, eq in enumerate(all_queries[:num_retrieval]):
        query_text = eq["query"]
        ground_truth_ids = set(eq.get("relevant_passage_ids", []))
        
        # We only evaluate recall for queries that have ground-truth relevant passages in our dataset
        has_ground_truth = len(ground_truth_ids) > 0
        if has_ground_truth:
            valid_recall_queries += 1
            
        # Search
        results, latency = retriever.search(query_text, k=k)
        retrieval_latencies.append(latency)
        
        # Calculate Recall
        if has_ground_truth:
            retrieved_ids = set([r["document_id"] for r in results])
            # If at least one ground-truth passage was retrieved, count as hit (Recall@K = 1.0 for this query)
            if ground_truth_ids.intersection(retrieved_ids):
                recall_hits += 1
                
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{num_retrieval} retrieval queries...")
            
    # Calculate retrieval percentiles
    r_p50 = np.percentile(retrieval_latencies, 50)
    r_p70 = np.percentile(retrieval_latencies, 70)
    r_p100 = np.max(retrieval_latencies)
    recall_accuracy = (recall_hits / valid_recall_queries * 100) if valid_recall_queries > 0 else 0.0
    
    # ----------------------------------------------------
    # 2. RAG End-to-End Benchmarking (STT, Retrieval, LLM, Total)
    # ----------------------------------------------------
    print(f"\n--- 2. Running RAG Benchmark ({num_rag} queries, K={k}) ---")
    rag_retrieval_latencies = []
    rag_generation_latencies = []
    rag_total_latencies = []
    relevance_passed_count = 0
    
    for idx, eq in enumerate(all_queries[:num_rag]):
        query_text = eq["query"]
        
        # Execute full RAG pipeline
        res = orchestrator.query_rag(query_text, k=k)
        
        rag_retrieval_latencies.append(res["latency_ms"]["retrieval"])
        rag_generation_latencies.append(res["latency_ms"]["generation"])
        rag_total_latencies.append(res["latency_ms"]["total_rag"])
        
        if res["relevance_passed"]:
            relevance_passed_count += 1
            
        print(f"  [{idx + 1}/{num_rag}] Query: '{query_text[:40]}...'")
        print(f"        Passed Guardrails: {res['relevance_passed']} | Total Latency: {res['latency_ms']['total_rag']:.2f} ms")
        
    # Calculate RAG percentiles
    # Total RAG
    t_p50 = np.percentile(rag_total_latencies, 50)
    t_p70 = np.percentile(rag_total_latencies, 70)
    t_p100 = np.max(rag_total_latencies)
    
    # Generation (only for queries that actually called the LLM)
    actual_gen_latencies = [l for l in rag_generation_latencies if l > 0]
    g_p50 = np.percentile(actual_gen_latencies, 50) if actual_gen_latencies else 0.0
    g_p70 = np.percentile(actual_gen_latencies, 70) if actual_gen_latencies else 0.0
    g_p100 = np.max(actual_gen_latencies) if actual_gen_latencies else 0.0
    
    # ----------------------------------------------------
    # 3. Report Generation
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("                 EVALUATION REPORT                ")
    print("="*50)
    print(f"Total Retrieval Queries Run   : {num_retrieval}")
    print(f"Queries with Ground Truth     : {valid_recall_queries}")
    print(f"Recall@{k} Accuracy           : {recall_accuracy:.2f}% ({recall_hits}/{valid_recall_queries} hits)")
    print("-" * 50)
    print("Latency Metrics (Retrieval Only):")
    print(f"  P50 (Median)                : {r_p50:.2f} ms")
    print(f"  P70                         : {r_p70:.2f} ms")
    print(f"  P100 (Max)                  : {r_p100:.2f} ms")
    print("-" * 50)
    print(f"Total RAG Queries Run         : {num_rag}")
    print(f"Passed Relevance Guardrails   : {relevance_passed_count}/{num_rag} ({relevance_passed_count/num_rag*100:.1f}%)")
    print("-" * 50)
    print("Latency Metrics (Full RAG Pipeline):")
    print(f"  RAG Total P50 (Median)      : {t_p50:.2f} ms")
    print(f"  RAG Total P70               : {t_p70:.2f} ms")
    print(f"  RAG Total P100 (Max)        : {t_p100:.2f} ms")
    if actual_gen_latencies:
        print("-" * 50)
        print("Latency Metrics (LLM Generation Only - Excluding Fallbacks):")
        print(f"  LLM Gen P50 (Median)        : {g_p50:.2f} ms")
        print(f"  LLM Gen P70                 : {g_p70:.2f} ms")
        print(f"  LLM Gen P100 (Max)          : {g_p100:.2f} ms")
    print("="*50)
    
    # Save results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_retrieval": num_retrieval,
            "num_rag": num_rag,
            "k": k,
            "relevance_threshold": orchestrator.threshold
        },
        "retrieval": {
            "p50_ms": float(r_p50),
            "p70_ms": float(r_p70),
            "p100_ms": float(r_p100),
            "recall_accuracy_percent": float(recall_accuracy),
            "hits": recall_hits,
            "total_valid": valid_recall_queries
        },
        "rag": {
            "total": {
                "p50_ms": float(t_p50),
                "p70_ms": float(t_p70),
                "p100_ms": float(t_p100)
            },
            "generation": {
                "p50_ms": float(g_p50),
                "p70_ms": float(g_p70),
                "p100_ms": float(g_p100)
            },
            "relevance_passed_count": relevance_passed_count,
            "total_run": num_rag
        }
    }
    
    os.makedirs(os.path.dirname(results_output_file), exist_ok=True)
    with open(results_output_file, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2)
    print(f"\nEvaluation metrics successfully saved to: {results_output_file}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Voice RAG Pipeline Performance.")
    parser.add_argument("--num-retrieval", type=int, default=100, help="Number of queries for retrieval evaluation.")
    parser.add_argument("--num-rag", type=int, default=15, help="Number of queries for full RAG evaluation.")
    parser.add_argument("--k", type=int, default=5, help="Top-K context documents to retrieve.")
    
    args = parser.parse_args()
    
    run_evaluation(
        num_retrieval=args.num_retrieval,
        num_rag=args.num_rag,
        k=args.k
    )
