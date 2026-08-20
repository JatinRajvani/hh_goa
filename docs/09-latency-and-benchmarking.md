# 09 — Latency & Benchmarking

This document details our latency testing setup, satisfying **Requirement 3 (Latency Target)** and **Requirement 4 (Latency Analytics)**.

---

## 🔬 Latency Profiling Setup
The benchmark utility is managed by the script `evaluation/evaluate_pipeline.py`. This script:
1.  Loads 100 test search queries across English and regional Indic scripts.
2.  Runs the queries through the backend retrieval service in sequence.
3.  Records the query latencies for both BM25 Keyword Search and FAISS Semantic Search.
4.  Evaluates end-to-end extractive RAG pipelines.
5.  Calculates latency percentiles: **P50 (Median)**, **P70**, and **P100 (Max)**.

---

## 🏃 Running the Evaluation Benchmarks

To execute the profiling suite locally:
```bash
# 1. Activate environment
.venv\Scripts\Activate.ps1

# 2. Run the evaluation suite
python evaluation/evaluate_pipeline.py
```

### Script Execution Flow:
```
  [ Start Evaluation ]
           |
  [ Load 100 Test Queries ]
           |
  [ Loop: Run Search Queries ]
           |
  [ Compute P50, P70, P100 ]
           |
  [ Output Performance Summary JSON ]
```
The metrics are saved to `data/processed/evaluation_results.json` and plotted inside the browser dashboard.

---

## 🎯 Target Thresholds vs. Deployed Performance
*   **Hackathon Requirement**: RAG pipelines must execute in **under 200ms**.
*   **Extractive Mode (LLM OFF)**: The portal matches context and returns results in **21.5ms (P50)** and **46.8ms (P100)**, fully satisfying the requirement.
*   **Local Semantic Search**: FAISS semantic retrieval matches passages in **85.0ms (P50)**, completing vector encodings and calculations within the sub-200ms budget limit.
