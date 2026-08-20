# Latency Analytics & Cloud Memory Optimization

This document outlines the latency benchmarks of our RAG pipeline, covering **Requirement 3 (Latency Target)** and **Requirement 4 (Latency Analytics)**, and details the memory optimizations implemented to run within Render's strict 512MB RAM limit.

---

## 1. Pipeline Latency Analytics

To measure latency under typical and worst-case loads, we evaluated the system across 100 retrieval test queries using our evaluation runner script [`evaluate_pipeline.py`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/evaluation/evaluate_pipeline.py). 

### A. Latency Results Summary Table

| Phase | Metric / Mode | P50 (Median) | P70 | P100 (Max) |
| :--- | :--- | :---: | :---: | :---: |
| **Retrieval Only** | Keyword Search (BM25) | **19.3 ms** | **20.2 ms** | **42.7 ms** |
| **Retrieval Only** | Semantic Vector Search (FAISS) | **~163.0 ms** | **~300.0 ms** | **~450.0 ms** |
| **End-to-End RAG** | BM25 + Extractive Mode (LLM OFF) | **21.5 ms** | **23.1 ms** | **46.8 ms** |
| **End-to-End RAG** | BM25 + Generative Mode (LLM ON) | ~750.0 ms | ~920.0 ms | ~1,500.0 ms |
| **Speech-to-Text** | ElevenLabs Scribe STT API | ~1.2 s | ~1.5 s | ~2.1 s |

### B. Latency Target Analysis (Sub-200ms Target)
* **Extractive Mode (LLM OFF)**: The full pipeline (BM25 retrieval + formatting) executes in **21.5ms (P50)**, comfortably beating the Hackathon's **200ms latency budget**.
* **Generative Mode (LLM ON)**: Adding the LLM (Groq Llama-3) increases latency to ~750ms due to network transit and generation tokens. The Extractive Toggle provides a critical fallback for situations where sub-200ms response time is strictly required.
* **Vector Search Latency (Above 200ms)**: Local semantic search has a ~163ms (P50) to ~450ms (P100) overhead. This goes above the 200ms target in P70/P100 runs because we loaded all 14 languages, creating a large local index of **21,000 passages**. Encoding the query text into a vector using the MiniLM model plus running a mathematical similarity comparison across all 21,000 vectors on a local CPU introduces significant processing overhead compared to BM25's lightweight token scanning.

### C. Chunk Count vs. Passage Count
* **Are the number of chunks and passages the same?**
  * **Yes**. In our final ingestion pipeline ([`build_multilingual_indices.py`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/ingestion/build_multilingual_indices.py)), we process 1,500 passages per language. Because the passages in the Hugging Face `MSMARCO-XI` dataset are already short (averaging 50–100 words), we do not split them further.
  * Therefore, **the number of chunks is exactly equal to the number of passages (21,000 passages = 21,000 chunks)**, ensuring that we preserve the complete context of each passage in a single index key.

---

## 2. Render Cloud Memory Optimizations

Deploying ML-driven RAG pipelines onto free cloud hosts like Render comes with a massive constraint: a **512MB RAM limit**. Loading standard ML binaries directly causes immediate Out-Of-Memory (OOM) server crashes. We bypassed this using three key techniques:

### A. Requirements Separation & Stripping
We split our project requirements into two configuration files:
1. **`requirements.txt` (Production/Render)**: Stripped of all heavy packages (`faiss-cpu`, `sentence-transformers`, `datasets`, `torch`). It only contains lightweight network-bound and API packages, reducing Render compilation and build times to **under 15 seconds**.
2. **`requirements-dev.txt` (Local Development)**: Contains the full local package list for developers who want to run the dense vector model and build indices locally.

### B. Lazy Imports (Bypassing Heavy Binaries)
By default, importing `faiss` or `sentence-transformers` automatically loads their compiled C++ binaries and model weights, immediately consuming over 450MB of RAM. 

We moved all heavy imports inside localized search methods:
```python
def load_dense_model_on_demand(self):
    if not self.model:
        # Loaded ONLY if dense search is explicitly toggled by user on localhost
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)
```
In production (`RETRIEVAL_MODE=sparse`), python **never** reaches this block, keeping our production RAM footprint at an extremely low **~45MB** (saving ~90% memory).

### C. Parquet Ingestion Offline
We perform all dataset extraction and indexing **offline**. Parquet downloads, sentence chunking, vector indexing, and JSON mappings are built once using [`ingestion/build_multilingual_indices.py`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/ingestion/build_multilingual_indices.py) and committed to git, meaning the online server only needs to read static mappings.
