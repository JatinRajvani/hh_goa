# 05 — Retrieval & Qdrant Integration

This document outlines the dual-retrieval engine architecture and highlights how Qdrant can be utilized for production scaling.

---

## 🔍 Retrieval System Comparison

| Attribute | Keyword Search (Okapi BM25) | Semantic Vector Search (FAISS) |
| :--- | :--- | :--- |
| **Matching Mechanism** | Exact term overlap & frequency (TF-IDF based). | Vector distance (Cosine Similarity in 384d space). |
| **Focus** | Matches *exact words* used in the query. | Matches the *meaning and intent* of the query. |
| **Local Startup RAM** | **~2 MB** (extremely lightweight). | **~450 MB** (requires loading PyTorch weights). |
| **Query Latency** | **< 2 ms** (instant lookup). | **~85 ms** (on CPU, with query vector encoding). |
| **Strengths** | Exact matches, product codes, unique names, acronyms. | Synonyms, paraphrasing, translation gaps, Indic language cross-matches. |
| **Weaknesses** | Fails on synonyms, typos, and different phrasings. | Can return false positives for words that are semantically close but logically distinct. |

---

## 🧮 BM25 Score Mapping & Alignment
Okapi BM25 raw scores are unbounded. To align BM25 search outputs with our RAG Relevance Guardrail threshold (`0.45`), we map the raw scores using:
$$\text{Score}_{\text{mapped}} = \min\left(0.99, 0.45 + \frac{\text{Raw Score}}{100}\right)$$
This maps lexical hits to a valid relevance range, allowing consistent guardrail evaluation across both search engines.

---

## 🛠️ Why We Chose FAISS Over Qdrant for Local Portal
1.  **Zero Overhead**: FAISS operates as a local in-process library, avoiding the need to run separate Docker containers or manage network overhead to an external cluster.
2.  **Strict Memory Budgets**: Render's 512MB RAM limit prevents us from spinning up database containers. Loading local indices on-demand maintains a minimal memory footprint.
3.  **Fast Flat Search**: For a dataset of 21,000 passages, a local flat index (`IndexFlatIP`) performs search queries on CPU in under 85ms, rendering complex cluster indexes unnecessary.

---

## 🚀 Production Scaling with Qdrant
For enterprise-level deployment (scaling past millions of multilingual passages), transitioning from local FAISS to a centralized vector search database like **Qdrant** is the recommended path:

### 1. Centralized Index & Clustering
Rather than saving `.faiss` index binaries to local disk (which scales poorly across stateless web containers), vectors are pushed to a centralized Qdrant instance.

### 2. High-Dimensional Indexing (HNSW)
For millions of passages, flat vector search becomes slow. Qdrant implements Hierarchical Navigable Small World (HNSW) graph indexing, keeping search query latencies sub-linear.

### 3. Hybrid Search Fusion
Qdrant supports native hybrid search, combining dense vectors with sparse indices (such as BM25). This allows a single API request to execute term-based and concept-based matches simultaneously, fusing them via Reciprocal Rank Fusion (RRF).

### 4. Dynamic Filtering
Qdrant enables metadata filtering (e.g. `filter={"language": "gu"}`), allowing instant partition search queries without retrieving or scanning foreign language chunks.
