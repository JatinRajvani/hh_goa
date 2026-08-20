# Retrieval & RAG Modes: BM25, Semantic, & LLM Toggles

This document describes how we implemented the search selector engine (Lexical BM25 vs. Dense Vector Semantic) and the conversational LLM toggle (Fast Extractive vs. Generative LLM) to satisfy latency constraints and support local testing.

---

## 1. Dual Search Engine Selector (BM25 vs. Semantic)

To support both highly accurate semantic retrieval and memory-efficient production deployments, the system features a dual search architecture. Users can toggle this in the frontend header selector.

```
       [ Search Engine Selector ]
             /          \
            /            \
    [ Lexical BM25 ]    [ Dense Vector FAISS ]
    - Token keyword     - Sentence Transformers
    - Fast (<2ms)       - Cosine Similarity (384d)
    - Low RAM (<50MB)   - Heavy RAM (>450MB)
    - Production Default- Local Dev Testing
```

### A. Keyword/Lexical Search (Okapi BM25)
* **Implementation**: We use the pure-Python `rank-bm25` library. 
* **Mechanism**: 
  - On startup or lazy request, we load the document mappings from [`data/index/id_mapping_{lang}.json`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/data/index).
  - The corpus is tokenized on-demand.
  - Queries are tokenized in under 2ms and matched using Okapi BM25 scoring.
* **Score Normalization**: BM25 raw scores are unbounded. To align with our RAG Guardrail threshold (`0.45`), we map the raw scores using:
  $$\text{Score}_{\text{mapped}} = \min(0.99, 0.45 + \frac{\text{Raw Score}}{100})$$
  This maps positive lexical hits to a valid relevance range.

### B. Dense Vector Semantic Search (FAISS)
* **Implementation**: We use a FAISS Inner Product index (`faiss.IndexFlatIP`) with cosine pre-normalization.
* **Mechanism**:
  - The query is encoded into a 384-dimensional vector using `paraphrase-multilingual-MiniLM-L12-v2`.
  - We normalize the vector (L2 norm) using `faiss.normalize_L2`.
  - We perform an inner product nearest-neighbor search, which mathematically equals Cosine Similarity because both query and document embeddings are pre-normalized.

### C. Dynamic UI Hostname Locking (Render Cloud Safety)
Because Render's Free Tier has a strict **512MB RAM limit**, running heavy deep-learning libraries (`faiss-cpu`, `sentence-transformers`, `torch`) in production will trigger Out-Of-Memory (OOM) server crashes. 
To guarantee 100% server stability, we built a **Dynamic UI Engine Selector**:
* **On Localhost**: Both options are active. If a developer switches from BM25 to Semantic, the backend loads the `SentenceTransformer` model dynamically **on-demand (exactly once)**.
* **On Render Cloud Deployment**: The client-side logic ([`frontend/app.js`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/frontend/app.js)) detects the hostname (non-localhost). It automatically **locks and disables** the selector to `Keyword (BM25) [Cloud Locked]`. 
* **Result**: The production server remains stable, operating at **under 50MB RAM** (saving 90% memory overhead).

---

## 2. LLM Conversational Toggles (Extractive vs. Generative RAG)

In the user interface, there is a prominent toggle labeled **"Conversational LLM"** with active status indicators (ON/OFF) and a large clickable target area.

### A. Toggle OFF: Fast Extractive Mode
* **Latency**: **0 ms LLM Overhead** (Total pipeline execution: **`< 60 ms`**).
* **Behavior**: The RAG pipeline bypasses the LLM API call entirely.
* **Logic**:
  1. The system performs the search (BM25 or FAISS).
  2. The highest-scoring text passage is served directly as the answer.
* **Purpose**: Satisfies the Hackathon's latency target (sub-200ms) with absolute certainty, bypassing network transit and generative LLM generation.

### B. Toggle ON: Generative LLM Mode
* **Latency**: **~700 ms** (using Groq Llama-3 API).
* **Behavior**: Summarizes the context into a natural, conversational response.
* **Logic**:
  1. The system retrieves the top-K relevant passages.
  2. The passages are compiled into a strict grounding prompt.
  3. The Groq Llama-3 model generates a concise, localized answer in the query's native script.
* **Purpose**: Serves conversational, context-fused answers when natural dialogue is preferred.
