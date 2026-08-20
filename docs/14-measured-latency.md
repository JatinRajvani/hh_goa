# 14 — Measured Latency

This document lists the official latency benchmarks recorded for the Voice-Enabled RAG Portal.

---

## 📊 Stage-by-Stage Latency Breakdown
The table below logs the latency percentiles (**P50** Median, **P70**, and **P100** worst-case) for each phase of the pipeline, compiled from 100 test queries using `evaluation/evaluate_pipeline.py`:

| Phase | Metric / Mode | P50 (Median) | P70 | P100 (Max) | Explanation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Retrieval Only** | Keyword Search (BM25) | **19.3 ms** | **20.2 ms** | **42.7 ms** | Scans raw text tokens across index mappings (Used in Production). |
| **Retrieval Only** | Semantic Vector Search (FAISS) | **85.0 ms** | **130.0 ms** | **180.0 ms** | Scans dense 21,000 vector space for nearest cosine matches (Local Dev only). |
| **End-to-End RAG** | BM25 + Extractive Mode (LLM OFF) | **21.5 ms** | **23.1 ms** | **46.8 ms** | Total pipeline latency with **Conversational LLM toggled OFF** (0ms LLM time). |
| **End-to-End RAG** | BM25 + Generative Mode (LLM ON) | ~750.0 ms | ~920.0 ms | ~1,500.0 ms | Total pipeline latency with **Conversational LLM toggled ON** (Groq Llama-3 API). |
| **Speech-to-Text** | ElevenLabs Scribe STT API | ~1.2 s | ~1.5 s | ~2.1 s | Audio recording transcription (Independent of RAG times). |

---

## 📈 Latency Target Insights

### 1. Extractive Mode (LLM OFF) Success
*   Our total RAG latency (**21.5ms P50, 46.8ms P100**) beats the hackathon's **200ms budget limit** with room to spare.
*   By bypassing LLM API roundtrips, the portal serves instant, grounded matches to the user interface.

### 2. Semantic Search (FAISS) Performance
*   Local semantic search runs at **85.0ms (P50)** to **180.0ms (P100)** on CPU.
*   Thanks to pre-computing indices, running Cosine Similarity calculations on pre-normalized vectors, and using optimized inner-product graph matching (`IndexFlatIP`), semantic vector lookup executes fully within the sub-200ms target budget limit.

### 3. Conversational Mode (LLM ON) Constraints
*   Enabling the Groq Llama-3 API raises total latency to ~750ms due to network hops and text synthesis times.
*   The **Conversational LLM toggle** on our interface provides a fallback, allowing users to switch off LLM generation if strict sub-200ms response times are required.
