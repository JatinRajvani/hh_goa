# 12 — Milestones & Roadmap

This document outlines the engineering journey and milestones completed during the development of our Voice-Enabled RAG Portal.

---

## 🗺️ Project Timeline & Milestones

```mermaid
gantt
    title Development Timeline
    dateFormat  YYYY-MM-DD
    section Backend Core
    Ingestion & Chunking            :done,    des1, 2026-08-01, 2026-08-04
    Dual Search Indexing            :done,    des2, 2026-08-05, 2026-08-07
    STT & Script Detection          :done,    des3, 2026-08-08, 2026-08-11
    section API & UI
    API Harness & Relevance Guard   :done,    des4, 2026-08-12, 2026-08-15
    UI Development & Latency Charts  :done,    des5, 2026-08-16, 2026-08-18
    Production Optimizations        :done,    des6, 2026-08-19, 2026-08-20
```

---

## 🎯 Completed Milestones

### 📍 Milestone 1: Ingestion & Sentence-Aware Chunking
*   Established parquet dataset streaming from Hugging Face for the `ai4bharat/MSMARCO-XI` dataset.
*   Designed regex sentence-boundary chunking, achieving a **11.7% Recall@5 boost** over raw baseline passages.

### 📍 Milestone 2: Dual Search Indexing
*   Built and compiled indices for 14 languages (21,000 passages).
*   Integrated Okapi BM25 keyword matching and FAISS vector matching using the `paraphrase-multilingual-MiniLM-L12-v2` embedding model.

### 📍 Milestone 3: Speech-to-Text & Script Detection
*   Connected the ElevenLabs Scribe STT API to handle audio files in WebM format.
*   Engineered a fast dual-layer script detector (Unicode script block mapping in **0.01ms** + `langdetect` fallback).

### 📍 Milestone 4: API Harness & Relevance Guardrails
*   Exposed FastAPI endpoints for text and voice query submissions.
*   Enforced a strict **`0.45` relevance threshold** and integrated fallback response maps in 14 scripts.

### 📍 Milestone 5: UI Development & Memory Optimizations
*   Created a Glassmorphic browser dashboard displaying latency benchmarks.
*   Separated dev requirements (`faiss`, `sentence-transformers`) from production dependencies (`requirements.txt`), allowing the portal to boot under a lightweight **45MB RAM footprint** on Render Free Cloud Tier.
