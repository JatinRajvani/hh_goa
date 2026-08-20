# 10 — Project Structure

This document outlines the directory tree and file organization of the Voice-Enabled RAG Portal.

---

## 📂 Directory Layout

```
hh-goa/
├── backend/                       # FastAPI web server
│   ├── app.py                     # Main FastAPI endpoints and API entry point
├── data/                          # Serialized data indices
│   ├── index/                     # Compiled BM25 map files & FAISS indexes
│   ├── processed/                 # Evaluation inputs and test query datasets
├── docs/                          # Specialized technical guides (01-15)
├── evaluation/                    # Benchmarking suite
│   ├── evaluate_pipeline.py       # Latency percentile test runner
├── frontend/                      # Glassmorphic user interface
│   ├── index.html                 # Main dashboard layout
│   ├── index.css                  # UI styling sheet
│   ├── app.js                     # Browser controller and audio recorder logic
├── ingestion/                     # Ingestion scripts
│   ├── build_multilingual_indices.py  # Streams HF parquet and builds search maps
│   ├── chunking_experiments.py    # Chunking comparison test suite
├── retrieval/                     # Core search logic
│   ├── search.py                  # BM25/FAISS local search engine
│   ├── stt_service.py             # ElevenLabs speech-to-text bridge
│   ├── rag_orchestrator.py        # Language detection & prompt grounding loop
├── requirements.txt               # Production requirements (stripped)
├── requirements-dev.txt           # Development requirements (full models)
├── .env.example                   # Template env file
└── README.md                      # Product submission guide
```

---

## 🧱 Key Components Overview
*   **`backend/app.py`**: Boots the Uvicorn server, exposes endpoints for query text and voice files, and manages CORS policies for local frontend execution.
*   **`retrieval/search.py`**: Contains the retrieval logic. Implements `RetrievalService` which dynamically lazy-loads PyTorch sentence models on localhost queries and executes vector distance checks.
*   **`retrieval/rag_orchestrator.py`**: Executes the core RAG logic. Performs character Unicode range script matches for script detection, enforces grounding rules, and calls the Groq LLM API.
*   **`frontend/app.js`**: Controls the user interface. Manages the audio recording lifecycle, visualizes pipeline latency telemetry charts, and implements security lockouts to disable semantic vector dropdowns on cloud deployments.
