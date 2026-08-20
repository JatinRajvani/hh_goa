# Local Setup & Execution Guide

This document provides step-by-step instructions for running the Voice-Enabled RAG Portal on your local machine. It details the steps for configuring the system in **BM25 Keyword-Only Mode** (lightweight) or **FAISS Semantic + BM25 Hybrid Mode** (full dense vector).

---

## 📋 Prerequisites
* **Python**: Python 3.10 or 3.11 is recommended.
* **FFmpeg**: Required if you plan to convert or test audio speech transcripts locally.
* **API Keys**: 
  * A [Groq API Key](https://console.groq.com/) (for generating answers via Llama-3).
  * An [ElevenLabs API Key](https://elevenlabs.io/app/sign-up) (for transcribing voice inputs).

---

## 🛠️ Step 1: Clone & Setup Environment

1. Clone the repository and navigate to the project directory:
   ```bash
   cd hh-goa
   ```
2. Create and activate a Python virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Create your `.env` file in the root directory:
   ```env
   GROQ_API_KEY="your_groq_api_key_here"
   ELEVENLABS_API_KEY="your_elevenlabs_api_key_here"
   EMBEDDING_MODEL="paraphrase-multilingual-MiniLM-L12-v2"
   RAG_RELEVANCE_THRESHOLD="0.45"
   ```

---

## 📦 Step 2: Choose Your Running Mode & Install Dependencies

You can configure the portal to run in one of two modes:

### Option A: BM25 Keyword-Only Mode (Lightweight)
Use this option to test lexical retrieval quickly without loading heavy neural network weights or installing compiled C/C++ libraries. This simulates our Render production setup.
* **RAM Footprint**: < 50 MB
* **Dependencies**: No PyTorch, no HuggingFace downloads, no FAISS compiled binary required.
* **Installation**:
  ```bash
  pip install -r requirements.txt
  ```
* **`.env` Configuration**:
  ```env
  RETRIEVAL_MODE="sparse"
  ```

---

### Option B: FAISS Semantic + BM25 Hybrid Mode (Full)
Use this option to enable high-quality semantic searching, query embeddings, and local FAISS vector matching. This mode allows the frontend selector to switch freely between BM25 and Semantic search.
* **RAM Footprint**: ~450 MB (loads PyTorch weights)
* **Dependencies**: Includes `faiss-cpu`, `sentence-transformers`, and `datasets`.
* **Installation**:
  ```bash
  pip install -r requirements-dev.txt
  ```
* **`.env` Configuration**:
  ```env
  RETRIEVAL_MODE="dense"
  ```

---

## 📥 Step 3: Run Ingestion (Download Data & Build Indices)

Before running the server, you need to ingestion-download the dataset records from Hugging Face (`ai4bharat/MSMARCO-XI`) and compile the search indices:
```bash
python ingestion/build_multilingual_indices.py
```

### What this script does:
1. Downloads the Parquet files for all 14 languages from Hugging Face (cached locally).
2. Extracts unique passages and queries (up to 1,500 passages per language).
3. Saves mapping indices to [`data/index/id_mapping_{lang}.json`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/data/index) (used by both BM25 and FAISS).
4. Generates dense vector embeddings and saves vector files to `data/index/index_{lang}.faiss` (if running Option B).

---

## 💻 Step 4: Start the Web Dashboard Server

Once indexing is complete, start the FastAPI web server:
```bash
# Start via Python app wrapper
python backend/app.py
```
Alternatively, launch directly via Uvicorn:
```bash
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

* Open your browser and navigate to: **`http://127.0.0.1:8000/`**
* You will see the Glassmorphic interface. If running on `localhost`, the Search Engine selector dropdown in the header will be fully unlocked, allowing you to test and compare **Keyword (BM25)** vs. **Semantic (FAISS)** search.

---

## 🖥️ Step 5: Test via Command Line CLI

You can also run tests directly from the terminal without launching the browser interface. The [`retrieval/search.py`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/retrieval/search.py) file features an interactive CLI.

### A. Run a Raw Retrieval Search:
```bash
python retrieval/search.py --query "what is social security disability" --lang en --k 3
```

### B. Run End-to-End RAG (Retrieval + LLM Generation):
```bash
python retrieval/search.py --query "what is social security disability" --lang en --rag
```

### C. Run Voice RAG using an Audio File:
Pass an audio recording to transcribing speech and query the RAG pipeline:
```bash
python retrieval/search.py --voice "development files/bj_food_stamps.mp3" --lang en --rag
```

---

## 📊 Step 6: Run Latency Benchmarking

To measure and record the P50, P70, and P100 latency percentiles across the pipeline:
```bash
python evaluation/evaluate_pipeline.py
```
This runs 100 queries through the retrieval engine and 15 voice requests through the STT+LLM RAG pipeline, saving the performance charts metadata to `data/processed/evaluation_results.json`.
