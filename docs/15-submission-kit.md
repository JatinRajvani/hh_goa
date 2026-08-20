# 15 — Submission Kit

This document compiles the submission details, hosting links, and instructions for evaluating the Voice-Enabled RAG Portal.

---

## 🔗 Submission Details
*   **Live Working Portal**: [hh-goa.onrender.com](https://hh-goa.onrender.com)
    *   *First load note*: Render's Free tier spins down web containers during idle periods. The first load after dormancy may take ~30 seconds.
*   **Video 1 (Detailed Technical Deep-Dive)**: [youtu.be/4FzrUFFhS18](https://youtu.be/4FzrUFFhS18)
*   **Video 2 (Product Demo Video)**: [youtu.be/IcSiq11f61E](https://youtu.be/IcSiq11f61E)
*   **Tag / Hashtag**: `#RAGInGoa` (Submission Deadline: August 22, 2026)

---

## 🏃 Evaluation Quick Start (Local Run)

### 1. Configure Credentials
Create a `.env` file in the root directory:
```env
GROQ_API_KEY="your_groq_api_key_here"
ELEVENLABS_API_KEY="your_elevenlabs_api_key_here"
```

### 2. Launch Local Hybrid Portal (FAISS + BM25)
To test side-by-side search performance, install development dependencies and boot the portal:
```bash
# Install development requirements
pip install -r requirements-dev.txt

# Run dataset ingestion
python ingestion/build_multilingual_indices.py

# Boot the web portal
python backend/app.py
```
Open **`http://127.0.0.1:8000/`** to interact with the glassmorphic portal.

### 3. Evaluate Latencies
Execute the latency benchmarking suite:
```bash
python evaluation/evaluate_pipeline.py
```
This runs 100 retrieval checks, displaying performance summaries and logging latency metrics to `data/processed/evaluation_results.json`.
