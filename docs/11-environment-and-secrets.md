# 11 — Environment & Secrets

This document explains the environment variables required to run the Voice-Enabled RAG Portal.

---

## 📋 Environment Configuration Template
To configure the application, create a file named `.env` in the root folder using the template below:

```env
# 🔑 API Access Tokens
GROQ_API_KEY="gsk_..."
ELEVENLABS_API_KEY="el_..."

# ⚙️ Search Configuration
EMBEDDING_MODEL="paraphrase-multilingual-MiniLM-L12-v2"
RETRIEVAL_MODE="sparse" # "dense" (FAISS/BM25) or "sparse" (BM25 only)
RAG_RELEVANCE_THRESHOLD="0.45"
```

---

## ⚙️ Variables Explanation

### 1. `GROQ_API_KEY`
*   **Purpose**: Authentication key for Groq API.
*   **Role**: Used to generate conversational RAG responses using the Llama-3 model in Generative Mode (LLM ON).

### 2. `ELEVENLABS_API_KEY`
*   **Purpose**: Authentication key for ElevenLabs Scribe API.
*   **Role**: Used to transcribe audio WebM files captured by the browser microphone.

### 3. `EMBEDDING_MODEL`
*   **Purpose**: Specifies the sentence transformer embedding model to use.
*   **Default**: `paraphrase-multilingual-MiniLM-L12-v2` (maps 14 languages to a shared 384-dimensional vector space).

### 4. `RETRIEVAL_MODE`
*   **Purpose**: Controls startup model loading.
*   **Values**:
    *   `sparse`: Bypasses PyTorch, loading only lightweight JSON index maps (runs under 45MB RAM for Render deployment).
    *   `dense`: Instantiates the full neural retrieval service, supporting local hybrid vector matching.

### 5. `RAG_RELEVANCE_THRESHOLD`
*   **Purpose**: Sets similarity confidence bounds.
*   **Default**: `0.45`
*   **Role**: Filters out off-topic searches. Queries with match scores below `0.45` are blocked from contacting the LLM and return localized fallback text.
