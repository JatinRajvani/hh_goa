# 08 — Harness & Telemetry

This document details the API schemas, error routing boundaries, and latency logs representing our system test harness, covering **Requirement 5 (Harness)** of the hackathon.

---

## 📋 Structured API Schemas
FastAPI endpoints use typed Pydantic models to validate client requests and format server responses:

### 1. Request Schema (`QueryRequest`)
```python
class QueryRequest(BaseModel):
    query: str
    language: str = "default"  # "default" triggers language auto-detection
    llm_enabled: bool = False
    retrieval_mode: str = "default"  # "default", "dense" (FAISS), or "sparse" (BM25)
```

### 2. Response Schema (`QueryResponse`)
```python
class QueryResponse(BaseModel):
    query: str
    detected_lang: str
    retrieval_mode: str
    answer: str
    sources: List[dict]
    metrics: dict  # Contains detailed millisecond execution timestamps
```

---

## 🛡️ Exception Handlers & Fallbacks
To keep the application responsive under high concurrency or external network disruptions, the backend wraps major service calls in safe fallback handlers:

```
          [ API Request ]
                 |
        [ Execute Pipeline ]
                 |
        /----------------\
       |  Groq API Error  |
        \----------------/
          /            \
       (No)            (Yes)
        /                \
  [ Return LLM ]   [ Fallback: Return Extractive Passage ]
                   [ Log Error & Flag Telemetry ]
```

*   **Groq API Outage Fallback**: If the Groq service fails, times out, or triggers rate-limit exceptions, the orchestrator automatically falls back to **Fast Extractive Mode**, serving the raw, highest-ranked search passage directly.
*   **STT Error Handling**: If ElevenLabs transcription fails, the backend throws clean JSON errors and cleans up any open file sockets.

---

## 📊 Telemetry & Latency Logs
Every response includes a detailed performance telemetry object. This tracks execution time in milliseconds for each segment of the pipeline:

```json
{
  "query": "Who built B Reactor?",
  "detected_lang": "en",
  "retrieval_mode": "sparse",
  "answer": "Hanford's B Reactor was built by the Manhattan Project...",
  "metrics": {
    "stt_ms": 0.0,
    "lang_detect_ms": 0.03,
    "retrieval_ms": 1.45,
    "llm_generation_ms": 0.0,
    "total_rag_ms": 1.48
  }
}
```
This telemetry data allows the frontend client to render visual performance charts in real-time.
