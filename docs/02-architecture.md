# 02 — Architecture

This document describes the end-to-end subsystem layout and data flow.

---

##  UFO Sequence Diagram
The sequence diagram below visualizes how a voice-initiated request moves through the system:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Browser (HTML/JS)
    participant API as FastAPI Backend
    participant STT as ElevenLabs Scribe STT
    participant Det as Language Detector
    participant DB as Retrieval Engine (BM25/FAISS)
    participant LLM as Groq LLM API

    User->>UI: Press microphone & record question
    UI->>API: Send WebM Audio Binary (/api/query-voice)
    API->>STT: Stream Audio Payload
    STT-->>API: Return Transcribed Text Script
    API->>Det: Run Auto-Language Detection
    Det-->>API: Match Script Code (en, hi, gu, ta, etc.)
    API->>DB: Query Selected Index (Top-K Chunks)
    DB-->>API: Return Passages + Match Scores
    
    alt Highest Match Score < 0.45 (Guardrail Triggered)
        API-->>UI: Return Localized Fallback Answer ("No sufficient info found")
    else Highest Match Score >= 0.45 (Passed Guardrail)
        alt Conversational LLM Mode is OFF
            API-->>UI: Return Highest Ranked Extractive Chunk (0ms LLM Latency)
        else Conversational LLM Mode is ON
            API->>LLM: Send Grounded Prompt (Query + Sources + Rules)
            LLM-->>API: Return Synthesized Conversational Answer
            API-->>UI: Return Answer + Source Metadata
        end
    end
    API->>UI: Return Detailed Phase Latency Measurements
    UI->>User: Display Answer, Sources, & Plot Latency Charts
```

---

## 🧱 Component Breakdown

### 1. Frontend Web App (Browser client)
*   **Technologies**: HTML5, Vanilla CSS (Glassmorphic theme), JavaScript.
*   **Recording**: Captures mic audio via `MediaRecorder` API into a **WebM** buffer stream.
*   **Latency Monitoring**: Logs timestamps for API roundtrips and visualizes sub-millisecond stage latencies in a latency dashboard (using dynamic status indicators).

### 2. FastAPI Orchestration Server
*   **Endpoint `/api/query-text`**: Serves text-based searches.
*   **Endpoint `/api/query-voice`**: Receives voice files, routes them to STT, detects query languages, runs retrieval searches, applies guardrails, invokes the LLM (if selected), and aggregates latencies.

### 3. Speech-to-Text (STT) Service
*   **Integration**: ElevenLabs Scribe API.
*   **Script Handling**: Recovers transcribing output directly in native scripts (e.g. Gujarati script for Gujarati audio), bypassing translating steps prior to retrieval.

### 4. Language Auto-Detector
*   **Layer 1 (Unicode Script Bounds)**: Matches character ranges to map languages in **0.01ms** (e.g. Tamil range `0x0b80` to `0x0bff` maps directly to `ta`).
*   **Layer 2 (langdetect Fallback)**: Resolves shared Devanagari range scripts (Hindi, Marathi, Sanskrit, Nepali) and English.

### 5. Dual Retrieval Engine
*   **Okapi BM25 Index**: Pure Python text token-scanning. Runs in cloud production at low memory.
*   **FAISS Vector Index**: Dense vector database executing Cosine Similarity nearest-neighbor search. Utilizes `paraphrase-multilingual-MiniLM-L12-v2`.

### 6. Grounding Guardrail
*   Rejects matches with scores below `0.45` confidence, blocking downstream LLM calls and serving localized fallback messages to prevent hallucinations.
