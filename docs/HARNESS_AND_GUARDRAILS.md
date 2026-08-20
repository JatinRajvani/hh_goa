# Voice RAG Harness, Language Detection, & Grounding Guardrails

This document describes the structured orchestration, voice-to-text integration, language auto-detection, and accuracy guardrails that protect the pipeline from hallucinations and off-topic queries. This covers **Requirement 1 (STT)**, **Requirement 5 (Harness)**, and **Requirement 6 (Guardrails)**.

---

## 1. Speech-to-Text (STT) Integration & Audio Harness

We integrated the **ElevenLabs Scribe API** to perform multilingual speech transcription.

```
 [ Browser Microphone ] --- (WebM Stream) ---> [ FastAPI Server ]
                                                    |
                                           [ Temp Audio File ]
                                                    |
 [ Transcribed Script ] <-- (JSON Response) -- [ Scribe STT API ]
```

### The Audio Flow:
1. **Frontend Capture**: The client browser records microphone input using the HTML5 `MediaRecorder` API, compressing audio into standard **WebM** container format.
2. **FastAPI Route**: The `/api/query-voice` endpoint receives the multipart file upload and saves it to a secure, temporary path.
3. **STT Invocation**: The [`STTService`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/retrieval/stt_service.py) reads the file stream and calls ElevenLabs Scribe.
   - If the user selects a specific language, we pass the corresponding ISO language code to restrict Scribe's script decoder.
   - If set to **Auto-Detect**, Scribe transcribes the audio in its native script directly, handling code-switching.
4. **Error Recovery & Cleanup**: To prevent memory leaks, we wrap the transaction in a `try...finally` block that guarantees the temporary audio file is **deleted from disk** upon completion or failure.

---

## 2. Dual-Layer Language Auto-Detection

To retrieve context in the correct language script (e.g. Gujarati vs. Marathi indices), the pipeline must detect the script instantly. We built a high-speed, dual-layer detector in [`RAGOrchestrator.detect_language`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/retrieval/rag_orchestrator.py):

### Layer 1: Unicode Script Block Match (Latency: ~0.01 ms)
Unique Indian scripts use dedicated Unicode ranges. We instantly scan the characters of the query:
* **Gujarati (`gu`)**: `0x0a80` to `0x0aff`
* **Tamil (`ta`)**: `0x0b80` to `0x0bff`
* **Kannada (`kn`)**: `0x0c80` to `0x0cff`
* **Malayalam (`ml`)**: `0x0d00` to `0x0d7f`
* **Punjabi (`pa`)**: `0x0a00` to `0x0a7f`
* **Odia (`or`)**: `0x0b00` to `0x0b7f`
* **Urdu (`ur`)**: `0x0600` to `0x06ff`
* **Bengali/Assamese (`bn`/`as`)**: `0x0980` to `0x09ff`

### Layer 2: Frequency Fallback (`langdetect`)
Some languages share script sets (e.g., Hindi, Marathi, Nepali, and Sanskrit all share the **Devanagari** script range `0x0900` to `0x097f`). 
For these scripts and Latin text (English), we run character-frequency mapping via the `langdetect` library to resolve the specific language code.

---

## 3. Pipeline Harness & Error Handling

To satisfy **Requirement 5 (Harness)**, the [`RAGOrchestrator`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/retrieval/rag_orchestrator.py) acts as a strict state machine. It prevents raw LLM prompt-in/text-out failures through structured stages:

```
  [ Query Input ] ---> [ Language Detection ]
                             |
                   [ Index Search (Top-K) ]
                             |
             /--- [ Relevance Guardrail Check ] ---\
           (Pass)                                 (Fail)
             |                                      |
     [ LLM Grounding ]                     [ Localized Fallback ]
             |                                      |
 [ Conversational Answer ]                 [ "No Info Found" Msg ]
```

* **Inputs & Outputs**: Strictly typed schemas using `Pydantic` models for queries, parameter overrides, and returning structured JSON containing search latencies, script codes, relevance scores, source documents, and answers.
* **API Resiliency**: Groq API calls are wrapped in robust error catch blocks. If the LLM generates a timeout or API limits are hit, the orchestrator defaults gracefully to returning the raw top-ranked extractive text chunk, ensuring the end-user always receives a helpful response.

---

## 4. Grounding & Relevance Guardrails

To meet **Requirement 6 (Guardrail your model)**, we protect the RAG pipeline from answering off-topic queries or hallucinating facts:

### A. Relevance Score Guardrail (Threshold: `0.45`)
When a query returns search results, we inspect the highest-ranked chunk's similarity score:
* **Score $\ge 0.45$**: Search results are considered relevant. The pipeline continues to answer generation.
* **Score $< 0.45$**: The query is marked as out-of-domain or off-topic. LLM execution is blocked to save API tokens and prevent hallucinations.
* **Action**: The system responds with a **Localized Fallback Message** matching the target language.

#### Localized Guardrail Fallbacks:
* **English (`en`)**: *"I couldn't find sufficient information..."*
* **Hindi (`hi`)**: *"मुझे प्रदान की गई जानकारी में इसका उत्तर नहीं मिला।"*
* **Gujarati (`gu`)**: *"મને પ્રદાન કરેલી માહિતીમાં આનો ઉત્તર મળ્યો નથી."*
* **Tamil (`ta`)**: *"வழங்கப்பட்ட அறிவுத் தளத்தில் பதிலளிக்க போதுமான தகவல் கிடைக்கவில்லை."*
*(Mappings are implemented for all 14 languages in [`RAGOrchestrator.py`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/retrieval/rag_orchestrator.py)).*

### B. LLM Prompt Grounding Constraints
If relevance passes and the user enables **Conversational LLM**, we wrap the query and context inside strict grounding instructions:
1. **Source Grounding**: Answer in the query script using **ONLY** the facts explicitly supported by the provided sources.
2. **No Speculation**: Do NOT extrapolate, speculate, or utilize outside knowledge.
3. **Uncertainty Enforcement**: If the sources are unrelated, the LLM is instructed to reply with the exact localized fallback text.
4. **Reasoning Flexibility**: The LLM is permitted to bridge technical/layman terms (e.g. medical synonyms) if supported semantically, but cannot introduce new claims.
