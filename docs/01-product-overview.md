# 01 — Product Overview

## Introduction
The **Voice-Enabled Multilingual RAG Portal** is a high-performance, production-ready system designed for the **Hackathon (Goa 2026 - Task 2)**. The application allows users to speak or type search queries in **14 different languages** (English and 13 native Indic languages), transcribes the audio, automatically detects the query language/script, retrieves relevant context from our pre-processed dataset, and delivers grounded answers either as instant extracts or LLM-synthesized responses.

---

## 🎯 Target Goal
The primary objective of the portal is to serve accurate, relevant search results and grounded answers under a strict latency budget:
*   **Target Latency**: The end-to-end retrieval process (excluding external voice transcription APIs) must execute in **under 200ms**.
*   **Robust Accuracy**: Prevent hallucinations and off-topic queries by applying strict similarity thresholds and structured grounding prompts.
*   **Cloud Resilience**: Deploy and run reliably in production on Render's Free Cloud Tier within a strict **512MB RAM constraint**.

---

## 🎙️ Core Product Features
1.  **Voice-First Interface**: Press a microphone button, record audio in any supported language, and receive instant transcribed search queries via the ElevenLabs Scribe STT API.
2.  **Multilingual Support**: Fully supports English and 13 native Indic scripts:
    *   *Hindi (hi), Gujarati (gu), Tamil (ta), Kannada (kn), Malayalam (ml), Punjabi (pa), Marathi (mr), Bengali (bn), Telugu (te), Odia (or), Urdu (ur), Sanskrit (sa), Assamese (as), Nepali (ne)*.
3.  **Dual Search Engines**:
    *   **Keyword Search (BM25)**: Instant, ultra-lightweight term-frequency lookup (Default & locked on cloud servers to run under 45MB RAM).
    *   **Semantic Search (FAISS)**: Near-instant vector similarity search using multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) for conceptual and cross-lingual matches (Available in local environments).
4.  **Flexible LLM Modes**:
    *   **Fast Extractive Mode (LLM OFF)**: Bypasses the LLM completely to serve raw matched text chunks in under **30ms** (guaranteeing sub-200ms performance).
    *   **Generative Mode (LLM ON)**: Fuses Groq Llama-3 API to summarize the retrieved passages into a natural, conversational response.
5.  **Relevance & Grounding Guardrails**: A hard threshold (`0.45` similarity confidence) protects the system. Queries falling below this score are blocked, and a localized "no information found" fallback is returned.
6.  **Interactive Latency Analytics**: Real-time charts in the frontend display precise millisecond benchmarks for every pipeline phase (STT, Detection, Search, and LLM Generation).
