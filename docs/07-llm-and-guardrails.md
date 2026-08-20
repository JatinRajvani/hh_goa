# 07 — LLM & Guardrails

This document describes our LLM generation pipeline and the safety guardrails protecting system reliability, covering **Requirement 6 (Guardrails)** of the hackathon specifications.

---

## 🛠️ Conversational LLM Integration
We leverage the **Groq Llama-3 API** to generate high-speed conversational responses. The backend dynamically supports two query modes:

### A. Fast Extractive Mode (LLM OFF)
*   **Latency**: **0 ms LLM Overhead** (Total response time under **30 ms**).
*   **Behavior**: Bypasses the LLM entirely. The highest-scoring text passage retrieved from index matches is served directly as the answer.
*   **Benefit**: Absolute guarantee of meeting sub-200ms latency limits in production.

### B. Generative Mode (LLM ON)
*   **Latency**: **~700 ms** (network transit and token generation).
*   **Behavior**: Fuses the retrieved passages into a grounded context, prompting Llama-3 to generate a natural, conversational response in the matching language script.

---

## 🛡️ Relevance Guardrails (Threshold: `0.45`)
To prevent the model from answering off-topic questions or generating hallucinations, the orchestrator applies a strict **Relevance Guardrail**:

```
  [ Index Search (Top-K) ]
             |
   (Highest Match Score)
             |
      /--------------\
     |  Score >= 0.45 |
      \--------------/
        /          \
    (Yes)          (No)
      /              \
 [ Run RAG ]    [ Block LLM & Return Localized Fallback ]
```

*   If the top search result score is below **`0.45`**, downstream LLM execution is blocked to save API tokens and avoid hallucinated fabrications.
*   The system immediately responds with a **Localized Fallback Message** in the query's script.

### Localized Fallback Messages:
*   **English (`en`)**: *"I couldn't find sufficient information in the dataset to answer your question."*
*   **Hindi (`hi`)**: *"मुझे प्रदान की गई जानकारी में इसका उत्तर नहीं मिला।"*
*   **Gujarati (`gu`)**: *"મને પ્રદાન કરેલી માહિતીમાં આનો ઉત્તર મળ્યો નથી."*
*   **Tamil (`ta`)**: *"வழங்கப்பட்ட அறிவுத் தளத்தில் பதிலளிக்க போதுமான தகவல் கிடைக்கவில்லை."*

---

## 📜 Grounding Prompt Rules
When Generative Mode is active, we wrap the user's query and source passages inside strict grounding instructions:
1.  **Source Constraint**: Answer the question using **ONLY** the facts explicitly mentioned in the provided context.
2.  **No Speculation**: If the context does not contain the answer, reply with the exact localized fallback text.
3.  **Strict Language Match**: Generate the answer using the same script block as the query text.
4.  **No Extrapolations**: Do not leverage outside knowledge or facts.
