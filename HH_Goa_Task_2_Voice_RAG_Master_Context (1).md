
# HH Goa 2026 — Task 2: Voice-Enabled RAG Model
## Detailed Implementation Context & Master Build Plan

> **Purpose of this document:** This file is a reusable context document for another AI model, developer, or team member. It explains what the HH Goa Task 2 requires, what we are building, the architecture, implementation order, constraints, evaluation plan, and important decisions.
>
> **Primary source:** HH Goa 2026 Shortlisting Task 2 brief provided by the user.
>
> **Important:** Where the official task brief does not specify an implementation detail (for example, the exact embedding model, vector database, LLM, chunk size, backend framework, or frontend framework), treat the recommendation in this document as an engineering recommendation—not as an HH Goa requirement.

---

# 1. Task Overview

HH Goa Task 2 requires building a **voice-enabled Retrieval-Augmented Generation (RAG) system**.

The required high-level flow is:

```text
Voice Input
    ↓
Speech-to-Text
    ↓
Query
    ↓
Chunking / Retrieval using a Vector DB
    ↓
Relevant Context
    ↓
Answer Generation
```

The official task brief describes the system as:

**Voice input → Speech-to-text → Chunking/Retrieval (vector DB) → Answer generation**

The goal is an end-to-end system where a user can speak a question, the system transcribes it, retrieves relevant information from the provided dataset, and generates an answer.

---

# 2. Official Dataset

The task provides the following dataset:

**AI4Bharat/MSMARCO-XI**

Hugging Face:
`https://huggingface.co/datasets/ai4bharat/MSMARCO-XI`

This dataset is the required knowledge source for the RAG pipeline.

## Important decision

Do **not** replace the HH Goa dataset with an unrelated personal JSON dataset.

Our own JSON files may be created for:

- processed/chunked data
- metadata
- evaluation queries
- configuration
- test cases

But the actual knowledge used by the RAG system must originate from the provided **MSMARCO-XI** dataset.

## First implementation task

Before designing the final ingestion pipeline, load the actual dataset and inspect:

- dataset splits
- number of records
- column names
- data types
- example records
- available text fields
- question fields
- answer fields, if present
- metadata
- language information, if present
- document/context fields
- duplicate or empty records

Do not assume the dataset schema. Inspect it first.

---

# 3. What RAG Means for This Project

RAG = **Retrieval-Augmented Generation**.

A normal LLM interaction is approximately:

```text
User Question
     ↓
LLM
     ↓
Answer
```

A RAG system adds an external retrieval step:

```text
User Question
     ↓
Retrieve relevant information
     ↓
Relevant Context
     ↓
LLM
     ↓
Grounded Answer
```

The retriever searches the project's knowledge base. The retrieved information is then provided to the LLM as context.

The LLM should be instructed to answer using the retrieved context and avoid inventing unsupported information.

## Core distinction

The retriever's responsibility:

> Find relevant evidence.

The LLM's responsibility:

> Use that evidence to formulate a useful answer.

---

# 4. Complete System Architecture

The recommended architecture is:

```text
                           OFFLINE / INDEXING
                           ==================

                       MSMARCO-XI Dataset
                                ↓
                         Data Inspection
                                ↓
                         Data Cleaning
                                ↓
                       Document Preparation
                                ↓
                         Chunking Strategies
                                ↓
                         Embedding Generation
                                ↓
                         Vector Index / DB
                                ↓
                         Persisted RAG Index


                           ONLINE / QUERY
                           =============

                         User Voice Input
                                ↓
                       Speech-to-Text
                    (Sarvam OR ElevenLabs)
                                ↓
                         Text Query
                                ↓
                      Input Validation
                                ↓
                       Query Embedding
                                ↓
                    Vector Similarity Search
                                ↓
                       Top-K Candidates
                                ↓
                   Relevance / Threshold Check
                                ↓
                     Optional Reranking
                                ↓
                       Retrieved Context
                                ↓
                     LLM Answer Generation
                                ↓
                       Grounding Check
                                ↓
                     Structured Response
                                ↓
                         Frontend / User
```

---

# 5. Two Different Pipelines

A critical architectural decision is to separate the system into:

## A. Offline ingestion/indexing pipeline

This should run when preparing the knowledge base.

```text
Dataset
  ↓
Clean
  ↓
Prepare documents
  ↓
Chunk
  ↓
Embed
  ↓
Store in vector index
```

This work should NOT happen for every user query.

Example:

```bash
python ingestion/build_index.py
```

After this completes, the application should have a reusable vector index.

---

## B. Online query pipeline

This runs when the user asks a question.

```text
Voice
  ↓
STT
  ↓
Text Query
  ↓
Embedding
  ↓
Vector Search
  ↓
Relevant Chunks
  ↓
LLM
  ↓
Grounded Answer
```

The online pipeline must be optimized for low latency.

---

# 6. Speech-to-Text

The official requirement is:

> Use either Sarvam or ElevenLabs for voice-to-text.

Only one needs to be selected.

Recommended approach:

```text
Microphone
    ↓
Audio
    ↓
Sarvam OR ElevenLabs
    ↓
Transcribed text
```

Do not build a custom speech recognition model unless there is a strong reason.

The STT layer should expose a simple interface such as:

```text
transcribe(audio) -> text
```

This keeps voice functionality independent from the RAG system.

---

# 7. Dataset Inspection

The first coding milestone should be dataset inspection.

Create a script such as:

```text
ingestion/inspect_dataset.py
```

It should print:

```text
Dataset name
Available splits
Number of records per split
Columns
Data types
First few records
Missing values
Sample text lengths
```

The purpose is to answer:

> What exactly does MSMARCO-XI contain, and which fields should become searchable knowledge?

Do not decide the chunking strategy before inspecting the real structure.

---

# 8. Data Preparation

After inspection, create a normalization layer.

Conceptually:

```text
Raw Dataset Record
        ↓
Normalization
        ↓
Internal Document Object
```

A useful internal representation could conceptually look like:

```json
{
  "id": "unique-id",
  "text": "searchable textual content",
  "metadata": {
    "source": "MSMARCO-XI",
    "language": "..."
  }
}
```

The exact fields must be based on the actual dataset schema.

Do not invent metadata that does not exist.

---

# 9. Chunking

## What is chunking?

Chunking means dividing large textual content into smaller meaningful pieces.

Example:

```text
Large Document
      ↓
-----------------------------
Chunk 1
Chunk 2
Chunk 3
Chunk 4
-----------------------------
```

The purpose is to make retrieval more precise.

Instead of retrieving a huge document, the system retrieves the smaller section that is relevant to the query.

---

# 10. Chunking Requirement from HH Goa

The task explicitly says that the chunking strategy should be substantial and that the submission should not use only a naive fixed-size chunking approach.

The task specifically mentions ideas such as:

- multiple chunking strategies
- overlap handling
- semantic splitting
- metadata-aware chunking

Therefore, the final implementation should demonstrate experimentation or selection among multiple strategies.

---

# 11. Recommended Chunking Development Strategy

Do NOT begin with a highly complicated chunker.

Build progressively.

## Strategy 1 — Baseline

Start with sentence-aware or token-based chunks with controlled overlap.

Example concept:

```text
Chunk size: approximately N tokens
Overlap: approximately M tokens
```

The actual values should be experimentally selected.

## Strategy 2 — Sentence-aware

Keep sentence boundaries intact.

Avoid splitting in the middle of sentences where possible.

## Strategy 3 — Semantic chunking

Group semantically related sentences or passages.

The goal is:

```text
One chunk ≈ one coherent information unit
```

rather than:

```text
One chunk = arbitrary characters
```

## Strategy 4 — Metadata-aware

If the dataset contains useful metadata, preserve it with every chunk.

Example:

```json
{
  "text": "...",
  "metadata": {
    "document_id": "...",
    "language": "...",
    "source": "MSMARCO-XI"
  }
}
```

---

# 12. Chunk Metadata

Every chunk should ideally have enough metadata to trace it back to the original dataset.

Conceptually:

```json
{
  "chunk_id": "...",
  "document_id": "...",
  "chunk_index": 3,
  "text": "...",
  "metadata": {
    "source": "MSMARCO-XI"
  }
}
```

This is useful for:

- debugging
- evaluation
- source display
- retrieval analysis
- duplicate detection
- grounding checks

---

# 13. Embeddings

An embedding converts text into a numerical vector representing semantic information.

Conceptually:

```text
"What is machine learning?"
             ↓
       Embedding Model
             ↓
[0.12, -0.83, 0.44, ...]
```

Each searchable chunk also receives an embedding.

```text
Chunk 1 → Vector
Chunk 2 → Vector
Chunk 3 → Vector
...
```

The query vector can then be compared with chunk vectors.

## Important

The exact embedding model is NOT specified in the HH Goa brief.

Therefore:

- select a suitable embedding model
- document why it was selected
- benchmark its retrieval quality and latency
- keep the embedding generation modular

---

# 14. Vector Database / Vector Index

After embeddings are created, store them in a vector search system.

Possible engineering choices include:

- FAISS
- Qdrant
- Chroma
- another appropriate vector database

For a two-day implementation, a local FAISS-based index is a strong baseline because it is simple and fast.

Architecture:

```text
Chunk
  ↓
Embedding
  ↓
Vector Index
  ↓
Metadata Store
```

The vector index should support:

```text
query vector
    ↓
nearest-neighbor search
    ↓
top-K chunks
```

---

# 15. Retrieval

When the user asks:

```text
"What is X?"
```

the online system performs:

```text
Question
   ↓
Query Embedding
   ↓
Vector Search
   ↓
Top-K Results
```

Example concept:

```text
Query
 ↓
Chunk 392 → similarity 0.93
Chunk 184 → similarity 0.89
Chunk 921 → similarity 0.84
```

The highest-ranked relevant chunks become the context for the LLM.

---

# 16. Retrieval Should Not Automatically Mean "Answer"

A critical guardrail:

If retrieval quality is poor, the system should not blindly ask the LLM to answer.

Example:

```text
User Query
   ↓
Vector Search
   ↓
Best similarity = 0.31
   ↓
Below relevance threshold
   ↓
Do not generate unsupported answer
```

Instead return something such as:

> "I couldn't find sufficient information in the provided knowledge base to answer that reliably."

The exact wording can be refined later.

---

# 17. Top-K Retrieval

The retriever should return a configurable number of candidates.

Example:

```text
TOP_K = 5
```

Then:

```text
Query
 ↓
Top 5 chunks
 ↓
Optional reranking
 ↓
Best context
```

Do not assume that top-K=5 is optimal. Make it configurable and evaluate it.

---

# 18. Optional Reranking

A possible advanced pipeline is:

```text
Vector Search
     ↓
Top 20 candidates
     ↓
Reranker
     ↓
Top 5 relevant chunks
     ↓
LLM
```

Reranking can improve retrieval quality, but it can also increase latency.

Because HH Goa has a strict latency target, reranking should only be retained if the quality improvement justifies its latency cost.

For the first working version:

```text
Query
 ↓
Vector Search
 ↓
Top-K
 ↓
LLM
```

is sufficient.

---

# 19. LLM Generation

The LLM receives:

```text
User Question
+
Retrieved Context
```

Conceptually:

```text
SYSTEM:
You answer questions using the supplied context.
Do not invent information.
If the context does not support an answer,
say that sufficient information was not found.

CONTEXT:
[Retrieved chunk 1]
[Retrieved chunk 2]
[Retrieved chunk 3]

QUESTION:
[user question]
```

The exact LLM is not specified in the task brief.

Choose a model/API based on:

- latency
- cost
- quality
- availability
- structured output support

Keep the LLM layer modular.

---

# 20. Grounding / Hallucination Control

The task explicitly requires guardrails around:

- off-topic queries
- unsafe/inappropriate inputs
- hallucinations
- answers not grounded in retrieved context

Therefore, implement multiple checks.

## Input relevance

Reject or safely handle clearly unsupported/off-topic queries.

## Retrieval relevance

Use similarity scores or another relevance mechanism.

## Generation grounding

The model should be explicitly instructed:

```text
Use retrieved context.
Do not invent facts.
If context is insufficient, refuse to answer confidently.
```

## Output validation

Validate that:

- response is non-empty
- response has expected structure
- unsupported claims are minimized
- fallback is used when evidence is insufficient

---

# 21. Harness / Orchestration

The task explicitly asks for a proper harness around the model rather than a single raw prompt-in/text-out call.

The recommended orchestration is:

```text
Request
  ↓
Validate input
  ↓
Speech-to-text (if voice)
  ↓
Normalize query
  ↓
Create query embedding
  ↓
Retrieve candidates
  ↓
Check relevance
  ↓
Optional reranking
  ↓
Build context
  ↓
Call LLM
  ↓
Validate/ground answer
  ↓
Return structured response
```

The harness should handle:

- structured input
- structured output
- errors
- retries where appropriate
- timeouts
- retrieval failures
- STT failures
- LLM failures
- empty results

---

# 22. Suggested Backend Structure

A practical structure:

```text
hh-goa-voice-rag/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── voice.py
│   │   │   └── query.py
│   │   │
│   │   ├── services/
│   │   │   ├── stt_service.py
│   │   │   ├── embedding_service.py
│   │   │   ├── retrieval_service.py
│   │   │   ├── llm_service.py
│   │   │   └── rag_service.py
│   │   │
│   │   ├── guardrails/
│   │   │   ├── input_guard.py
│   │   │   ├── relevance_guard.py
│   │   │   └── grounding_guard.py
│   │   │
│   │   └── config.py
│   │
│   └── requirements.txt
│
├── ingestion/
│   ├── inspect_dataset.py
│   ├── load_dataset.py
│   ├── preprocess.py
│   ├── chunking/
│   │   ├── baseline.py
│   │   ├── sentence.py
│   │   └── semantic.py
│   ├── embeddings.py
│   └── build_index.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── index/
│
├── evaluation/
│   ├── test_queries.json
│   ├── benchmark.py
│   ├── retrieval_eval.py
│   └── latency_report.py
│
├── frontend/
│
├── .env.example
├── README.md
└── .gitignore
```

This is a recommended structure, not an HH Goa-mandated structure.

---

# 23. Recommended Development Order

Because the submission deadline is close, implement in this exact order.

## Phase 1 — Dataset

```text
[1] Load MSMARCO-XI
[2] Inspect schema
[3] Identify useful text fields
[4] Normalize records
```

Do not proceed until the dataset structure is understood.

---

## Phase 2 — Baseline RAG

```text
[5] Implement baseline chunking
[6] Generate embeddings
[7] Build FAISS/vector index
[8] Implement similarity retrieval
[9] Test retrieval with text queries
```

Goal:

```text
Text Query
    ↓
Relevant Chunks
```

---

## Phase 3 — Generation

```text
[10] Connect LLM
[11] Build context prompt
[12] Generate grounded answer
[13] Add "insufficient context" fallback
```

Goal:

```text
Text Query
    ↓
Retriever
    ↓
Context
    ↓
LLM
    ↓
Answer
```

At this point the core RAG system works.

---

## Phase 4 — Guardrails

```text
[14] Input validation
[15] Relevance threshold
[16] Off-topic handling
[17] Grounding instructions
[18] Output validation
[19] Error handling
```

---

## Phase 5 — Voice

```text
[20] Integrate Sarvam OR ElevenLabs
[21] Audio upload/recording endpoint
[22] Convert audio → text
[23] Send text into existing RAG pipeline
```

Architecture:

```text
Voice
 ↓
STT
 ↓
Existing Text RAG
```

Do not create a second independent RAG pipeline for voice.

---

## Phase 6 — Evaluation

```text
[24] Prepare test queries
[25] Run multiple queries
[26] Measure retrieval latency
[27] Measure generation latency
[28] Calculate P50
[29] Calculate P70
[30] Calculate P100
```

The official task specifically requires P50/P70/P100 latency numbers over a reasonable number of test queries.

---

## Phase 7 — Chunking Experiments

Once the baseline works:

```text
[31] Baseline chunking
[32] Sentence-aware chunking
[33] Semantic/metadata-aware approach
[34] Compare retrieval quality
[35] Compare latency
[36] Select final strategy
```

Do not sacrifice the working pipeline just to create a complicated chunking system.

---

## Phase 8 — Frontend

Finally:

```text
[37] Microphone button
[38] Recording state
[39] Transcription display
[40] Answer display
[41] Optional retrieved-source display
[42] Error/fallback states
```

Keep the UI simple and reliable.

---

# 24. Latency Requirement

The official target says the full process from chunking/vector DB retrieval through final output should complete in **under 200 ms**.

This is a very aggressive target, especially if external APIs are included.

Therefore, latency engineering must be intentional.

## Offline work

Do these before user queries:

- dataset loading
- cleaning
- chunking
- embedding generation
- index construction

## Online work

Keep this optimized:

```text
Query
 ↓
Embedding
 ↓
Vector search
 ↓
Context construction
 ↓
LLM
```

Avoid:

- re-embedding the whole dataset
- rebuilding the vector index
- chunking the entire dataset
- unnecessary network calls
- unnecessarily large contexts

---

# 25. Latency Instrumentation

Instrument each stage.

Example:

```text
request_start
stt_start
stt_end
embedding_start
embedding_end
retrieval_start
retrieval_end
llm_start
llm_end
response_end
```

Calculate:

```text
STT latency
Embedding latency
Retrieval latency
LLM latency
Total RAG latency
```

For the benchmark, clearly document which stages are included in the official reported latency.

Do not invent latency numbers.

---

# 26. P50 / P70 / P100

Run a reasonable number of test queries.

Example test dataset:

```json
[
  {"query": "..."},
  {"query": "..."},
  {"query": "..."}
]
```

Collect latency:

```text
82 ms
91 ms
104 ms
...
```

Then calculate:

```text
P50 = median-like 50th percentile
P70 = 70th percentile
P100 = maximum observed latency
```

Report actual measured values.

---

# 27. Evaluation Beyond Latency

Latency alone is not enough.

Evaluate:

## Retrieval quality

Ask:

> Did the retrieved chunks actually contain information relevant to the query?

Measure or manually inspect:

- relevance
- top-K retrieval
- similarity scores
- failure cases

## Answer quality

Check:

- correctness
- groundedness
- completeness
- hallucination rate

## Failure handling

Test:

- valid questions
- irrelevant questions
- empty queries
- malformed requests
- unsupported questions
- low-retrieval-confidence queries
- STT errors
- LLM/API failures

---

# 28. Example End-to-End Request

User says:

```text
🎤 "What is artificial intelligence?"
```

### Step 1 — STT

```text
"What is artificial intelligence?"
```

### Step 2 — Query embedding

```text
Query
 ↓
Embedding
```

### Step 3 — Retrieval

```text
Vector DB
 ↓
Top relevant chunks
```

### Step 4 — Relevance check

If relevant:

```text
Continue
```

If not:

```text
Return insufficient-context response
```

### Step 5 — Generation

```text
Question + Retrieved Context
        ↓
       LLM
        ↓
Grounded answer
```

### Step 6 — Response

Return something like:

```json
{
  "transcript": "What is artificial intelligence?",
  "answer": "...",
  "sources": [
    {
      "chunk_id": "...",
      "score": 0.91
    }
  ],
  "latency_ms": {
    "retrieval": 8,
    "generation": 70,
    "total_rag": 82
  }
}
```

The exact response schema can be refined during implementation.

---

# 29. Important Engineering Principle

Do not build everything in one giant file.

Avoid:

```text
main.py
    ↓
load dataset
    ↓
chunk
    ↓
embed
    ↓
retrieve
    ↓
STT
    ↓
LLM
    ↓
guardrails
    ↓
response
```

Instead separate responsibilities:

```text
STT Service
Embedding Service
Retrieval Service
LLM Service
Guardrail Service
RAG Orchestrator
```

This makes testing and debugging much easier.

---

# 30. Recommended Technology Direction

The official brief only specifies the STT providers and dataset. The following is an implementation recommendation.

## Backend

Python

Possible framework:

- FastAPI

## Dataset

Hugging Face `datasets`

## Embeddings

A suitable sentence/document embedding model.

The final choice should be based on:

- semantic retrieval quality
- language support
- model size
- inference speed

## Vector search

Initial recommendation:

**FAISS**

Reason:

- easy local setup
- fast similarity search
- good for a short submission timeline
- low operational complexity

Qdrant can be considered if a persistent vector database/API architecture is preferred.

## LLM

Choose a low-latency API/model that is available to the team.

The task brief does not mandate a particular LLM.

## STT

Choose one:

- Sarvam
- ElevenLabs

## Frontend

Use whatever frontend stack the team can implement quickly.

Do not let frontend development delay the RAG backend.

---

# 31. Environment Variables

API keys must not be hardcoded.

Use:

```text
.env
```

Conceptually:

```text
STT_API_KEY=
LLM_API_KEY=
EMBEDDING_CONFIG=
VECTOR_DB_CONFIG=
```

Provide:

```text
.env.example
```

in the repository.

Never commit real API keys.

---

# 32. GitHub Repository Expectations

The repository should be clean and understandable.

Recommended README sections:

```text
1. Project Overview
2. Architecture
3. Dataset
4. Setup
5. Environment Variables
6. Dataset Ingestion
7. Building the Vector Index
8. Running Backend
9. Running Frontend
10. RAG Pipeline
11. Chunking Strategies
12. Guardrails
13. Evaluation
14. Latency Results
15. Limitations
16. Future Improvements
```

---

# 33. What NOT to Do

## Do not use unrelated personal data as the knowledge base

The required dataset is MSMARCO-XI.

## Do not build only a generic chatbot

The system must actually retrieve from the supplied dataset.

## Do not use only naive fixed-size chunking

The task explicitly asks for thoughtful chunking.

## Do not hardcode answers

Answers must come through retrieval + generation.

## Do not let the LLM freely hallucinate

Use retrieval relevance and grounding guardrails.

## Do not rebuild the index on every query

Index offline.

## Do not focus on UI first

Backend/RAG first.

## Do not invent benchmark numbers

Measure actual performance.

---

# 34. Two-Day Execution Plan

Because the submission deadline is close, prioritize the working end-to-end system.

## Day 1 — Core RAG

### Block 1

```text
Load MSMARCO-XI
Inspect schema
```

### Block 2

```text
Clean/normalize data
Implement baseline chunking
```

### Block 3

```text
Embeddings
FAISS/vector index
```

### Block 4

```text
Text query
 ↓
Retrieval
 ↓
LLM
 ↓
Answer
```

### Block 5

```text
Guardrails
Error handling
```

### Block 6

```text
Test retrieval
Fix obvious failures
```

---

# 35. Day 2 — Voice + Optimization + Demo

### Block 1

```text
Sarvam OR ElevenLabs
 ↓
Speech-to-text
```

### Block 2

```text
Voice
 ↓
STT
 ↓
RAG
 ↓
Answer
```

### Block 3

```text
Chunking experiments
```

### Block 4

```text
Latency instrumentation
P50/P70/P100
```

### Block 5

```text
Simple frontend
```

### Block 6

```text
Deployment
```

### Block 7

```text
README
Demo video
Process video
Final testing
```

---

# 36. Submission Requirements

The official submission requires:

- submission form
- GitHub repository link
- live working link
- two videos

The brief specifies:

## Video 1 — Team/process video

- 90 seconds
- show how the team is working
- process rather than product

## Video 2 — Demo video

- show the actual project working end-to-end

Both videos must be uploaded to:

- Instagram
- X
- LinkedIn

by every individual team member.

At least one Instagram account must be public.

Every post must include:

```text
#RAGInGoa
```

---

# 37. Final Architecture to Aim For

The final system should look like:

```text
                         ┌─────────────────────┐
                         │    MSMARCO-XI       │
                         └──────────┬──────────┘
                                    │
                              OFFLINE PIPELINE
                                    │
                     ┌──────────────▼──────────────┐
                     │ Data Cleaning / Preparation │
                     └──────────────┬──────────────┘
                                    ↓
                     ┌─────────────────────────────┐
                     │ Multiple Chunking Strategies│
                     └──────────────┬──────────────┘
                                    ↓
                     ┌─────────────────────────────┐
                     │     Embedding Generation    │
                     └──────────────┬──────────────┘
                                    ↓
                     ┌─────────────────────────────┐
                     │      Vector Index / DB      │
                     └──────────────┬──────────────┘
                                    │
                                    │
                              ONLINE PIPELINE
                                    │
             🎤 Voice ──────────────┤
                ↓                   │
        Sarvam/ElevenLabs          │
                ↓                   │
             Text Query             │
                ↓                   │
        ┌──────────────────┐        │
        │ Input Validation │        │
        └────────┬─────────┘        │
                 ↓                  │
        ┌──────────────────┐        │
        │ Query Embedding  │        │
        └────────┬─────────┘        │
                 ↓                  │
        ┌──────────────────┐        │
        │ Vector Retrieval │◄───────┘
        └────────┬─────────┘
                 ↓
        ┌──────────────────────┐
        │ Relevance / Guardrail│
        └──────────┬───────────┘
                   ↓
        ┌──────────────────────┐
        │ Retrieved Context    │
        └──────────┬───────────┘
                   ↓
        ┌──────────────────────┐
        │         LLM          │
        └──────────┬───────────┘
                   ↓
        ┌──────────────────────┐
        │ Grounding Validation │
        └──────────┬───────────┘
                   ↓
              Final Answer
                   ↓
                User
```

---

# 38. Immediate Next Action

Do NOT start by implementing the complete architecture above.

Start with exactly this:

```text
1. Create Python project
2. Install Hugging Face datasets library
3. Load MSMARCO-XI
4. Inspect its splits
5. Inspect its columns
6. Print several real examples
7. Measure text lengths
8. Determine which fields contain the searchable knowledge
```

After that, the next implementation decision should be made from the **actual MSMARCO-XI schema**, not from assumptions.

The first milestone is therefore:

```text
MSMARCO-XI
    ↓
Successfully loaded
    ↓
Schema understood
    ↓
Sample records understood
```

Only then proceed to chunking.

---

# 39. Core Mental Model for the Developer/AI

If another AI model needs to understand this project quickly, remember:

```text
HH Goa gives us:
    ↓
MSMARCO-XI knowledge
    ↓
We prepare it
    ↓
We split it into useful chunks
    ↓
We convert chunks to embeddings
    ↓
We store them in a vector index
    ↓
User speaks a question
    ↓
Sarvam/ElevenLabs converts voice → text
    ↓
We embed the question
    ↓
We retrieve relevant chunks
    ↓
We check whether the retrieval is good enough
    ↓
We give relevant context + question to the LLM
    ↓
LLM generates a grounded answer
    ↓
We validate the answer
    ↓
Return response
```

The project is **not merely a voice chatbot**.

It is:

> **A low-latency, voice-enabled, dataset-grounded RAG system with thoughtful chunking, vector retrieval, orchestration, guardrails, and measurable latency.**

That is the implementation target.
