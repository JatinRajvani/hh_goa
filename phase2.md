# HH Goa 2026 — Task 2

## Phase 2: English Retrieval Corpus Extraction & Preparation

## 1. Context

We are building the HH Goa 2026 Task 2 project:

**Voice-Enabled Retrieval-Augmented Generation (RAG) Model**

The official pipeline required by the task is:

```text
Voice Input
    ↓
Speech-to-Text
    ↓
Retrieval / Vector DB
    ↓
Relevant Context
    ↓
Answer Generation
```

The official dataset is:

```text
ai4bharat/MSMARCO-XI
```

Hugging Face:

```text
https://huggingface.co/datasets/ai4bharat/MSMARCO-XI
```

The project is being developed phase-by-phase.

### Phase 1 has already been completed conceptually.

We inspected the dataset structure and confirmed that each record contains fields approximately like:

```text
source_lang
target_lang
meta
query
Answer
query_id
query_type
passages
    ├── is_selected
    ├── English_passages
    └── Translated_passages
Eng_Query
Eng_Answer
```

The dataset is organized into language-specific Parquet files such as:

```text
train/asmtrain.parquet
train/bentrain.parquet
train/gujtrain.parquet
train/hintrain.parquet
...
```

and corresponding validation files.

We also discovered that directly converting the nested `passages` structure through the current PyArrow/Pandas path causes:

```text
pyarrow.lib.ArrowNotImplementedError:
Nested data conversions not implemented for chunked array outputs
```

Therefore, do not blindly use a Pandas conversion of the complete nested Parquet structure.

---

# 2. Goal of Phase 2

The goal of this phase is:

> **Extract the English passages from MSMARCO-XI and convert them into a clean, manageable retrieval corpus that can later be embedded and indexed.**

At the end of this phase we should have:

```text
MSMARCO-XI
    ↓
English passages extracted
    ↓
Clean retrieval documents
    ↓
Local JSONL/JSON dataset
    ↓
Ready for Phase 3: Embeddings + Vector Retrieval
```

### Important

This phase is ONLY about:

* reading/extracting the dataset
* selecting English passages
* cleaning them
* assigning IDs
* preserving useful metadata
* creating a clean local corpus
* validating the extracted data

Do NOT implement:

* embeddings
* FAISS
* vector database
* LLM
* RAG generation
* speech-to-text
* frontend
* voice
* final API
* production deployment

Those belong to later phases.

---

# 3. Why We Are Starting With English

For the first working retrieval implementation, we are intentionally starting with the English side of MSMARCO-XI.

The dataset provides:

```text
Eng_Query
Eng_Answer
English_passages
```

This allows us to establish a reliable baseline before adding multilingual/Indic-language retrieval.

The initial pipeline should therefore be:

```text
English Query
      ↓
English Passage Corpus
      ↓
[Phase 3] Embeddings
      ↓
[Phase 3] Vector Search
```

Later, the architecture can be extended to Gujarati, Hindi, or other supported languages.

Do not build the multilingual pipeline in this phase.

---

# 4. Understand the Raw Record

A raw record conceptually looks like:

```json
{
  "source_lang": "eng_Latn",
  "target_lang": "asm_Beng",

  "meta": {
    "model_name": "...",
    "temperature": 0.0,
    "max_tokens": 4096,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  },

  "query": "Translated query",
  "Answer": "Translated answer",
  "query_id": 1185869,
  "query_type": "DESCRIPTION",

  "passages": {
    "is_selected": [1, 0, 0, 0, ...],
    "English_passages": [
      "English passage 1",
      "English passage 2",
      "..."
    ],
    "Translated_passages": [
      "...",
      "..."
    ]
  },

  "Eng_Query": "Original English query",
  "Eng_Answer": "Original English answer"
}
```

The exact values will vary by record.

---

# 5. What We Want to Extract

The primary retrieval corpus should be based on:

```text
passages.English_passages
```

Each individual English passage should become a retrieval document.

For example:

```text
Raw record
query_id = 1185869

English_passages:
    [0] passage A
    [1] passage B
    [2] passage C
    ...
```

should become:

```text
Document 1185869_0
Document 1185869_1
Document 1185869_2
...
```

---

# 6. Important: Do NOT Use the Dataset Answer as the Retrieval Document

Do NOT construct the retrieval corpus as:

```text
query + Answer
```

and embed that.

The purpose of RAG is to retrieve evidence/context and then allow the later LLM to generate an answer.

We want:

```text
Question
   ↓
Retrieve evidence/passages
   ↓
LLM
   ↓
Generate answer
```

not:

```text
Question
   ↓
Retrieve pre-written answer
   ↓
Return answer
```

Therefore:

### Primary retrieval text

```text
English_passages
```

### Evaluation/reference information

```text
Eng_Query
Eng_Answer
Answer
query
is_selected
```

These can be retained separately.

---

# 7. `is_selected` Is Important

Each record contains:

```text
is_selected
```

for the candidate passages.

Example:

```text
is_selected:
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

This provides a relevance signal indicating which candidate passage(s) were selected for the query.

Do NOT use `is_selected` to remove all non-selected passages from the retrieval corpus.

Instead:

```text
Passage 0 → indexed → is_selected = 1
Passage 1 → indexed → is_selected = 0
Passage 2 → indexed → is_selected = 0
...
```

This preserves the retrieval challenge.

Later, in Phase 3/4, we can use this information to evaluate whether our vector retriever actually retrieves the relevant passage.

---

# 8. Proposed Internal Document Schema

Convert each passage into a clean object approximately like:

```json
{
  "document_id": "1185869_0",
  "text": "The presence of communication amid scientific minds...",
  "metadata": {
    "query_id": 1185869,
    "passage_index": 0,
    "is_selected": 1,
    "source_lang": "eng_Latn",
    "target_lang": "asm_Beng",
    "query_type": "DESCRIPTION"
  }
}
```

The exact metadata structure may be adjusted if the actual dataset requires it.

### Required fields

At minimum:

```text
document_id
text
query_id
passage_index
is_selected
```

### Useful optional metadata

```text
source_lang
target_lang
query_type
```

Do not copy unnecessary generation metadata such as:

```text
temperature
top_p
frequency_penalty
presence_penalty
max_tokens
```

into the searchable text.

They are generation/provenance metadata and are not useful for semantic retrieval.

---

# 9. Preserve Evaluation Information Separately

Because we want to evaluate retrieval later, maintain a separate evaluation representation.

For example:

```json
{
  "query_id": 1185869,
  "query": "What was the immediate impact of the success of the Manhattan Project?",
  "relevant_passage_ids": [
    "1185869_0"
  ]
}
```

The exact relevant passage IDs should be generated from:

```text
is_selected == 1
```

This will allow later evaluation:

```text
Query
 ↓
Retriever
 ↓
Top-K documents
 ↓
Compare with relevant_passage_ids
```

Potential future metrics:

```text
Recall@1
Recall@5
Recall@10
MRR
```

Do not implement the complete evaluation system yet unless it is trivial to include.

---

# 10. Data Extraction Strategy

The dataset is large, so do NOT download/process the entire 55+ GB repository just for initial development.

We need a staged approach.

## Development dataset

First extract a manageable subset.

For example:

```text
5,000–50,000 retrieval documents
```

The exact size should be determined based on available disk space, memory, network speed, and processing time.

The goal is:

> Build a working pipeline first, then scale.

Do not assume that the entire dataset must be locally downloaded before development can continue.

---

# 11. Efficient File Access

The current dataset repository exposes language-specific Parquet files.

For English retrieval, determine which file(s) contain the English source/query content needed for the corpus.

Do not assume that the entire repository must be loaded.

Prefer an approach that can:

* access the required Parquet shard
* read records incrementally
* avoid loading the entire nested dataset into Pandas
* stop after a configurable number of records/passages
* avoid loading all 55+ GB into RAM

Possible approaches can include:

* Hugging Face streaming
* PyArrow dataset/Parquet APIs
* HTTP/range-based access
* another appropriate incremental reader

However, because the nested `passages` field has already caused a PyArrow/Pandas conversion error, test the chosen extraction method with a small sample before processing thousands of records.

---

# 12. Do Not Convert the Entire Nested Column to Pandas

We previously encountered:

```text
ArrowNotImplementedError:
Nested data conversions not implemented for chunked array outputs
```

Therefore avoid a pipeline like:

```text
Parquet
 ↓
Entire nested table
 ↓
Pandas DataFrame
```

Instead prefer:

```text
Parquet
 ↓
Incremental record/batch reading
 ↓
Extract passages
 ↓
Clean passage
 ↓
Write output
```

The extraction layer should isolate the nested structure rather than forcing the entire dataset into a flat DataFrame.

---

# 13. Cleaning Rules

Each extracted passage should be normalized before storage.

Apply conservative cleaning only.

Recommended:

* remove leading/trailing whitespace
* normalize repeated whitespace
* remove empty strings
* skip null values
* skip extremely short meaningless passages
* preserve punctuation
* preserve the original wording as much as possible
* do not aggressively rewrite or summarize passages

Do NOT:

* use an LLM to rewrite passages
* summarize passages
* translate English passages
* merge unrelated passages
* remove meaningful punctuation
* change the factual content

The goal is to create a faithful retrieval corpus.

---

# 14. Deduplication

Check for duplicate passages.

At minimum:

```text
exact text duplicate
```

can be detected using a normalized text hash.

However, do not aggressively remove records before understanding the dataset.

If duplicate content exists across different records, preserve the source metadata where useful.

A safe approach is:

```text
normalized_text_hash
```

and report:

```text
total passages
unique passages
duplicate count
```

before deciding how aggressively to deduplicate.

---

# 15. Output Format

Prefer **JSONL** for the intermediate corpus.

Example:

```text
data/processed/english_passages.jsonl
```

Each line:

```json
{"document_id":"1185869_0","text":"...","metadata":{"query_id":1185869,"passage_index":0,"is_selected":1,"source_lang":"eng_Latn","target_lang":"asm_Beng","query_type":"DESCRIPTION"}}
```

JSONL is preferable because:

* one record per line
* easy streaming
* easy debugging
* doesn't require loading the entire dataset into memory
* easy to process later for embeddings

---

# 16. Evaluation Query Output

Also create:

```text
data/processed/evaluation_queries.jsonl
```

Example:

```json
{
  "query_id": 1185869,
  "query": "what was the immediate impact of the success of the manhattan project?",
  "relevant_passage_ids": [
    "1185869_0"
  ]
}
```

Use `Eng_Query` for the English retrieval benchmark.

Use `is_selected` to generate `relevant_passage_ids`.

This file will be used in later retrieval evaluation.

---

# 17. Suggested Project Structure After Phase 2

The project should look approximately like:

```text
hh-goa/
│
├── ingestion/
│   ├── inspect_dataset.py
│   ├── check_dataset.py
│   └── extract_english_passages.py
│
├── data/
│   ├── raw/
│   │
│   └── processed/
│       ├── english_passages.jsonl
│       └── evaluation_queries.jsonl
│
├── backend/
│
├── retrieval/
│
├── evaluation/
│
└── README.md
```

Do not create unnecessary files just for the sake of the structure.

---

# 18. Extraction Script Requirements

Create:

```text
ingestion/extract_english_passages.py
```

It should have a clear configuration section.

For example:

```python
MAX_RECORDS = 5000
MAX_PASSAGES = 50000
OUTPUT_FILE = "data/processed/english_passages.jsonl"
EVAL_OUTPUT_FILE = "data/processed/evaluation_queries.jsonl"
```

The exact values can be adjusted.

The script should:

1. access the appropriate MSMARCO-XI data source
2. read records incrementally
3. extract `English_passages`
4. extract `is_selected`
5. extract `query_id`
6. extract `Eng_Query`
7. extract `query_type`
8. extract language metadata
9. create document IDs
10. clean passage text
11. skip invalid/empty passages
12. write retrieval documents to JSONL
13. create evaluation query records
14. print progress
15. print final statistics

---

# 19. Important Statistics to Print

At the end, report:

```text
Records processed:
Passages encountered:
Passages written:
Empty passages skipped:
Duplicate passages:
Selected passages:
Non-selected passages:
Unique queries:
Output file:
Evaluation file:
```

Example:

```text
========== Extraction Complete ==========

Records processed:       5,000
Passages encountered:    50,000
Passages written:        47,321
Empty passages skipped:  120
Duplicate passages:      2,559
Selected passages:       5,000
Non-selected passages:   42,321
Unique queries:           5,000

Output:
data/processed/english_passages.jsonl

Evaluation:
data/processed/evaluation_queries.jsonl
```

These numbers are examples only. Never hardcode them.

---

# 20. Validate the Output

After extraction, create a small validation step.

Check:

```text
Does every document have:
    document_id?
    text?
    query_id?
    passage_index?
    is_selected?
```

Check:

```text
Are there empty texts?
Are document IDs unique?
Are query IDs valid?
Are selected flags valid?
```

Print a few examples:

```text
Document 1:
...

Document 2:
...

Document 3:
...
```

Also inspect:

```text
evaluation_queries.jsonl
```

and verify that relevant passage IDs actually exist in the corpus.

---

# 21. Important Constraint: No Embeddings Yet

Do NOT implement:

```text
Embedding Model
```

in Phase 2.

Do NOT install large embedding models unnecessarily.

Do NOT create:

```text
FAISS
Qdrant
Chroma
Pinecone
```

yet.

Those belong to the next phase.

The output of Phase 2 should simply be:

```text
Clean English retrieval corpus
+
Evaluation query/reference data
```

---

# 22. Phase 2 Success Criteria

Phase 2 is complete only when all of these are true:

### Dataset access

* [ ] English data can be accessed successfully.
* [ ] No full 55+ GB download is required for initial development.
* [ ] Extraction is incremental or otherwise memory-safe.

### Corpus

* [ ] English passages are extracted.
* [ ] Empty/invalid passages are handled.
* [ ] Each passage has a unique document ID.
* [ ] Useful metadata is preserved.
* [ ] Retrieval text contains the passage itself, not the dataset answer.

### Evaluation

* [ ] `is_selected` is preserved.
* [ ] English queries are preserved.
* [ ] Relevant passage IDs can be derived.
* [ ] Evaluation queries are stored separately.

### Output

* [ ] `english_passages.jsonl` exists.
* [ ] `evaluation_queries.jsonl` exists.
* [ ] Sample records have been validated.
* [ ] Extraction statistics are printed.

### Performance

* [ ] The script does not load the entire dataset into RAM.
* [ ] The script can stop after a configurable number of records/passages.
* [ ] Processing can be repeated without corrupting output.

---

# 23. What Comes After This Phase

Do NOT implement the following now, but understand the next stage:

```text
Phase 2
English Corpus Extraction
        ↓
Phase 3
Embeddings + Vector Index
        ↓
Phase 4
Retrieval Evaluation
        ↓
Phase 5
RAG Generation + Grounding
        ↓
Phase 6
Voice / STT
        ↓
Phase 7
Guardrails + Harness
        ↓
Phase 8
Latency Optimization + P50/P70/P100
        ↓
Phase 9
Frontend + Deployment + Demo
```

---

# 24. Instructions to the Coding Agent

You are working on **Phase 2 only**.

Before writing code:

1. Inspect the current project structure.
2. Reuse existing code where appropriate.
3. Do not unnecessarily restructure the entire project.
4. Confirm how the Hugging Face authentication is currently configured.
5. Confirm the dataset access method.
6. Test reading a very small number of records first.
7. Do not download the entire dataset.
8. Do not use Pandas to materialize the complete nested `passages` field.
9. Handle nested passage data directly.
10. Make extraction configurable.

Then implement:

```text
ingestion/extract_english_passages.py
```

and produce:

```text
data/processed/english_passages.jsonl
data/processed/evaluation_queries.jsonl
```

After implementation:

* run the extraction on a small development subset
* validate the output
* report statistics
* show 3–5 sample retrieval documents
* show 2–3 evaluation queries
* explain any assumptions made

If there is a dataset-access problem, diagnose it rather than bypassing it with fabricated data.

Do not create fake MSMARCO-XI records.

Do not replace the dataset with a manually created JSON knowledge base.

---

# 25. Final Expected Result of Phase 2

At the end of this phase, we should be able to say:

> "We have successfully extracted a clean English retrieval corpus from MSMARCO-XI. Each English passage is represented as an independent retrieval document with traceable metadata and relevance information. We also have an evaluation query set containing the English queries and their ground-truth relevant passage IDs. The corpus is now ready for embedding and vector indexing."

The next agent/phase should then take:

```text
data/processed/english_passages.jsonl
```

and build:

```text
English Passage
      ↓
Embedding
      ↓
Vector Index
      ↓
Query Retrieval
```

Do not jump ahead until the Phase 2 success criteria are satisfied.
