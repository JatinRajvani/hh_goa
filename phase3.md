# HH Goa 2026 — Task 2

## Phase 3: Embeddings & Local Vector Indexing

We have successfully completed Phase 3 of the project.

### Technical Implementation

1. **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
2. **Vector Store**: Local **FAISS** (`IndexFlatIP`) with L2-normalized query and document vectors.
3. **Indexing Script**: [`ingestion/build_index.py`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/ingestion/build_index.py)
4. **Retrieval Service**: [`retrieval/search.py`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/retrieval/search.py)

### Persisted Artifacts

- Vector Index: [`data/index/index.faiss`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/data/index/index.faiss) (73.07 MB)
- Document Mapping: [`data/index/id_mapping.json`](file:///c:/Users/Jatin%20Rajvani/Desktop/hh-goa/data/index/id_mapping.json) (25.14 MB)

### Results & Metrics

- Total Indexed Documents: **49,885**
- Average Retrieval Latency: **69.17 ms**
- Minimum Retrieval Latency: **61.76 ms**
- Maximum Retrieval Latency: **90.56 ms**

We can now run search queries interactively by executing:
```powershell
& .venv\Scripts\python.exe retrieval/search.py --interactive
```


questions
what direction does phloem flow
what does the american flag sticker on cars mean
different types of social security disability
what causes middle back pain
how to use sysdate in sql
what was the immediate impact of the success of the manhattan project?
crevice define
coal miner's daughter cast