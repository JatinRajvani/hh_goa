# Dual-Mode Retrieval Integration Tasks

- [x] Add `RETRIEVAL_MODE=dense` to `.env` and `.env.example`
- [x] Update `retrieval/search.py` load logic to bypass loading model when mode is `sparse`
- [x] Implement lazy BM25 index initialization from mappings in `search.py` when in `sparse` mode
- [x] Update search query path in `search.py` to execute BM25 lexical search when in `sparse` mode
- [x] Verify `dense` (semantic) RAG search functions correctly
- [x] Verify `sparse` (BM25) RAG search functions correctly (without loading sentence transformer)
- [x] Update walkthrough documentation
