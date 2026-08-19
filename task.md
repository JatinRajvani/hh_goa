# Dynamic Retrieval Engine Selector Tasks

- [x] Modify `search` method signature and query routing in `retrieval/search.py` to accept dynamic `mode` parameter
- [x] Update `query_rag` and `query_rag_voice` in `retrieval/rag_orchestrator.py` to accept and forward `retrieval_mode`
- [x] Add `retrieval_mode` parsing to API endpoint routes in `backend/app.py`
- [x] Add the Search Engine dropdown selector in the header layout of `frontend/index.html`
- [x] Update `frontend/app.js` to check hostname, lock selector to BM25 in cloud production, and send selected mode in query requests
- [x] Update `frontend/app.js` updateUI function to dynamically rename Vector/Lexical Retrieval latency headers
- [x] Verify dynamic local engine switching (Side-by-Side testing)
- [x] Update walkthrough documentation
