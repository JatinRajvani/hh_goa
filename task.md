# Auto-Detect Language Integration Tasks

- [x] Implement `detect_language` helper in `retrieval/rag_orchestrator.py` using `langdetect`
- [x] Update `query_rag` and `query_rag_voice` in `retrieval/rag_orchestrator.py` to route to detected language when `lang == "auto"`
- [x] Update default language to `"auto"` in `backend/app.py` endpoints
- [x] Add `"auto"` option to CLI parser in `retrieval/search.py`
- [x] Add `Auto-Detect` option to dropdown menu in `frontend/index.html` as default selection
- [x] Update `frontend/app.js` to show detected language feedback on UI
- [x] Verify Auto-detect routing for multiple languages (English, Hindi, Gujarati, Tamil, Marathi)
- [x] Update walkthrough documentation
