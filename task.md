# Mode Toggle Integration Tasks

- [x] Modify `retrieval/rag_orchestrator.py` to bypass LLM generation when `use_llm` is False
- [x] Modify `backend/app.py` request structures and endpoint arguments
- [x] Modify `frontend/index.html` to place the "Conversational LLM" toggle switch
- [x] Modify `frontend/index.css` to style the toggle switch slider
- [x] Modify `frontend/app.js` to parse and pass `use_llm` in payloads
- [x] Verify Extractive RAG (0.0ms generation, sub-200ms total)
- [x] Verify Conversational LLM RAG (Groq generation)
- [x] Update walkthrough documentation
