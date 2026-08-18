import json
import os
import sys
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure parent directory is in path to allow relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

app = FastAPI(title="Voice-Enabled RAG System")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator variable (lazy initialized)
orchestrator = None

def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        from retrieval.rag_orchestrator import RAGOrchestrator
        orchestrator = RAGOrchestrator()
    return orchestrator

class QueryRequest(BaseModel):
    query: str
    k: int = 5

@app.post("/api/query")
def query_text(request: QueryRequest):
    try:
        orch = get_orchestrator()
        result = orch.query_rag(request.query, k=request.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query-voice")
def query_voice(file: UploadFile = File(...), k: int = Form(5)):
    # Save the uploaded audio file to a temporary file
    temp_dir = tempfile.gettempdir()
    # Try to keep extension or fallback to .wav
    ext = os.path.splitext(file.filename)[1] or ".wav"
    temp_path = os.path.join(temp_dir, f"upload_{os.urandom(8).hex()}{ext}")
    
    try:
        print(f"Saving uploaded voice recording to {temp_path}...")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        orch = get_orchestrator()
        result = orch.query_rag_voice(temp_path, k=k)
        return result
    except Exception as e:
        print(f"Error querying voice RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Temporary file {temp_path} removed.")
            except Exception as e:
                print(f"Error removing temporary file: {e}")

@app.get("/api/evaluation-results")
def get_evaluation_results():
    results_path = os.path.join(parent_dir, "data", "processed", "evaluation_results.json")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Evaluation results not found. Please run evaluate_pipeline.py first.")
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files for the frontend
frontend_dir = os.path.join(parent_dir, "frontend")
os.makedirs(frontend_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
