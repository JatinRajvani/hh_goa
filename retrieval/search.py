import os
import sys
import json
import time
import argparse
import faiss
from sentence_transformers import SentenceTransformer

# Safely reconfigure standard streams to UTF-8 for Windows command line compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure parent directory is in path to allow relative imports when run as script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


DEFAULT_INDEX_DIR = "data/index"
MODEL_NAME = "all-MiniLM-L6-v2"


class RetrievalService:
    def __init__(self, index_dir=DEFAULT_INDEX_DIR, model_name=MODEL_NAME):
        self.index_dir = index_dir
        self.model_name = model_name
        self.index = None
        self.id_mapping = None
        self.model = None
        
    def load(self):
        """Loads FAISS index, document mapping, and embedding model."""
        index_file = os.path.join(self.index_dir, "index.faiss")
        map_file = os.path.join(self.index_dir, "id_mapping.json")
        
        if not os.path.exists(index_file) or not os.path.exists(map_file):
            raise FileNotFoundError(
                f"Index files not found in '{self.index_dir}'. Please run the build_index.py script first."
            )
            
        print(f"Loading FAISS index from {index_file}...")
        self.index = faiss.read_index(index_file)
        
        print(f"Loading document mapping from {map_file}...")
        with open(map_file, "r", encoding="utf-8") as f:
            self.id_mapping = json.load(f)
            
        print(f"Loading embedding model '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name)
        print("Retrieval Service initialized and ready.")
        
    def search(self, query, k=5):
        """
        Embed the text query, perform similarity search on FAISS, and return Top-K matching documents.
        Returns:
            list of dicts containing: document_id, text, metadata, score
            retrieval_latency_ms: float
        """
        if not self.index or not self.id_mapping or not self.model:
            raise RuntimeError("RetrievalService is not loaded. Call load() first.")
            
        start_time = time.time()
        
        # 1. Encode query
        query_vector = self.model.encode(query, convert_to_numpy=True)
        query_vector = query_vector.astype("float32").reshape(1, -1)
        
        # 2. Normalize vector (IP with L2 normalized vectors is Cosine Similarity)
        faiss.normalize_L2(query_vector)
        
        # 3. Perform FAISS search
        scores, indices = self.index.search(query_vector, k)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # 4. Map index offsets back to document mappings
        results = []
        for idx_in_retrieved, offset in enumerate(indices[0]):
            if offset == -1:  # FAISS returns -1 if there are fewer results than k
                continue
            
            score = float(scores[0][idx_in_retrieved])
            doc = self.id_mapping[offset]
            
            results.append({
                "document_id": doc["document_id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": score
            })
            
        return results, latency_ms


def main():
    parser = argparse.ArgumentParser(description="Test RAG Vector Retrieval Service.")
    parser.add_argument("--query", type=str, help="Text query to search for.")
    parser.add_argument("--voice", type=str, help="Path to audio file for voice RAG search.")
    parser.add_argument("--k", type=int, default=5, help="Number of Top-K results to retrieve.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive search shell.")
    parser.add_argument("--rag", action="store_true", help="Enable LLM RAG generation and grounding.")
    
    args = parser.parse_args()
    
    # Conditionally load services
    orchestrator = None
    service = None
    try:
        if args.rag:
            from retrieval.rag_orchestrator import RAGOrchestrator
            orchestrator = RAGOrchestrator()
        else:
            service = RetrievalService()
            service.load()
    except Exception as e:
        print(f"Error loading service: {e}")
        return
        
    if args.voice:
        if args.rag:
            print(f"\nQuerying Voice RAG with audio file: '{args.voice}' (K={args.k})")
            res = orchestrator.query_rag_voice(args.voice, k=args.k)
            print(f"\nTranscript: '{res['transcript']}'")
            print(f"Answer:\n{res['answer']}")
            print(f"\nRelevance Passed: {res['relevance_passed']} (Best Score: {res['best_score']:.4f})")
            print(f"Latency: STT={res['latency_ms']['stt']:.2f} ms | Retrieval={res['latency_ms']['retrieval']:.2f} ms | Gen={res['latency_ms']['generation']:.2f} ms | Total={res['latency_ms']['total_rag']:.2f} ms")
        else:
            from retrieval.stt_service import STTService
            stt = STTService()
            print(f"\nTranscribing audio file: '{args.voice}'...")
            transcript, stt_latency = stt.transcribe(args.voice)
            print(f"Transcript: '{transcript}' (STT Latency: {stt_latency:.2f} ms)")
            
            print(f"\nSearching for: '{transcript}' (K={args.k})")
            results, latency = service.search(transcript, k=args.k)
            print(f"Search completed in {latency:.2f} ms")
            
            print("\nResults:")
            for idx, res in enumerate(results):
                print(f"\n[{idx + 1}] Document ID: {res['document_id']} | Score: {res['score']:.4f}")
                print(f"Text: {res['text']}")
                print(f"Selected: {res['metadata'].get('is_selected')}")
                
    elif args.query:
        if args.rag:
            print(f"\nQuerying RAG: '{args.query}' (K={args.k})")
            res = orchestrator.query_rag(args.query, k=args.k)
            print(f"\nAnswer:\n{res['answer']}")
            print(f"\nRelevance Passed: {res['relevance_passed']} (Best Score: {res['best_score']:.4f})")
            print(f"Latency: Retrieval={res['latency_ms']['retrieval']:.2f} ms | Gen={res['latency_ms']['generation']:.2f} ms | Total={res['latency_ms']['total_rag']:.2f} ms")
        else:
            print(f"\nSearching for: '{args.query}' (K={args.k})")
            results, latency = service.search(args.query, k=args.k)
            print(f"Search completed in {latency:.2f} ms")
            
            print("\nResults:")
            for idx, res in enumerate(results):
                print(f"\n[{idx + 1}] Document ID: {res['document_id']} | Score: {res['score']:.4f}")
                print(f"Text: {res['text']}")
                print(f"Selected: {res['metadata'].get('is_selected')}")
            
    elif args.interactive:
        print("\nInteractive Search Shell (RAG enabled)." if args.rag else "\nInteractive Search Shell.")
        print("Type 'exit' or 'quit' to stop.")
        while True:
            try:
                query = input("\nEnter query > ").strip()
                if not query:
                    continue
                if query.lower() in ["exit", "quit"]:
                    break
                    
                if args.rag:
                    res = orchestrator.query_rag(query, k=args.k)
                    print(f"\nAnswer:\n{res['answer']}")
                    print(f"\nRelevance Passed: {res['relevance_passed']} (Best Score: {res['best_score']:.4f})")
                    print(f"Latency: Retrieval={res['latency_ms']['retrieval']:.2f} ms | Gen={res['latency_ms']['generation']:.2f} ms | Total={res['latency_ms']['total_rag']:.2f} ms")
                else:
                    results, latency = service.search(query, k=args.k)
                    print(f"Search completed in {latency:.2f} ms")
                    for idx, res in enumerate(results):
                        print(f"\n  [{idx + 1}] Document ID: {res['document_id']} (Score: {res['score']:.4f})")
                        print(f"      Selected Flag: {res['metadata'].get('is_selected')}")
                        print(f"      Text: {res['text'][:200]}...")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Default run simple test
        test_query = "what was the immediate impact of the success of the manhattan project?"
        if args.rag:
            print(f"\nRunning default RAG test query: '{test_query}'")
            res = orchestrator.query_rag(test_query, k=3)
            print(f"\nAnswer:\n{res['answer']}")
            print(f"\nRelevance Passed: {res['relevance_passed']} (Best Score: {res['best_score']:.4f})")
            print(f"Latency: Retrieval={res['latency_ms']['retrieval']:.2f} ms | Gen={res['latency_ms']['generation']:.2f} ms | Total={res['latency_ms']['total_rag']:.2f} ms")
        else:
            print(f"\nRunning default test query: '{test_query}'")
            results, latency = service.search(test_query, k=3)
            print(f"Search completed in {latency:.2f} ms")
            for idx, res in enumerate(results):
                print(f"\n[{idx + 1}] Document ID: {res['document_id']} | Score: {res['score']:.4f}")
                print(f"Text: {res['text']}")


if __name__ == "__main__":
    main()

