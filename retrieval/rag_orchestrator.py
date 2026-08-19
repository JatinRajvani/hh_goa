import os
import sys
import time
from dotenv import load_dotenv

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

from retrieval.search import RetrievalService
from retrieval.llm_service import LLMService
from retrieval.stt_service import STTService


# Load environment variables
load_dotenv()

FALLBACK_MESSAGES = {
    "en": "I couldn't find sufficient information in the provided knowledge base to answer that reliably.",
    "hi": "मुझे प्रदान की गई जानकारी में इसका उत्तर नहीं मिला।",
    "gu": "મને પ્રદાન કરેલી માહિતીમાં આનો ઉત્તર મળ્યો નથી.",
    "ta": "வழங்கப்பட்ட அறிவுத் தளத்தில் பதிலளிக்க போதுமான தகவல் கிடைக்கவில்லை.",
    "mr": "दिलेल्या माहितीमध्ये याचे उत्तर सापडले नाही.",
    "ur": "فراہم کردہ معلومات میں اس کا جواب نہیں ملا۔",
    "bn": "প্রদত্ত তথ্যের মধ্যে এর উত্তর পাওয়া যায়নি।",
    "kn": "ಒದಗಿಸಿದ ಮಾಹಿತಿ ಮೂಲಗಳಲ್ಲಿ ಇದಕ್ಕೆ ಉತ್ತರ ಕಂಡುಬಂದಿಲ್ಲ.",
    "ml": "നൽകിയിട്ടുള്ള വിവരങ്ങളിൽ ഇതിനുള്ള ഉത്തരം കണ്ടെത്താനായില്ല.",
    "pa": "ਦਿੱਤੀ ਗਈ ਜਾਣਕਾਰੀ ਵਿੱਚ ਇਸਦਾ ਉੱਤਰ ਨਹੀਂ ਮਿਲਿਆ।",
    "or": "ପ୍ରଦତ୍ତ ସୂଚନା ମଧ୍ୟରେ ଏହାର ଉତ୍ତର ମିଳିଲା ନାହିଁ ।",
    "as": "প্ৰদান কৰা তথ্যৰ ভিতৰত ইয়াৰ উত্তৰ পোৱা নগ’ল।",
    "sa": "प्रदत्तायां सूचनायां अस्य उत्तरं न प्राप्तम्।",
    "ne": "प्रदान गरिएको जानकारीमा यसको उत्तर भेटिएन."
}


class RAGOrchestrator:
    def __init__(self, index_dir="data/index"):
        self.retrieval_service = RetrievalService(index_dir=index_dir)
        self.retrieval_service.load()
        self.llm_service = LLMService()
        self.stt_service = STTService()
        
        # Load threshold setting (default to 0.60)
        try:
            self.threshold = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.60"))
        except ValueError:
            self.threshold = 0.60
            
        print(f"RAG Orchestrator initialized. Relevance threshold: {self.threshold}")
        
    def _build_prompt(self, query: str, passages: list[dict], lang: str = "en") -> str:
        """Constructs the grounding prompt for the LLM."""
        context_blocks = []
        for idx, p in enumerate(passages):
            context_blocks.append(f"Source [{idx+1}] (Doc ID: {p['document_id']}):\n{p['text']}")
            
        context_str = "\n\n".join(context_blocks)
        
        fallback_msg = FALLBACK_MESSAGES.get(lang, FALLBACK_MESSAGES["en"])
        lang_names = {
            "en": "English", "hi": "Hindi", "gu": "Gujarati", "ta": "Tamil",
            "mr": "Marathi", "ur": "Urdu", "bn": "Bengali", "kn": "Kannada",
            "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
            "sa": "Sanskrit", "ne": "Nepali"
        }
        target_lang = lang_names.get(lang, "English")
        
        prompt = f"""You are a helpful, factual assistant. Your task is to answer the user query based on the provided sources below.

Rules:
1. Answer the query in {target_lang} using ONLY the facts explicitly mentioned or semantically supported by the sources. 
2. You are allowed to use semantic reasoning and synonyms (for example, you can connect medical terms like "thoracic spine" to layman terms like "middle back" if the context supports it).
3. Do NOT extrapolate, speculate, or use unrelated outside knowledge not supported by the sources.
4. If the sources are completely unrelated, or do not contain enough information to formulate a valid answer to the query, you must reply exactly with: "{fallback_msg}"
5. Keep the answer concise and direct.

Sources:
{context_str}

Query: {query}

Answer:"""
        return prompt

    def query_rag(self, query: str, k: int = 5, lang: str = "en", use_llm: bool = False) -> dict:
        """
        Coordinates full RAG flow: Retrieval -> Relevance check -> Grounding prompt -> LLM generation.
        """
        start_time = time.time()
        fallback_msg = FALLBACK_MESSAGES.get(lang, FALLBACK_MESSAGES["en"])
        
        # 1. Retrieve matching documents
        results, search_latency_ms = self.retrieval_service.search(query, k=k, lang=lang)
        
        # Check relevance: if empty results or best score is below threshold
        best_score = results[0]["score"] if results else 0.0
        
        if not results or best_score < self.threshold:
            # Short-circuit and return fallback immediately (no LLM call)
            total_latency = (time.time() - start_time) * 1000
            return {
                "query": query,
                "answer": fallback_msg,
                "sources": results,
                "latency_ms": {
                    "retrieval": search_latency_ms,
                    "generation": 0.0,
                    "total_rag": total_latency
                },
                "relevance_passed": False,
                "best_score": best_score
            }
            
        # Fast Extractive Mode: Bypasses LLM API call entirely
        if not use_llm:
            total_latency = (time.time() - start_time) * 1000
            return {
                "query": query,
                "answer": results[0]["text"] if results else fallback_msg,
                "sources": results,
                "latency_ms": {
                    "retrieval": search_latency_ms,
                    "generation": 0.0,
                    "total_rag": total_latency
                },
                "relevance_passed": True,
                "best_score": best_score
            }
            
        # 2. Build context prompt
        prompt = self._build_prompt(query, results, lang=lang)
        
        # 3. Call LLM
        try:
            answer, llm_latency_ms = self.llm_service.generate(prompt)
        except Exception as e:
            # Fallback or error logging
            print(f"Error during LLM generation: {e}")
            total_latency = (time.time() - start_time) * 1000
            return {
                "query": query,
                "answer": f"Generation Error: {e}",
                "sources": results,
                "latency_ms": {
                    "retrieval": search_latency_ms,
                    "generation": 0.0,
                    "total_rag": total_latency
                },
                "relevance_passed": True,
                "best_score": best_score
            }
            
        total_latency = (time.time() - start_time) * 1000
        
        # Double check if the LLM output says it couldn't find the answer (post-generation grounding fallback check)
        # Even if the context was passed, the LLM might decide the passages don't actually support the specific question.
        cleaned_answer = answer.strip()
        
        return {
            "query": query,
            "answer": cleaned_answer,
            "sources": results,
            "latency_ms": {
                "retrieval": search_latency_ms,
                "generation": llm_latency_ms,
                "total_rag": total_latency
            },
            "relevance_passed": True,
            "best_score": best_score
        }

    def query_rag_voice(self, audio_path: str, k: int = 5, lang: str = "en", use_llm: bool = False) -> dict:
        """
        Coordinates full voice RAG flow: Speech-to-Text -> Text RAG -> Return structured response.
        """
        # 1. Transcribe audio to text
        try:
            # Pass language code hint to ElevenLabs (en, hi, gu)
            transcript, stt_latency = self.stt_service.transcribe(audio_path, language_code=lang)
        except Exception as e:
            print(f"Error during Speech-to-Text conversion: {e}")
            fallback_msg = FALLBACK_MESSAGES.get(lang, FALLBACK_MESSAGES["en"])
            return {
                "query": "",
                "transcript": f"[STT Error: {e}]",
                "answer": f"Speech-to-Text Error: {e}",
                "sources": [],
                "latency_ms": {
                    "stt": 0.0,
                    "retrieval": 0.0,
                    "generation": 0.0,
                    "total_rag": 0.0
                },
                "relevance_passed": False,
                "best_score": 0.0
            }
            
        # 2. Run text-based RAG query
        res = self.query_rag(transcript, k=k, lang=lang, use_llm=use_llm)
        
        # 3. Add voice transcription details and adjust latency
        res["transcript"] = transcript
        res["latency_ms"]["stt"] = stt_latency
        res["latency_ms"]["total_rag"] += stt_latency
        
        return res


if __name__ == "__main__":
    # Test execution
    try:
        orchestrator = RAGOrchestrator()
        test_queries = [
            "does bj's accept food stamps",
            "what is the capital of india"
        ]
        
        for q in test_queries:
            print(f"\n--- Querying RAG: '{q}' ---")
            res = orchestrator.query_rag(q, k=3)
            print(f"Answer: {res['answer']}")
            print(f"Relevance passed: {res['relevance_passed']} (Best score: {res['best_score']:.4f})")
            print(f"Latency: Retrieval={res['latency_ms']['retrieval']:.2f}ms, Gen={res['latency_ms']['generation']:.2f}ms, Total={res['latency_ms']['total_rag']:.2f}ms")
    except Exception as e:
        print(f"Orchestrator test failed: {e}")
