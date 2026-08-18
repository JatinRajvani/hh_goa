import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMService:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        
        if self.provider == "groq":
            self.api_key = os.getenv("GROQ_API_KEY")
            self.model_name = os.getenv("LLM_MODEL", "groq/compound-mini")
            
            if not self.api_key:
                raise ValueError("Groq API Key not found. Please set GROQ_API_KEY in your .env file.")
                
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            print(f"LLM Service: Initialized Groq client with model '{self.model_name}'")
            
        else:  # Default to gemini
            self.api_key = os.getenv("Gemini_API_KEY") or os.getenv("GEMINI_API_KEY")
            self.model_name = os.getenv("LLM_MODEL", "gemini-3.6-flash")
            
            if not self.api_key:
                raise ValueError("Gemini API Key not found. Please set Gemini_API_KEY or GEMINI_API_KEY in your .env file.")
                
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"LLM Service: Initialized Gemini client with model '{self.model_name}'")
            
    def generate(self, prompt: str) -> tuple[str, float]:
        """
        Generates content for a given prompt using the configured LLM provider.
        Returns:
            response_text: str
            latency_ms: float
        """
        start_time = time.time()
        
        try:
            if self.provider == "groq":
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=0.0
                )
                text = response.choices[0].message.content
            else:
                import google.generativeai as genai
                # Set temperature low to maximize adherence to context (grounding)
                generation_config = genai.types.GenerationConfig(
                    temperature=0.0,
                )
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                text = response.text
        except Exception as e:
            raise RuntimeError(f"Error calling {self.provider.capitalize()} API: {e}")
            
        latency_ms = (time.time() - start_time) * 1000
        return text, latency_ms

if __name__ == "__main__":
    # Quick sanity check
    try:
        service = LLMService()
        print("LLM Service initialized successfully.")
        print("Testing basic generation...")
        response, latency = service.generate("Say hello!")
        print(f"Response: '{response.strip().encode('ascii', 'ignore').decode()}' (Latency: {latency:.2f} ms)")
    except Exception as e:
        print(f"LLM Service initialization/test failed: {e}")
