import os
import sys
import time
from dotenv import load_dotenv

# Ensure parent directory is in path to allow relative imports when run as script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv()

class STTService:
    def __init__(self):
        # Support various environment variable naming patterns
        self.api_key = (
            os.getenv("eleven_lab_api") or 
            os.getenv("ELEVEN_LAB_API") or 
            os.getenv("ELEVENLABS_API_KEY")
        )
        
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API Key not found. Please set eleven_lab_api in your .env file."
            )
            
        # Initialize ElevenLabs client
        self.client = ElevenLabs(api_key=self.api_key)
        print("ElevenLabs STT Service initialized successfully.")
        
    def transcribe(self, audio_path: str, language_code: str = None) -> tuple[str, float]:
        """
        Transcribes the given audio file using ElevenLabs Scribe v2.
        Returns:
            text: str
            latency_ms: float
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        start_time = time.time()
        
        try:
            print(f"Uploading audio file '{audio_path}' to ElevenLabs (Language hint: {language_code})...")
            with open(audio_path, "rb") as audio_file:
                result = self.client.speech_to_text.convert(
                    file=audio_file,
                    model_id="scribe_v2",
                    tag_audio_events=False,
                    diarize=False,
                    language_code=language_code
                )
            text = result.text.strip()
        except Exception as e:
            raise RuntimeError(f"Error calling ElevenLabs STT API: {e}")
            
        latency_ms = (time.time() - start_time) * 1000
        return text, latency_ms

if __name__ == "__main__":
    # Test tool for STT Service
    if len(sys.argv) < 2:
        print("Usage: python stt_service.py <path_to_audio_file>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    try:
        service = STTService()
        print(f"Transcribing '{audio_file}'...")
        text, latency = service.transcribe(audio_file)
        print(f"\nTranscribed Text:\n{text}")
        print(f"\nLatency: {latency:.2f} ms")
    except Exception as e:
        print(f"STT test failed: {e}")
