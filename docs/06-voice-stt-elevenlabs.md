# 06 — Voice STT via ElevenLabs

This document outlines the voice capture, audio ingestion, and transcribing workflow, satisfying **Requirement 1 (STT)** of the hackathon.

---

## 🎙️ Why ElevenLabs Scribe?
Transcribing regional Indic scripts requires high accuracy, support for native accents, and code-switching handling (mixing English and regional languages in speech). **ElevenLabs Scribe** provides state-of-the-art multilingual audio transcription.

---

## 🔄 Voice Ingestion Workflow

```
 [ Browser Microphone ] --- (WebM stream) ---> [ FastAPI Server ]
                                                    |
                                           [ Temp Audio File ]
                                                    |
 [ Transcribed Script ] <-- (JSON Response) -- [ Scribe STT API ]
```

### 1. Client-Side Capture
The frontend leverages the HTML5 `MediaRecorder` API to capture microphone inputs. The recorded audio chunks are compiled into a standard **WebM** buffer stream and sent via a multipart Form POST request to `/api/query-voice`.

### 2. FastAPI Multipart Handler
FastAPI handles the file upload, saving the binary payload into a temporary file:
```python
# Save temporary voice file
temp_file_path = f"temp_audio_{uuid.uuid4().hex}.webm"
with open(temp_file_path, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)
```

### 3. ElevenLabs Scribe API Request
The backend invokes the Scribe API via HTTP multipart requests:
```python
# STT Invocation (stt_service.py)
url = "https://api.elevenlabs.io/v1/speech-to-text"
headers = {"xi-api-key": self.api_key}
files = {"file": (filename, file_data, "audio/webm")}
data = {"model_id": "scribe_v1"}

# Optional: Add language ISO code to force script decoding
if language_code:
    data["language_code"] = language_code
```

### 4. Temporary File Lifecycle Safety
To prevent server memory leak and disk bloat, temporary audio files are deleted immediately after transcription, wrapped in a robust `try...finally` block:
```python
try:
    transcription = await stt_service.transcribe(temp_file_path, lang)
finally:
    # Always clean up temp files from disk
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
```
This guarantees that local and cloud storage remain clean, even if the API network connection fails.
