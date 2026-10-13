import os
import io
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from groq import Groq
from gtts import gTTS

app = FastAPI(title="TestMu Voice QA Assistant API")

# Enable CORS for Vercel integration and expose custom response headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-User-Transcript", "X-Bot-Reply"]
)

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.get("/")
def health_check():
    return {"status": "ok", "service": "TestMu Voice QA Assistant API"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/voice-agent")
async def voice_agent(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY environment variable is missing on backend."
        )

    groq_client = Groq(api_key=GROQ_API_KEY)

    # 1. Read and validate incoming audio file content
    content = await file.read()
    if not content or len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio file was empty. Hold button down while speaking."
        )

    temp_audio_path = None
    try:
        # Determine temporary file extension
        filename = file.filename or "audio.webm"
        suffix = f".{filename.split('.')[-1]}" if "." in filename else ".webm"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        # 2. Transcribe Audio via Groq Whisper API
        with open(temp_audio_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                response_format="text"
            )
        
        user_transcript = transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
        if not user_transcript:
            user_transcript = "No audible voice recognized."

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio Transcription Error: {str(e)}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    # 3. Handle QA Intent Simulation
    lowered_transcript = user_transcript.lower()
    simulated_context = ""
    
    if any(keyword in lowered_transcript for keyword in ["run", "trigger", "execute", "playwright", "cypress", "selenium"]):
        simulated_context = " Context Note: The user triggered a test run. Simulated status: Triggered Playwright Smoke Suite #1043 on Chrome & Safari."
    elif any(keyword in lowered_transcript for keyword in ["status", "report", "pass", "fail", "build", "metrics"]):
        simulated_context = " Context Note: Simulated status: Suite #1042 completed with 98% pass rate across 50 parallel browser threads."

    # 4. Generate LLM Response via Groq Llama 3.3
    system_prompt = (
        "You are TestMu AI Voice Assistant. Help QA engineers monitor automated test suites, "
        "check pass/fail logs, and trigger test executions. Keep responses concise and conversational "
        f"(maximum 2 short sentences).{simulated_context}"
    )

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Updated to active Groq model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_transcript}
            ],
            temperature=0.6,
            max_tokens=150
        )
        bot_reply = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation Error: {str(e)}")

    # 5. Synthesize TTS with gTTS
    try:
        tts = gTTS(text=bot_reply, lang="en", slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_bytes = mp3_fp.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Synthesis Error: {str(e)}")

    # 6. Sanitize custom HTTP headers (strictly ASCII)
    safe_transcript = user_transcript.encode("ascii", "ignore").decode("ascii") or "Audio processed"
    safe_bot_reply = bot_reply.encode("ascii", "ignore").decode("ascii") or "Response generated"

    headers = {
        "X-User-Transcript": safe_transcript,
        "X-Bot-Reply": safe_bot_reply
    }

    return Response(content=audio_bytes, media_type="audio/mpeg", headers=headers)
