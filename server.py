import asyncio
import io
import json
import os
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import instructor
from openai import AsyncOpenAI
from faster_whisper import WhisperModel

# ==========================================
# 1. INITIALIZE MODELS & CONFIG
# ==========================================

# A. Load Silero VAD Model
print("Loading Silero VAD...")
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

# B. Load Faster-Whisper Model
# Options: "tiny", "base", "small", "medium", "large-v3"
# Use device="cuda" and compute_type="float16" if GPU is available
print("Loading Faster-Whisper ASR...")
asr_model = WhisperModel("large-v3", device="gpu", compute_type="float32")

# ==========================================
# 2. PYDANTIC SCHEMAS FOR STRUCTURED ORDERING
# ==========================================

def transcribe_audio_chunk(pcm_bytes: bytes) -> str:
    """Converts 16kHz PCM S16LE bytes to float32 numpy array and transcribes using Faster-Whisper."""
    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    # Gain normalization: scale to use full dynamic range
    max_val = np.abs(audio_float32).max()
    if max_val > 0 and max_val < 0.5:
        audio_float32 = audio_float32 * (0.5 / max_val)

    # Trim leading/trailing silence to reduce noise
    non_silent = audio_float32[np.abs(audio_float32) > 0.01]
    if len(non_silent) > 0 and len(non_silent) < len(audio_float32):
        audio_float32 = non_silent

    hotwords = "Matcha Latte Espresso Cappuccino Caramel Macchiato Phở Gyoza Extra Boba Oat Milk Almond Milk Less Ice Extra Hot"

    segments, info = asr_model.transcribe(
        audio_float32,
        language="en",
        task="transcribe",
        beam_size=5,
        best_of=5,
        hotwords=hotwords,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        without_timestamps=True,
    )

    text = " ".join([segment.text for segment in segments]).strip()
    return text

# ==========================================
# 4. FASTAPI & WEBSOCKET ENGINE
# ==========================================

app = FastAPI()

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected over WebSocket.")
    
# Session state
    audio_buffer = bytearray()
    speech_buffer = bytearray()
    is_speech_active = False
    silence_frames = 0
    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2  # 16-bit PCM

    # 512 samples = ~32ms per frame for Silero VAD
    FRAME_SIZE_SAMPLES = 512
    FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * BYTES_PER_SAMPLE

    vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)
    vad_iterator.reset_states()

    print("WebSocket audio session started. Awaiting audio chunks...")

    try:
        while True:
            # Receive PCM 16kHz audio chunk from client
            chunk = await websocket.receive_bytes()
            audio_buffer.extend(chunk)

            # Process in 512-sample frames for VAD
            while len(audio_buffer) >= FRAME_SIZE_BYTES:
                frame_bytes = bytes(audio_buffer[:FRAME_SIZE_BYTES])
                audio_buffer = audio_buffer[FRAME_SIZE_BYTES:]

                # Convert bytes to tensor for Silero VAD
                int16_frame = np.frombuffer(frame_bytes, dtype=np.int16)
                float32_frame = torch.from_numpy(int16_frame.astype(np.float32) / 32768.0)

                # Run VAD check
                speech_dict = vad_iterator(float32_frame, return_seconds=False)

                if speech_dict:
                    if 'start' in speech_dict:
                        print("--> Speech Started...")
                        is_speech_active = True
                        speech_buffer = bytearray()
                        await websocket.send_json({"status": "SPEAKING"})

                    if 'end' in speech_dict:
                        print("--> Speech Stopped. Triggering Processing Pipeline...")
                        is_speech_active = False
                        try:
                            await websocket.send_json({"status": "PROCESSING"})
                        except (WebSocketDisconnect, RuntimeError):
                            print("Client disconnected before transcription.")
                            vad_iterator.reset_states()
                            continue

                        # Use accumulated speech buffer for transcription
                        full_audio_bytes = bytes(speech_buffer)

                        # Step A: Run Faster-Whisper ASR in a thread to avoid blocking the event loop
                        raw_text = await asyncio.to_thread(transcribe_audio_chunk, full_audio_bytes)
                        print(f"[ASR Transcript]: '{raw_text}'")

                        if raw_text:
                            try:
                                await websocket.send_json({"type": "TRANSCRIPT", "text": raw_text})
                            except (WebSocketDisconnect, RuntimeError):
                                print("Client disconnected during transcription.")
                                vad_iterator.reset_states()
                                continue

                        vad_iterator.reset_states()

                # Accumulate frames while speech is active (or until speech end detected)
                if is_speech_active or speech_dict:
                    speech_buffer.extend(frame_bytes)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error in socket session: {e}")

# Frontend HTML for POC Testing
@app.get("/")
async def get_index():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)