import modal

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .run_commands(
        "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install(
        "numpy<2",
        "fastapi",
        "uvicorn",
        "instructor",
        "openai",
        "ollama",
        "faster-whisper",
        "pydantic",
        "websockets",
    )
)

app = modal.App("asr-poc")


@app.function(gpu="t4", image=image, timeout=3600)
@modal.asgi_app()
def serve():
    import os
    import sys
    import numpy as np
    import torch
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    import asyncio
    import io
    import json
    import instructor
    from openai import AsyncOpenAI
    from faster_whisper import WhisperModel

    print("Loading Silero VAD...")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    print("Loading Faster-Whisper ASR...")
    asr_model = WhisperModel("large-v3", device="cuda", compute_type="float32")

    def transcribe_audio_chunk(pcm_bytes: bytes) -> str:
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        max_val = np.abs(audio_float32).max()
        if max_val > 0 and max_val < 0.5:
            audio_float32 = audio_float32 * (0.5 / max_val)
        non_silent = audio_float32[np.abs(audio_float32) > 0.01]
        if len(non_silent) > 0 and len(non_silent) < len(audio_float32):
            audio_float32 = non_silent
        hotwords = "Matcha Latte Espresso Cappuccino Caramel Macchiato Phĩ Gyoza Extra Boba Oat Milk Almond Milk Less Ice Extra Hot"
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

    asgi_app = FastAPI()

    @asgi_app.websocket("/ws/audio")
    async def websocket_audio_endpoint(websocket: WebSocket):
        await websocket.accept()
        print("Client connected over WebSocket.")
        audio_buffer = bytearray()
        speech_buffer = bytearray()
        is_speech_active = False
        silence_frames = 0
        SAMPLE_RATE = 16000
        BYTES_PER_SAMPLE = 2
        FRAME_SIZE_SAMPLES = 512
        FRAME_SIZE_BYTES = FRAME_SIZE_SAMPLES * BYTES_PER_SAMPLE
        vad_iterator = VADIterator(vad_model, sampling_rate=SAMPLE_RATE)
        vad_iterator.reset_states()
        print("WebSocket audio session started. Awaiting audio chunks...")
        try:
            while True:
                chunk = await websocket.receive_bytes()
                audio_buffer.extend(chunk)
                while len(audio_buffer) >= FRAME_SIZE_BYTES:
                    frame_bytes = bytes(audio_buffer[:FRAME_SIZE_BYTES])
                    audio_buffer = audio_buffer[FRAME_SIZE_BYTES:]
                    int16_frame = np.frombuffer(frame_bytes, dtype=np.int16)
                    float32_frame = torch.from_numpy(int16_frame.astype(np.float32) / 32768.0)
                    speech_dict = vad_iterator(float32_frame, return_seconds=False)
                    if speech_dict:
                        if "start" in speech_dict:
                            print("--> Speech Started...")
                            is_speech_active = True
                            speech_buffer = bytearray()
                            await websocket.send_json({"status": "SPEAKING"})
                        if "end" in speech_dict:
                            print("--> Speech Stopped. Triggering Processing Pipeline...")
                            is_speech_active = False
                            try:
                                await websocket.send_json({"status": "PROCESSING"})
                            except (WebSocketDisconnect, RuntimeError):
                                print("Client disconnected before transcription.")
                                vad_iterator.reset_states()
                                continue
                            full_audio_bytes = bytes(speech_buffer)
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
                    if is_speech_active or speech_dict:
                        speech_buffer.extend(frame_bytes)
        except WebSocketDisconnect:
            print("Client disconnected.")
        except Exception as e:
            print(f"Error in socket session: {e}")

    @asgi_app.get("/")
    async def get_index():
        html = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Voice Order POC</title>
<style>
body{font-family:sans-serif;margin:40px;background:#f4f6f8}
.card{background:white;padding:24px;border-radius:8px;max-width:600px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}
button{padding:12px 24px;font-size:16px;border:none;border-radius:4px;cursor:pointer}
.start{background:#10b981;color:white}
.stop{background:#ef4444;color:white}
.status{margin-top:10px;font-weight:bold;color:#64748b}
</style></head>
<body><div class="card">
<h2>High-Precision Voice Order (Backend STT)</h2>
<p>Try saying: <i>"Can I get two hot matcha lattes with oat milk and extra boba?"</i></p>
<button id="recordBtn" class="start" onclick="toggleRecording()">Start Speaking</button>
<div id="status" class="status">Status: Disconnected</div>
<h3>1. Transcribed Speech (ASR):</h3>
<div id="transcript" style="font-size:18px;font-weight:500;color:#1e293b;">-</div>
</div>
<script>
let ws;let audioContext;let processor;let globalStream;let isRecording=false;
async function toggleRecording(){
const btn=document.getElementById("recordBtn");
if(!isRecording){ws=new WebSocket((window.location.protocol==="https:"?"wss":"ws")+"://"+window.location.host+"/ws/audio");
ws.onopen=()=>{document.getElementById("status").innerText="Status: Listening...";startAudioCapture();};
ws.onmessage=(event)=>{const data=JSON.parse(event.data);if(data.status){document.getElementById("status").innerText="Status: "+data.status;}if(data.type==="TRANSCRIPT"){document.getElementById("transcript").innerText=data.text;}};
ws.onclose=()=>{document.getElementById("status").innerText="Status: Disconnected";};
btn.innerText="Stop";btn.className="stop";isRecording=true;
}else{stopAudioCapture();if(ws)ws.close();btn.innerText="Start Speaking";btn.className="start";isRecording=false;}}
async function startAudioCapture(){
globalStream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true},video:false});
audioContext=new(window.AudioContext||window.webkitAudioContext)({sampleRate:16000});
const source=audioContext.createMediaStreamSource(globalStream);
processor=audioContext.createScriptProcessor(2048,1,1);
processor.onaudioprocess=(e)=>{if(!isRecording||ws.readyState!==WebSocket.OPEN)return;
const inputData=e.inputBuffer.getChannelData(0);const pcm16=new Int16Array(inputData.length);
for(let i=0;i<inputData.length;i++){let s=Math.max(-1,Math.min(1,inputData[i]));pcm16[i]=s<0?s*0x8000:s*0x7FFF;};
ws.send(pcm16.buffer);};source.connect(processor);processor.connect(audioContext.destination);};
function stopAudioCapture(){if(processor)processor.disconnect();if(audioContext)audioContext.close();if(globalStream)globalStream.getTracks().forEach(t=>t.stop());}
</script></body></html>"""
        return HTMLResponse(content=html)

    return asgi_app
