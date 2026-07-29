# ASR PoC

A proof-of-concept real-time voice transcription system. It captures audio from the browser, detects speech segments with Silero VAD, and transcribes them with Faster-Whisper — deployed on Modal's free T4 GPU.

## Architecture

1. **VAD (Silero VAD)** — Detects speech start/end in 512-sample frames at 16 kHz
2. **ASR (Faster-Whisper)** — Transcribes speech to text with menu-item vocabulary biasing

Both stages run over a WebSocket connection (`/ws/audio`) with real-time status updates (`SPEAKING`, `PROCESSING`, `TRANSCRIPT`).

## Deployment

This project runs on **Modal** (free T4 GPU). No local GPU or Ollama needed.

### Deploy
```bash
modal token new          # authenticate once
modal deploy deploy.py   # deploys to Modal's T4 GPU
```

Modal builds a container image, downloads the `large-v3` ASR model, and starts the FastAPI server on a public URL. Open that URL in your browser to test.

- **GPU**: T4 (CUDA 12.4)
- **First cold start**: ~30 seconds (model loading)
- **Subsequent calls**: fast (model stays cached)

## Dependencies

| Package | Role |
|---|---|
| `fastapi` | Web framework and API server |
| `uvicorn` | ASGI server |
| `faster-whisper` | Whisper-based ASR transcription |
| `torch` | PyTorch (required by Silero VAD and Faster-Whisper) |
| `numpy` | Audio array processing |
| `pydantic` | Data validation schemas |
| `silero-vad` | Voice Activity Detection (loaded via `torch.hub`) |

The browser frontend (`index.html`) uses the Web Audio API and native WebSockets — no additional frontend dependencies.

## Menu Vocabulary Bias

The ASR model is biased toward menu items via an `initial_prompt` passed to Faster-Whisper:

`Matcha Latte`, `Espresso`, `Cappuccino`, `Caramel Macchiato`, `Phở`, `Gyoza`, `Extra Boba`, `Oat Milk`, `Almond Milk`, `Less Ice`, `Extra Hot`

## Project Structure

```
asr-poc/
├── deploy.py          # Modal deployment config
├── server.py          # FastAPI + WebSocket ASR server
├── index.html         # Browser frontend (voice recording UI)
├── test_asr.py        # Standalone ASR benchmark script
├── requirements.txt   # Python dependencies (excludes torch/torchaudio)
└── README.md
```