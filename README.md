# ASR PoC

A proof-of-concept real-time voice order system using local LLMs and speech recognition. It captures audio from the browser, detects speech segments with Silero VAD, transcribes them with Faster-Whisper, and extracts structured order data via a local Ollama LLM using Instructor.

## Architecture

The pipeline runs entirely locally:

1. **VAD (Silero VAD)** — Detects speech start/end in 512-sample frames at 16 kHz
2. **ASR (Faster-Whisper)** — Transcribes speech to text with menu-item vocabulary biasing via `initial_prompt`. Uses `device="cuda"` (GPU) with `compute_type="float32"`.
3. **NLU (Ollama + Instructor)** — Extracts structured JSON (`OrderPayload`) from the transcript using the `llama3.1` model

All three stages run over a WebSocket connection (`/ws/audio`) with real-time status updates (`SPEAKING`, `PROCESSING`, `TRANSCRIPT`, `ORDER_RESULT`).

## Environment

- **OS:** macOS (Darwin x86_64)
- **Python:** 3.12.13 (default system Python via Homebrew)
- **Ollama:** 0.32.5 (installed via Homebrew, running on port 11434)
- **Model:** llama3.1 (4.9 GB, Q4_K_M quantized, 8B parameters, 128K context)
- **Working directory:** `/Users/apple/SourceCode/asr-poc`
- **Virtual env:** `.venv/` in the project directory (Python 3.12)

## Dependencies

| Package | Role |
|---|---|
| `fastapi` | Web framework and API server |
| `uvicorn` | ASGI server |
| `instructor` | Structured output extraction from Ollama/LLM |
| `ollama` | Async Ollama client (transitive, via `openai` SDK) |
| `openai` | Async OpenAI-compatible client (used with Ollama) |
| `faster-whisper` | Whisper-based ASR transcription |
| `torch` | PyTorch (required by Silero VAD and Faster-Whisper) |
| `numpy` | Audio array processing |
| `pydantic` | Data validation schemas |
| `silero-vad` | Voice Activity Detection model (loaded via `torch.hub`) |

The browser frontend (`index.html`) uses the Web Audio API and native WebSockets — no additional frontend dependencies.

## Cloud Deployment (Modal — Free GPU)

The app can be deployed to Modal for free GPU testing. This uses a T4 GPU in Modal's cloud — no local GPU needed.

### Prerequisites
- Modal account: [modal.com](https://modal.com) (free signup)
- Modal CLI installed: `pip3 install modal`

### Deploy
```bash
modal token new          # authenticate once
modal deploy deploy.py   # deploys to Modal's T4 GPU
```

This builds a container image, downloads the `large-v3` ASR model, and starts the FastAPI server on a public URL. The output will show the endpoint URL (e.g., `https://khiem-nguyen-ict--asr-poc-serve.modal.run`). Open that URL in your browser to test.

### What Modal deploys
- Image built from `deploy.py` (installation instructions are inside it)
- GPU: T4 (CUDA 12.4)
- Server starts on port 8000, exposed as the Modal endpoint URL
- The `large-v3` model is cached in the container image
- First cold start takes ~30 seconds; subsequent calls are fast

## Setup & Running (Local)

### Prerequisites
- Homebrew (for `ollama` and `python@3.12`)

### Quick start
```bash
# Install Ollama (if not already installed)
brew install ollama
brew services start ollama

# Pull the model
ollama pull llama3.1

# Create and activate a Python 3.12 virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (torch + torchaudio from PyTorch CUDA index for T4 GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# macOS only: set OpenMP workaround (prevents SIGSEGV crashes)
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1

# Run the server
python server.py
```

The server starts on `http://0.0.0.0:8000` by default. Use the `PORT` environment variable to change it:

```bash
PORT=8080 python server.py
```

### What the server does

- **Web UI** at `/` — Click "Start Speaking" to begin voice recording; transcribed text and structured order JSON appear in real time
- **WebSocket audio endpoint** at `/ws/audio` — Accepts 16 kHz PCM S16LE audio bytes

## Menu Vocabulary Bias

The ASR model is biased toward menu items via two mechanisms:

1. **Whisper `initial_prompt`** — A text prompt listing all menu items (`Matcha Latte`, `Espresso`, `Cappuccino`, `Caramel Macchiato`, `Phở`, `Gyoza`, `Extra Boba`, `Oat Milk`, `Almond Milk`, `Less Ice`, `Extra Hot`) is passed to Faster-Whisper to improve recognition of menu terms.
2. **LLM system prompt** — The Instructor prompt corrects phonetic spelling issues (e.g., "macha" → "Matcha Latte") and normalizes items to the canonical menu list.