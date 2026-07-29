import modal

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "--index-url", "https://download.pytorch.org/whl/cu124",
        "torch", "torchaudio",
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
    from server import app
    return app