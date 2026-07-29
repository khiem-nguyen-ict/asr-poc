import modal

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .run_commands(
        'pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124'
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
    .add_local_file("server.py", remote_path="/root/server.py")
    .add_local_file("index.html", remote_path="/root/index.html")
)

app = modal.App("asr-poc")

@app.function(gpu="t4", image=image, timeout=3600)
@modal.asgi_app()
def serve():
    from server import app
    return app