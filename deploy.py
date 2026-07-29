import modal
import os

project_root = os.path.dirname(os.path.abspath(__file__))

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
    .add_local_file(os.path.join(project_root, "server.py"), remote_path="/root/server.py", copy=True)
    .add_local_file(os.path.join(project_root, "index.html"), remote_path="/root/index.html", copy=True)
)

app = modal.App("asr-poc")


@app.function(gpu="t4", image=image, timeout=3600)
@modal.asgi_app()
def serve():
    from server import app
    return app
