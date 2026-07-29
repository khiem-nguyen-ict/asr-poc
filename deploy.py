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


def _load_module(name, path):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@app.function(gpu="t4", image=image, timeout=3600)
@modal.asgi_app()
def serve():
    import os
    root = os.path.dirname(os.path.abspath(__file__))
    _load_module("server", os.path.join(root, "server.py"))
    import server
    return server.app
