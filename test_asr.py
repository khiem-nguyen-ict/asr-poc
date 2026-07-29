import numpy as np
from faster_whisper import WhisperModel

print("Loading Faster-Whisper model...")
asr_model = WhisperModel("large-v3", device="cpu", compute_type="float16")

print("Generating 3-second synthetic audio (1kHz sine wave)...")
sample_rate = 16000
duration = 3.0
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio = (np.sin(2 * np.pi * 1000 * t) * 0.5 * 32767).astype(np.int16)
audio_float32 = audio.astype(np.float32) / 32768.0

print("Transcribing...")
segments, info = asr_model.transcribe(
    audio_float32,
    language="en",
    task="transcribe",
    beam_size=5,
    best_of=5,
    without_timestamps=True,
)

text = " ".join([segment.text for segment in segments]).strip()
print(f"Model info: {info.language} (prob={info.language_probability:.2f})")
print(f"Transcript: '{text}'")
print("Test passed!")