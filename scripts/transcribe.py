"""Transcribes WAV files for audio-berry's accuracy checks, entirely locally.

Reads one path per line on stdin and writes one line of text per path on stdout.
Run with: uv run --no-project --with faster-whisper python scripts/transcribe.py
"""

import subprocess
import sys

import numpy as np
from faster_whisper import WhisperModel

model = WhisperModel("base.en", device="cpu", compute_type="int8", cpu_threads=8)
print("ready", flush=True)

for line in sys.stdin:
    # Decoded with ffmpeg because faster-whisper's bundled decoder can break
    # against newer releases of its av dependency.
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", line.rstrip("\n"), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    segments, _ = model.transcribe(audio, language="en", beam_size=1, condition_on_previous_text=False)
    text = " ".join(segment.text.strip() for segment in segments)
    print(" ".join(text.split()), flush=True)
