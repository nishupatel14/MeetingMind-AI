"""Convert uploaded audio to a consistent WAV format for transcription.

Uses native ffmpeg subprocess for fast conversion (seconds vs minutes).
"""

import subprocess
import sys
from pathlib import Path


def convert_to_wav(source_path, destination_path, sample_rate=16000):
    """Convert an audio file to mono, 16-bit PCM WAV using native ffmpeg.

    This is dramatically faster than PyAV frame-by-frame Python processing:
    - PyAV loop:  ~15–20 minutes for a 2-hour file
    - ffmpeg CLI: ~2–3 seconds for a 2-hour file

    Parameters
    ----------
    source_path : str | Path
        Path to the input audio file (mp3, m4a, flac, wav, etc.)
    destination_path : str | Path
        Output path for the converted WAV file.
    sample_rate : int
        Target audio sample rate (default: 16000 Hz for Whisper).
    """
    source_path = Path(source_path)
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[AudioConverter] Converting: {source_path.name} -> {destination_path.name}")

    cmd = [
        "ffmpeg",
        "-y",                        # Overwrite output without asking
        "-i", str(source_path),      # Input file
        "-vn",                       # No video stream
        "-ac", "1",                  # Mono channel
        "-ar", str(sample_rate),     # Sample rate (16000 Hz)
        "-acodec", "pcm_s16le",      # 16-bit PCM encoding
        str(destination_path),       # Output file
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        print(f"[AudioConverter] Conversion complete: {destination_path.name}")
    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode(errors="replace")
        raise RuntimeError(
            f"[AudioConverter] ffmpeg conversion failed for '{source_path.name}'.\n"
            f"ffmpeg error:\n{stderr_output}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "[AudioConverter] ffmpeg not found. "
            "Please install ffmpeg and add it to your system PATH.\n"
            "Download: https://ffmpeg.org/download.html"
        )

