"""
Whisper Model Loader — Hybrid Mode (Option 1)

Uses faster-whisper (CTranslate2 engine) on CPU with INT8 quantization.
- Zero GPU VRAM consumption: full 2GB VRAM freed for NLP/Qwen fallback.
- Multi-threaded: uses all available CPU cores automatically.
- INT8 quantization: 2–4x faster CPU inference than standard PyTorch float32.
"""

from faster_whisper import WhisperModel
from ai_engine.config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    NUM_CORES,
)


class WhisperLoader:
    """Singleton loader for faster-whisper CTranslate2 model on CPU."""

    _model = None
    _device_used = None

    @classmethod
    def get_model(cls):
        """Load and cache the faster-whisper model (CPU, INT8)."""
        if cls._model is None:
            print(f"[WhisperLoader] Engine  : faster-whisper (CTranslate2)")
            print(f"[WhisperLoader] Model   : {WHISPER_MODEL}")
            print(f"[WhisperLoader] Device  : {WHISPER_DEVICE.upper()}")
            print(f"[WhisperLoader] Compute : {WHISPER_COMPUTE_TYPE}")
            print(f"[WhisperLoader] Threads : {NUM_CORES} CPU cores")

            device_idx = 0 if WHISPER_DEVICE == "cuda" else 0
            try:
                cls._model = WhisperModel(
                    WHISPER_MODEL,
                    device=WHISPER_DEVICE,
                    device_index=device_idx,
                    compute_type=WHISPER_COMPUTE_TYPE,
                    cpu_threads=NUM_CORES,
                    download_root="./models",
                )
                cls._device_used = WHISPER_DEVICE
                print(f"[WhisperLoader] Loaded faster-whisper successfully on {WHISPER_DEVICE.upper()} (Device {device_idx})")
            except Exception as err:
                print(f"[WhisperLoader] Warning: GPU Whisper init failed ({err}). Falling back to CPU...")
                cls._model = WhisperModel(
                    "base",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=NUM_CORES,
                    download_root="./models",
                )
                cls._device_used = "cpu"
                print("[WhisperLoader] Loaded faster-whisper successfully on CPU fallback")


        return cls._model

    @classmethod
    def get_device_used(cls):
        return cls._device_used if cls._device_used is not None else WHISPER_DEVICE


if __name__ == "__main__":
    WhisperLoader.get_model()