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
        """Load and cache the faster-whisper model (CPU or GPU)."""
        if cls._model is None:
            import torch

            # Determine correct device
            if WHISPER_DEVICE == "cuda":
                if torch.cuda.is_available():
                    try:
                        torch.cuda.set_device(0)  # Ensure device 0
                    except Exception:
                        pass
                    device = "cuda:0"  # Explicit device:0
                else:
                    device = "cpu"
            else:
                device = "cpu"

            print(f"[WhisperLoader] Engine  : faster-whisper (CTranslate2)")
            print(f"[WhisperLoader] Model   : {WHISPER_MODEL}")
            print(f"[WhisperLoader] Device  : {device.upper()}")
            print(f"[WhisperLoader] Compute : {WHISPER_COMPUTE_TYPE}")
            print(f"[WhisperLoader] Threads : {NUM_CORES} CPU cores")

            try:
                cls._model = WhisperModel(
                    WHISPER_MODEL,
                    device=device,  # Use explicit "cuda:0" or "cpu"
                    compute_type=WHISPER_COMPUTE_TYPE,
                    cpu_threads=NUM_CORES,
                    download_root="./models",
                )
                cls._device_used = device
                print(f"[WhisperLoader] ✓ Loaded faster-whisper successfully on {device.upper()}")
            except Exception as err:
                print(f"[WhisperLoader] ⚠ Warning: {device.upper()} failed ({err}). Falling back to CPU...")
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                cls._model = WhisperModel(
                    "base",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=NUM_CORES,
                    download_root="./models",
                )
                cls._device_used = "cpu"
                print("[WhisperLoader] ✓ Loaded faster-whisper successfully on CPU fallback")

        return cls._model

    @classmethod
    def get_device_used(cls):
        return cls._device_used if cls._device_used is not None else WHISPER_DEVICE


if __name__ == "__main__":
    WhisperLoader.get_model()