"""
Whisper Model Loader — Kaggle / Hybrid GPU Mode

Uses faster-whisper (CTranslate2).
"""

from faster_whisper import WhisperModel

from ai_engine.config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    NUM_CORES,
    MODELS_FOLDER,
)
try:
    from ai_engine.config import WHISPER_DEVICE_INDEX
except ImportError:
    WHISPER_DEVICE_INDEX = 1


class WhisperLoader:
    """Singleton loader for faster-whisper on dedicated GPU."""

    _model = None
    _device_used = None

    @classmethod
    def get_model(cls):
        """Load and cache the faster-whisper model on GPU."""

        if cls._model is not None:
            return cls._model

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "[WhisperLoader] No CUDA GPU found. "
                "MeetingMind AI Whisper transcriber requires a GPU. "
                "Please enable GPU in your environment."
            )

        device_index = int(WHISPER_DEVICE_INDEX)
        # Clamp to available GPU count
        device_index = min(device_index, torch.cuda.device_count() - 1)

        try:
            print(f"\n[WhisperLoader] Loading {WHISPER_MODEL} directly on GPU (cuda:{device_index})...")
            cls._model = WhisperModel(
                WHISPER_MODEL,
                device="cuda",
                device_index=device_index,
                compute_type=WHISPER_COMPUTE_TYPE,
                download_root=str(MODELS_FOLDER),
            )
            cls._device_used = f"cuda:{device_index}"
            print(
                f"[WhisperLoader] ✓ Successfully loaded "
                f"{WHISPER_MODEL} on GPU {device_index} ({WHISPER_COMPUTE_TYPE})"
            )
            return cls._model
        except Exception as error:
            print(f"\n[WhisperLoader] GPU loading failed on cuda:{device_index}: {error}")
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
            raise RuntimeError(f"[WhisperLoader] Failed to load Whisper on GPU: {error}")

    @classmethod
    def clear_model(cls):
        if cls._model is not None:
            del cls._model
            cls._model = None
            cls._device_used = None

        try:
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
        print("[WhisperLoader] ✓ Model released")


if __name__ == "__main__":
    WhisperLoader.get_model()