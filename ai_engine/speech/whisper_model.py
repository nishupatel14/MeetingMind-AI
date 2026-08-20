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
    """Singleton loader for faster-whisper."""

    _model = None
    _device_used = None

    @classmethod
    def get_model(cls):
        """Load and cache the faster-whisper model (GPU or CPU fallback)."""

        if cls._model is not None:
            return cls._model

        import torch

        device = "cuda" if WHISPER_DEVICE.lower().startswith("cuda") and torch.cuda.is_available() else "cpu"
        device_index = int(WHISPER_DEVICE_INDEX) if device == "cuda" else 0

        if device == "cuda":
            try:
                cls._model = WhisperModel(
                    WHISPER_MODEL,
                    device="cuda",
                    device_index=device_index,
                    compute_type=WHISPER_COMPUTE_TYPE,
                    cpu_threads=NUM_CORES,
                    download_root=str(MODELS_FOLDER),
                )
                cls._device_used = f"cuda:{device_index}"
                print(
                    f"\n[WhisperLoader] ✓ Successfully loaded "
                    f"{WHISPER_MODEL} on GPU "
                    f"{device_index}"
                )
                return cls._model
            except Exception as error:
                print("\n[WhisperLoader] GPU loading failed:")
                print(f"{type(error).__name__}: {error}")
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

        # --------------------------------------------------
        # CPU fallback
        # --------------------------------------------------
        print("\n[WhisperLoader] Loading CPU fallback...")
        cls._model = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=NUM_CORES,
            download_root=str(MODELS_FOLDER),
        )
        cls._device_used = "cpu"
        print("[WhisperLoader] ✓ CPU fallback loaded")
        return cls._model

    @classmethod
    def get_device_used(cls):
        return (
            cls._device_used
            if cls._device_used is not None
            else WHISPER_DEVICE
        )

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