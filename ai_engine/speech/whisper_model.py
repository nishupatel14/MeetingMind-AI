"""
Whisper Model Loader — Kaggle / Hybrid GPU Mode

Uses faster-whisper (CTranslate2).

Features:
- Supports CPU or CUDA
- Allows selecting a specific GPU
- Uses GPU 1 when configured for Kaggle
- Uses INT8/FP16 depending on configuration
- Falls back to CPU if GPU loading fails
- Singleton model loading to avoid duplicate VRAM usage
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
        """Load and cache the faster-whisper model."""

        if cls._model is not None:
            return cls._model

        import torch

        # ---------------------------------------------------------
        # Determine device & device index
        # ---------------------------------------------------------

        device = "cpu"
        gpu_index = 0

        if WHISPER_DEVICE.lower().startswith("cuda"):
            if torch.cuda.is_available():
                try:
                    if ":" in WHISPER_DEVICE:
                        gpu_index = int(WHISPER_DEVICE.split(":")[1])
                    else:
                        gpu_index = int(WHISPER_DEVICE_INDEX)


                    torch.cuda.set_device(gpu_index)
                    device = "cuda"

                    print(f"[WhisperLoader] CUDA device selected: device='{device}', device_index={gpu_index}")
                    print(f"[WhisperLoader] GPU name: {torch.cuda.get_device_name(gpu_index)}")
                    total_memory = torch.cuda.get_device_properties(gpu_index).total_memory / (1024 ** 3)
                    print(f"[WhisperLoader] GPU VRAM: {total_memory:.2f} GB")
                except Exception as err:
                    print(f"[WhisperLoader] GPU selection failed: {err}")
                    device = "cpu"
                    gpu_index = 0
            else:
                print("[WhisperLoader] CUDA unavailable. Using CPU.")
                device = "cpu"
                gpu_index = 0

        # ---------------------------------------------------------
        # Print configuration
        # ---------------------------------------------------------

        print("\n" + "=" * 60)
        print("[WhisperLoader] Loading faster-whisper")
        print("=" * 60)

        print(f"Model        : {WHISPER_MODEL}")
        print(f"Device       : {device.upper()}")
        print(f"Device Index : {gpu_index if device == 'cuda' else 'N/A'}")
        print(f"Compute      : {WHISPER_COMPUTE_TYPE}")
        print(f"Threads      : {NUM_CORES}")
        print("=" * 60)

        # ---------------------------------------------------------
        # Load model
        # ---------------------------------------------------------

        try:
            cls._model = WhisperModel(
                WHISPER_MODEL,
                device=device,
                device_index=gpu_index if device == "cuda" else 0,
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=NUM_CORES,
                download_root=str(MODELS_FOLDER),
            )

            cls._device_used = f"{device}:{gpu_index}" if device == "cuda" else "cpu"
            print(f"[WhisperLoader] ✓ Model loaded successfully on {cls._device_used.upper()}")


        except Exception as err:

            print(
                "\n[WhisperLoader] ⚠ GPU/model loading failed:"
            )

            print(err)

            print(
                "[WhisperLoader] Falling back to CPU INT8..."
            )

            # Free CUDA memory
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

            print(
                "[WhisperLoader] ✓ CPU fallback loaded successfully"
            )

        return cls._model

    @classmethod
    def get_device_used(cls):
        """Return the device currently used by Whisper."""

        if cls._device_used is not None:
            return cls._device_used

        return WHISPER_DEVICE

    @classmethod
    def clear_model(cls):
        """
        Completely release the cached Whisper model.
        Useful in notebooks when changing models or GPUs.
        """
        if cls._model is not None:
            print("[WhisperLoader] Releasing Whisper model...")
            del cls._model
            cls._model = None
            cls._device_used = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        print("[WhisperLoader] ✓ Model released")


if __name__ == "__main__":
    WhisperLoader.get_model()