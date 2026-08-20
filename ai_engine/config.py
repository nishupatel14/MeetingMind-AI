"""
Project configuration settings.
"""

from pathlib import Path

import torch

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET = PROJECT_ROOT / "datasets" / "raw" / "meetings"
PROCESSED_DATASET = PROJECT_ROOT / "datasets" / "processed"

WAV_FOLDER = PROCESSED_DATASET / "wav"
METADATA_FOLDER = PROCESSED_DATASET / "metadata"
TRANSCRIPT_FOLDER = PROCESSED_DATASET / "transcripts"
ACTION_ITEMS_FOLDER = PROCESSED_DATASET / "action_items"

# ==========================
# Audio Settings
# ==========================

SUPPORTED_FORMATS = [".mp3", ".wav", ".m4a", ".flac"]

TARGET_SAMPLE_RATE = 16000

MAX_FILE_SIZE_MB = 500

MIN_DURATION_SECONDS = 10

MAX_DURATION_SECONDS = 3 * 60 * 60

# ==========================
# Hybrid Device Configuration
# ─────────────────────────────────────────────────────
# Whisper  : faster-whisper on CPU (multi-threaded, no VRAM used)
# NLP      : Llama 3.3 70B / Gemini 3.6 Flash / Qwen 2.5 local model
# Converter: Native ffmpeg CLI (2–3 seconds for any file size)
# ==========================

import os
import sys
import multiprocessing

# Allow CUDA device 0 to be available for local NLP fallback (Qwen on GPU)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Register PyTorch CUDA DLLs on Windows for CTranslate2 / Faster-Whisper
if sys.platform == "win32" and torch.cuda.is_available():
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib):
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(torch_lib)
            except Exception:
                pass
        os.environ["PATH"] = torch_lib + os.path.pathsep + os.environ.get("PATH", "")

# Use all CPU cores for Whisper transcription and PyTorch operations
NUM_CORES = multiprocessing.cpu_count()
torch.set_num_threads(NUM_CORES)
os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)

# ── Hybrid Architecture Device Settings ──
if torch.cuda.is_available():
    DEVICE = "cuda"
    TORCH_DTYPE = torch.float16
    HF_DEVICE = 0
    WHISPER_DEVICE = "cuda"
    WHISPER_DEVICE_INDEX = 1
    WHISPER_COMPUTE_TYPE = "float16"
    WHISPER_MODEL = "large-v3-turbo"
    ACTION_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    gpu_name = torch.cuda.get_device_name(0)
    EXEC_MODE_STR = f"Hybrid Architecture [Whisper=GPU ({WHISPER_MODEL} device=cuda device_index=1) | NLP Engine=Qwen 2.5 3B | GPU={gpu_name}]"
else:
    WHISPER_DEVICE = "cpu"
    WHISPER_DEVICE_INDEX = 0
    WHISPER_COMPUTE_TYPE = "int8"
    WHISPER_MODEL = "small"
    DEVICE = "cpu"
    TORCH_DTYPE = torch.float32
    HF_DEVICE = -1
    ACTION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    EXEC_MODE_STR = "Hybrid Architecture [Whisper=CPU (small INT8) | Local CPU]"

MODELS_FOLDER = PROJECT_ROOT / "models"
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

# Whisper Transcription Settings
WHISPER_BEAM_SIZE = 1
WHISPER_LANGUAGE = "en"
WHISPER_VAD_FILTER = True
WHISPER_MIN_SILENCE_DURATION_MS = 500







# ==========================
# NLP Engine Model Settings
# ==========================

# Load .env file if available
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"\' ')  # .env always wins over system env

# Master Model Service Disable Toggle
DISABLE_CLOUD_API = os.environ.get("DISABLE_CLOUD_API", "false").lower() in {"true", "1", "yes"}

# Primary Integrated Model (Llama 3.3 70B)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
USE_GROQ_API = not DISABLE_CLOUD_API and bool(GROQ_API_KEY) and GROQ_API_KEY not in {"your-groq-api-key-here", ""}

# Secondary Integrated Model (Gemini 3.6 Flash)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash").strip()
USE_GEMINI_API = not DISABLE_CLOUD_API and bool(GEMINI_API_KEY) and GEMINI_API_KEY not in {"your-gemini-api-key-here", ""}

# Fallback Integrated Model (GPT-4o Mini)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
USE_OPENAI_API = not DISABLE_CLOUD_API and bool(OPENAI_API_KEY) and OPENAI_API_KEY not in {"your-openai-api-key-here", ""}

USE_CLOUD_API = USE_GROQ_API or USE_GEMINI_API or USE_OPENAI_API

SUMMARIZER_MODEL = "Falconsai/text_summarization"

# Local NLP Model: Qwen 2.5 1.5B Instruct on GPU (VRAM optimized), 0.5B on CPU
if torch.cuda.is_available():
    ACTION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # Highly intelligent 1.5B Instruct model (fits in GPU VRAM alongside Whisper)
else:
    ACTION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # Lightweight 0.5B model for CPU




TOPICS_FOLDER = (
    PROCESSED_DATASET /
    "topics"
)

SUMMARY_FOLDER = PROCESSED_DATASET / "summary"

DECISIONS_FOLDER = PROCESSED_DATASET / "decisions"

FUTURE_TOPICS_FOLDER = PROCESSED_DATASET / "future_topics"

OPEN_QUESTIONS_FOLDER = PROCESSED_DATASET / "open_questions"

KEY_INSIGHTS_FOLDER = PROCESSED_DATASET / "key_insights"

KEY_DISCUSSION_FOLDER = PROCESSED_DATASET / "key_discussion"

NEXT_STEPS_FOLDER = PROCESSED_DATASET / "next_steps"

REPORT_FOLDER = PROCESSED_DATASET / "reports"

PDF_REPORT_FOLDER = PROCESSED_DATASET / "pdf_reports"

PDF_REPORT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

print(f"[MeetingMind AI] Execution Mode: {EXEC_MODE_STR}")
if USE_GROQ_API:
    print(f"[MeetingMind AI] NLP Engine    : Llama 3.3 70B Model ({GROQ_MODEL})")
    if USE_GEMINI_API:
        print(f"[MeetingMind AI] NLP Fallback  : Gemini 3.6 Flash Model ({GEMINI_MODEL})")
    elif USE_OPENAI_API:
        print(f"[MeetingMind AI] NLP Fallback  : GPT-4o Mini Model ({OPENAI_MODEL})")
elif USE_GEMINI_API:
    print(f"[MeetingMind AI] NLP Engine    : Gemini 3.6 Flash Model ({GEMINI_MODEL})")
    if USE_OPENAI_API:
        print(f"[MeetingMind AI] NLP Fallback  : GPT-4o Mini Model ({OPENAI_MODEL})")
elif USE_OPENAI_API:
    print(f"[MeetingMind AI] NLP Engine    : GPT-4o Mini Model ({OPENAI_MODEL})")
else:
    print(f"[MeetingMind AI] NLP Engine    : Local PyTorch Model ({ACTION_MODEL})")
