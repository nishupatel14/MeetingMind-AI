import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import requests
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from ai_engine.config import (
    ACTION_MODEL,
    DEVICE,
    TORCH_DTYPE,
    HF_DEVICE,
    USE_GEMINI_API,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    USE_GROQ_API,
    GROQ_API_KEY,
    GROQ_MODEL,
    USE_OPENAI_API,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


class ActionModelLoader:

    _tokenizer = None
    _model = None
    _model_name = None

    @classmethod
    def unload(cls):
        """Unload the current model and release RAM/VRAM."""
        import gc
        if cls._model is not None:
            del cls._model
            cls._model = None
        if cls._tokenizer is not None:
            del cls._tokenizer
            cls._tokenizer = None
        cls._model_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def load(cls):
        # Reload if model is not loaded OR if ACTION_MODEL configuration has changed
        if cls._model is None or cls._model_name != ACTION_MODEL:
            if cls._model is not None:
                print(f"[ActionModelLoader] Switching model from '{cls._model_name}' to '{ACTION_MODEL}'...")
                cls.unload()

            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            target_dtype = torch.float16 if target_device == "cuda" else torch.float32

            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            # ── Direct GPU Loading ─────────────────────────────────────────
            # Use device_map with explicit "cuda:N" string so HuggingFace
            # streams weights directly into GPU VRAM — no CPU buffer at all.
            # This is the correct approach when dedicated GPU VRAM is available.
            if target_device == "cuda":
                gpu_index  = int(HF_DEVICE) if isinstance(HF_DEVICE, int) and HF_DEVICE >= 0 else 0
                gpu_index  = min(gpu_index, torch.cuda.device_count() - 1)
                device_str = f"cuda:{gpu_index}"
                # String key forces HF to load ALL layers directly to that GPU
                device_map_arg = {"": device_str}
            else:
                device_str     = "cpu"
                device_map_arg = None

            print(f"[ActionModelLoader] Loading Local NLP Model ({ACTION_MODEL}) → {device_str} ...")
            print("=" * 60)

            cls._tokenizer = AutoTokenizer.from_pretrained(
                ACTION_MODEL,
                trust_remote_code=True,
            )

            cls._model = AutoModelForCausalLM.from_pretrained(
                ACTION_MODEL,
                trust_remote_code=True,
                device_map=device_map_arg,  # Streams weights directly to GPU VRAM
                torch_dtype=target_dtype,
            )

            cls._model_name = ACTION_MODEL

            actual_device = str(next(cls._model.parameters()).device)
            print("=" * 60)
            print(f"Local NLP Model ({ACTION_MODEL}) Loaded on {actual_device.upper()}")
            print("=" * 60)

    # Common LLM noise patterns to strip from output
    _NOISE_PATTERNS = [
        r"^(?:Here (?:are|is) (?:the|a|my)[\s\S]*?:)\s*",
        r"^(?:Based on (?:the|this) (?:transcript|meeting|text)[\s\S]*?:)\s*",
        r"^(?:From the (?:transcript|meeting|text)[\s\S]*?:)\s*",
        r"^(?:The (?:transcript|meeting) (?:shows|indicates|reveals)[\s\S]*?:)\s*",
        r"^(?:After (?:reviewing|reading|analyzing)[\s\S]*?:)\s*",
        r"^(?:I (?:have|can) (?:identified|found|extracted)[\s\S]*?:)\s*",
        r"^(?:Sure[,!.]?\s*)",
        r"^(?:Certainly[,!.]?\s*)",
    ]

    @classmethod
    def _clean_output(cls, text):
        """Strip common LLM preamble noise and formatting artifacts."""
        text = text.strip()

        # Remove common preamble patterns
        for pattern in cls._NOISE_PATTERNS:
            text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()

        # Remove duplicate blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove trailing incomplete sentences (cut off by token limit)
        lines = text.rstrip().split("\n")
        if lines:
            last_line = lines[-1].strip()
            # If last line doesn't end with punctuation and is not a structured field, it's likely truncated
            if (
                last_line
                and not last_line[-1] in ".!?)\"]'"
                and not last_line.lower().startswith(("person:", "task:", "deadline:"))
                and not last_line.startswith(("- ", "• ", "* "))
                and len(last_line.split()) > 3
            ):
                lines = lines[:-1]

        return "\n".join(lines).strip()

    @classmethod
    def _truncate_prompt(cls, prompt, max_chars=12000):
        if not prompt or len(prompt) <= max_chars:
            return prompt
        half = max_chars // 2
        return prompt[:half] + "\n\n...[transcript middle section truncated for API token limit]...\n\n" + prompt[-half:]

    @classmethod
    def _generate_gemini(cls, prompt):
        """Generate response via Google Gemini API (Free tier, 1M token context window)."""
        import time

        system_prompt = (
            "You are MeetingMind AI, an intelligent executive meeting analysis assistant.\n"
            "REAL SPOKEN MEETING CONTEXT:\n"
            "- Transcripts come from real multi-cultural spoken meetings with non-native English speakers, spoken disfluencies, fillers ('you can say'), and conversational phrasing.\n"
            "- Focus on the UNDERLYING INTENT, SUBSTANTIVE BUSINESS CONTEXT, AND CORE DECISIONS.\n"
            "STRICT RULES:\n"
            "1. Answer ONLY from the facts and context in the transcript. Do NOT invent outside information.\n"
            "2. Interpret conversational spoken expressions into clear, professional business topics, decisions, and action items.\n"
            "3. If no decisions or action items exist in the text, reply explicitly as instructed — do NOT guess.\n"
            "4. Output clean results directly without preambles like 'Here are...' or 'Based on the transcript'."
        )

        candidate_models = [GEMINI_MODEL]
        for fallback_m in ["gemini-3.6-flash", "gemini-1.5-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"]:
            if fallback_m not in candidate_models:
                candidate_models.append(fallback_m)

        max_retries = 3
        retry_delay = 2

        safe_prompt = cls._truncate_prompt(prompt, max_chars=30000)

        for target_model in candidate_models:
            # 1. Try official google.genai SDK
            try:
                from google import genai  # type: ignore
                from google.genai import types  # type: ignore


                client = genai.Client(api_key=GEMINI_API_KEY)
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    max_output_tokens=1024,
                )
                response = client.models.generate_content(
                    model=target_model,
                    contents=safe_prompt,
                    config=config,
                )
                if response and response.text:
                    return cls._clean_output(response.text)
            except Exception:
                pass

            # 2. REST API fallback
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={GEMINI_API_KEY}"
            headers = {"Content-Type": "application/json"}
            full_prompt = f"{system_prompt}\n\nTask:\n{safe_prompt}"
            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024}
            }

            for attempt in range(1, max_retries + 1):
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                text = parts[0]["text"]
                                return cls._clean_output(text)
                        return None
                    elif response.status_code in (429, 503):
                        print(f"[ActionModelLoader] Gemini API ({target_model}) HTTP {response.status_code} (attempt {attempt}/{max_retries}) — retrying in {retry_delay}s...")
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                            retry_delay *= 2
                    else:
                        print(f"[ActionModelLoader] Gemini API ({target_model}) HTTP {response.status_code}: {response.text[:200]}")
                        break
                except Exception as e:
                    print(f"[ActionModelLoader] Gemini API Exception: {e}")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        break

        return None

    @classmethod
    def _generate_groq(cls, prompt):
        """Generate response via Groq API (OpenAI-compatible, free tier)."""
        import time

        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        }

        system_prompt = (
            "You are MeetingMind AI, an intelligent executive meeting analysis assistant.\n"
            "REAL SPOKEN MEETING CONTEXT:\n"
            "- Transcripts come from real multi-cultural spoken meetings with non-native English speakers, spoken disfluencies, fillers ('you can say'), and conversational phrasing.\n"
            "- Focus on the UNDERLYING INTENT, SUBSTANTIVE BUSINESS CONTEXT, AND CORE DECISIONS.\n"
            "STRICT RULES:\n"
            "1. Answer ONLY from the facts and context in the transcript. Do NOT invent outside information.\n"
            "2. Interpret conversational spoken expressions into clear, professional business topics, decisions, and action items.\n"
            "3. If no decisions or action items exist in the text, reply explicitly as instructed — do NOT guess.\n"
            "4. Output clean results directly without preambles like 'Here are...' or 'Based on the transcript'."
        )

        candidate_models = [GROQ_MODEL]
        for fallback_m in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"]:
            if fallback_m not in candidate_models:
                candidate_models.append(fallback_m)

        max_retries = 3

        for target_model in candidate_models:
            # Enforce safe character limit to respect Groq TPM limits
            current_prompt = cls._truncate_prompt(prompt, max_chars=10000)

            for attempt in range(1, max_retries + 1):
                payload = {
                    "model": target_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": current_prompt},
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1024,
                }

                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        text = data["choices"][0]["message"]["content"]
                        return cls._clean_output(text)
                    elif response.status_code == 413 or "request too large" in response.text.lower() or "tokens per minute" in response.text.lower():
                        print(f"[ActionModelLoader] Groq API ({target_model}) Request too large / TPM limit hit — truncating prompt to 6,000 chars...")
                        current_prompt = cls._truncate_prompt(prompt, max_chars=6000)
                        if attempt < max_retries:
                            time.sleep(2)
                            continue
                        else:
                            break
                    elif response.status_code in (429, 503):
                        print(f"[ActionModelLoader] Groq API ({target_model}) HTTP {response.status_code} — retrying attempt {attempt}/{max_retries}...")
                        if attempt < max_retries:
                            time.sleep(3)
                    else:
                        print(f"[ActionModelLoader] Groq API ({target_model}) HTTP {response.status_code}: {response.text[:200]}")
                        break
                except Exception as e:
                    print(f"[ActionModelLoader] Groq API Exception ({target_model}): {e}")
                    if attempt < max_retries:
                        time.sleep(2)
                    else:
                        break

        return None

    @classmethod
    def _generate_openai(cls, prompt):
        """Generate response via OpenAI Chat Completions API with retry on transient errors."""
        import time

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }

        system_prompt = (
            "You are MeetingMind AI, an intelligent executive meeting analysis assistant.\n"
            "REAL SPOKEN MEETING CONTEXT:\n"
            "- Transcripts come from real multi-cultural spoken meetings with non-native English speakers, spoken disfluencies, fillers ('you can say'), and conversational phrasing.\n"
            "- Focus on the UNDERLYING INTENT, SUBSTANTIVE BUSINESS CONTEXT, AND CORE DECISIONS.\n"
            "STRICT RULES:\n"
            "1. Answer ONLY from the facts and context in the transcript. Do NOT invent outside information.\n"
            "2. Interpret conversational spoken expressions into clear, professional business topics, decisions, and action items.\n"
            "3. If no decisions or action items exist in the text, reply explicitly as instructed — do NOT guess.\n"
            "4. Output clean results directly without preambles like 'Here are...' or 'Based on the transcript'."
        )

        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        # Retry up to 3 times for transient server errors (429 rate limit, 503 overload)
        max_retries = 3
        retry_delay = 2  # seconds (doubles each attempt: 2 → 4 → 8)

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    return cls._clean_output(text)
                elif response.status_code in (429, 503):
                    # Check if it's a billing/quota error (not retryable)
                    try:
                        err_code = response.json().get("error", {}).get("code", "")
                    except Exception:
                        err_code = ""
                    if err_code == "insufficient_quota":
                        print(f"[ActionModelLoader] OpenAI API Error: Insufficient quota — please add credits at platform.openai.com/settings/billing")
                        return None
                    print(f"[ActionModelLoader] OpenAI API HTTP {response.status_code} (attempt {attempt}/{max_retries}) — retrying in {retry_delay}s...")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        print(f"[ActionModelLoader] OpenAI API Error HTTP {response.status_code}: {response.text}")
                else:
                    # Non-retryable error (401, 404, etc.) — fail immediately
                    print(f"[ActionModelLoader] OpenAI API Error HTTP {response.status_code}: {response.text}")
                    return None
            except Exception as e:
                print(f"[ActionModelLoader] OpenAI API Exception: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return None

        return None

    @classmethod
    def generate(cls, prompt):
        # 1. Try Groq API (Primary Cloud Provider)
        if USE_GROQ_API:
            res = cls._generate_groq(prompt)
            if res is not None:
                return res
            print("[ActionModelLoader] Groq API unavailable. Trying Gemini fallback...")

        # 2. Try Google Gemini API (Secondary Cloud Provider)
        if USE_GEMINI_API:
            res = cls._generate_gemini(prompt)
            if res is not None:
                return res
            print("[ActionModelLoader] Gemini API unavailable. Trying OpenAI fallback...")

        # 3. Try OpenAI API (Paid fallback)
        if USE_OPENAI_API:
            res = cls._generate_openai(prompt)
            if res is not None:
                return res
            print("[ActionModelLoader] OpenAI API unavailable. Falling back to local PyTorch model...")

        # 4. Local PyTorch Model Fallback
        cls.load()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are MeetingMind AI, an intelligent executive meeting analysis assistant.\n"
                    "REAL SPOKEN MEETING CONTEXT:\n"
                    "- Transcripts come from real multi-cultural spoken meetings with non-native English speakers, spoken disfluencies, fillers ('you can say'), and conversational phrasing.\n"
                    "- Focus on the UNDERLYING INTENT, SUBSTANTIVE BUSINESS CONTEXT, AND CORE DECISIONS.\n"
                    "STRICT RULES:\n"
                    "1. Answer ONLY from the facts and context in the transcript. Do NOT invent outside information.\n"
                    "2. Interpret conversational spoken expressions into clear, professional business topics, decisions, and action items.\n"
                    "3. If no decisions or action items exist in the text, reply explicitly as instructed — do NOT guess.\n"
                    "4. Output clean results directly without preambles like 'Here are...' or 'Based on the transcript'."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        try:
            try:
                text = cls._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                text = f"{messages[0]['content']}\n\nTask: {prompt}\n\nResponse:"

            inputs = cls._tokenizer(
                text,
                return_tensors="pt",
            ).to(cls._model.device)

            pad_id = cls._tokenizer.eos_token_id if cls._tokenizer.eos_token_id is not None else cls._tokenizer.pad_token_id

            with torch.no_grad():
                outputs = cls._model.generate(
                    **inputs,
                    max_new_tokens=384,
                    do_sample=False,
                    repetition_penalty=1.2,
                    pad_token_id=pad_id,
                )

                generated = outputs[0][inputs["input_ids"].shape[1]:]
                result = cls._tokenizer.decode(generated, skip_special_tokens=True)

                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                return cls._clean_output(result)

        except (RuntimeError, Exception) as err:
            err_msg = str(err)
            print(f"[ActionModelLoader] CUDA/Generation Exception: {err_msg}")

            if "cuda" in err_msg.lower() or "timed out" in err_msg.lower() or "out of memory" in err_msg.lower():
                print("[ActionModelLoader] Windows CUDA TDR timeout detected — automatically switching local model to CPU...")
                try:
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    if cls._model is not None:
                        cls._model = cls._model.to("cpu")
                        inputs_cpu = cls._tokenizer(text, return_tensors="pt").to("cpu")
                        pad_id = cls._tokenizer.eos_token_id if cls._tokenizer.eos_token_id is not None else cls._tokenizer.pad_token_id
                        with torch.no_grad():
                            outputs = cls._model.generate(
                                **inputs_cpu,
                                max_new_tokens=256,
                                do_sample=False,
                                repetition_penalty=1.2,
                                pad_token_id=pad_id,
                            )
                            generated = outputs[0][inputs_cpu["input_ids"].shape[1]:]
                            return cls._clean_output(cls._tokenizer.decode(generated, skip_special_tokens=True))
                except Exception as cpu_err:
                    print(f"[ActionModelLoader] CPU fallback exception: {cpu_err}")

            return "Information extracted based on meeting transcript context."


if __name__ == "__main__":
    if USE_OPENAI_API:
        print("Testing OpenAI API Generation...")
        out = ActionModelLoader.generate("Say hello and confirm MeetingMind AI OpenAI integration in 1 short sentence.")
        print(f"Result: {out}")
    else:
        ActionModelLoader.load()
        print("Local Model Ready!")
