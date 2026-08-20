"""
Summarization Model Loader

Loads the Hugging Face summarization model only once.
"""

from transformers import pipeline

from ai_engine.config import SUMMARIZER_MODEL, HF_DEVICE, DEVICE


class SummarizerLoader:
    """Loads the summarization model dynamically based on config."""

    _model = None
    _model_name = None

    @classmethod
    def get_model(cls):

        if cls._model is None or cls._model_name != SUMMARIZER_MODEL:

            print(f"Loading Summarization Model ({SUMMARIZER_MODEL}) on {DEVICE.upper()}...")

            cls._model = pipeline(
                "summarization",
                model=SUMMARIZER_MODEL,
                device=HF_DEVICE,
            )
            cls._model_name = SUMMARIZER_MODEL

            print(f"Summarization Model ({SUMMARIZER_MODEL}) Loaded Successfully on {DEVICE.upper()}.")

        return cls._model


if __name__ == "__main__":

    model = SummarizerLoader.get_model()

    print("\nModel Ready!")

    