
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings

import whisper

from sentence_transformers import SentenceTransformer, util

try:
    import torch
except Exception:
    torch = None



class SpeechModelGenerator:
    """
    Loads Whisper / HF Whisper models ONCE per process.
    Used when user does NOT preload models via FastAPI startup.
    """
    _whisper_model = None
    _hf_model = None

    @staticmethod
    def _get_default_device():
        """Return -1 (CPU) or 0 (GPU)."""
        try:
            if torch is not None and torch.cuda.is_available():
                return 0
        except Exception:
            pass
        return -1

    @classmethod
    def whisper_model_generator(cls):
        """Lazy-load Whisper Base model."""

        if cls._whisper_model is None:
            cls._whisper_model = whisper.load_model("base")
        return cls._whisper_model

    


class GeminiModelCreation:
    """
    Creates a Gemini LLM wrapped as a LangChain Runnable.
    Loaded ONCE and reused across the app.
    """

    @staticmethod
    def gemini_model_creator():
        try:
            return ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL_NAME,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.0,        
                max_output_tokens=2048
            )
        except Exception as e:
            raise RuntimeError(f"Gemini model loading failed: {e}")
