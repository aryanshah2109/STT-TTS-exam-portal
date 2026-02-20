from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ai_ml.ModelCreator import GeminiModelCreation
from ai_ml.Speech2Text import SpeechModelGenerator
from ai_ml.MCQEvaluation import MCQEvaluationEngine
from app.core import models
from app.config import settings

from app.routers import (
    stt,
    evaluation,
    tts,
    rubrics,
    mcq_evaluation,
    QuestionGenerator
)

from dotenv import load_dotenv
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):

    # preload whisper
    models.whisper_model = SpeechModelGenerator.whisper_model_generator()

    # preload Gemini ONCE
    models.ai_model = GeminiModelCreation.gemini_model_creator()

    # preload sentence transformer
    models.st_model = MCQEvaluationEngine(settings.MCQ_EVAL_MODEL_NAME)

    yield

app = FastAPI(title="Examecho AI Service", lifespan=lifespan)

# Add CORS middleware for HF Spaces compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(QuestionGenerator.router)
app.include_router(rubrics.router)
app.include_router(tts.router)
app.include_router(stt.router)
app.include_router(evaluation.router)
app.include_router(mcq_evaluation.router)
