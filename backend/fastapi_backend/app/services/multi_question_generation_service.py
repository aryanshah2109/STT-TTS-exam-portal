from fastapi import HTTPException

from app.schemas.multi_question_generation import MultiQuestionGenerationRequest
from ai_ml.MultiQuestionsGenerator import MultiQuestionsGenerator
from app.core import models
from app.config import settings


model_name = settings.HF_EVAL_MODEL_NAME


class MultiQuestionGenerationService:

    def generate(self, payload: MultiQuestionGenerationRequest):

        data = payload.model_dump()

        try:
            result = MultiQuestionsGenerator(
                model_name=model_name,
                global_model=models.ai_model
            ).create_questions(data)

            if not result:
                raise ValueError("Empty model output")

            return result

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Model generation failed: {e}"
            )


generation_service = MultiQuestionGenerationService()
