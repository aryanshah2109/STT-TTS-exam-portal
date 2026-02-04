from app.schemas.rubrics import RubricsRequest
from ai_ml.Rubrics import RubricsEngine
from app.core import models
from app.config import settings
from fastapi import HTTPException

model_name = settings.HF_EVAL_MODEL_NAME


class RubricsService:
    def generate(self, payload: RubricsRequest):
        data = payload.model_dump()

        try:
            result = RubricsEngine(
                model_name=model_name,
                global_model=models.ai_model
            ).create_rubrics(data)

            if not isinstance(result, dict):
                raise ValueError("Invalid rubric output")

            if "question_text" not in result or "rubrics" not in result:
                raise ValueError("Missing required keys")

        except Exception as e:
            print("Rubrics generation error:", e)
            raise HTTPException(
                status_code=500,
                detail="Rubrics generation failed due to model error"
            )

        result["question_id"] = payload.question_id
        return result


generate_rubrics_service = RubricsService()
