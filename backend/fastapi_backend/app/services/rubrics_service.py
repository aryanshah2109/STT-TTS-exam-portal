from app.schemas.rubrics import RubricsRequest
from ai_ml.Rubrics import RubricsEngine
from app.core import models
from app.config import settings
from fastapi import HTTPException
import traceback

model_name = settings.HF_EVAL_MODEL_NAME


class RubricsService:
    def generate(self, payload: RubricsRequest):
        data = payload.model_dump()
        
        # Add debug logging
        print(f"DEBUG - Generating rubrics for: {data.get('question_text', '')[:50]}...")

        try:
            # Initialize rubrics engine
            engine = RubricsEngine(
                model_name=model_name,
                global_model=models.ai_model
            )
            
            # Generate rubrics
            result = engine.create_rubrics(data)

            # Validate result structure
            if not isinstance(result, dict):
                raise ValueError("Rubrics result is not a dictionary")

            required_keys = ["question_text", "rubrics"]
            missing_keys = [k for k in required_keys if k not in result]
            if missing_keys:
                raise ValueError(f"Missing keys in result: {missing_keys}")
            
            # Ensure rubrics is a list
            rubrics = result.get("rubrics", [])
            if not isinstance(rubrics, list):
                raise ValueError("Rubrics should be a list")
            
            # Filter out empty rubrics
            result["rubrics"] = [r for r in rubrics if r and isinstance(r, str) and r.strip()]

        except ValueError as e:
            print(f"Rubrics generation validation error: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=422,
                detail=f"Validation error: {str(e)}"
            )
        except Exception as e:
            print(f"Rubrics generation error: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Rubrics generation failed: {str(e)[:200]}"
            )

        result["question_id"] = payload.question_id
        return result


generate_rubrics_service = RubricsService()