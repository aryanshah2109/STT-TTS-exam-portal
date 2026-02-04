from ai_ml.Evaluation import EvaluationEngine
from app.schemas.evaluation import EvaluateAnswer
from app.core import models
from app.config import settings
import traceback

model_name = settings.HF_EVAL_MODEL_NAME


class EvaluationService:
    def evaluate(self, payload: EvaluateAnswer):
        data = payload.model_dump()
        
        # Ensure rubric is properly formatted
        rubric = data.get("rubric", [])
        if isinstance(rubric, str):
            rubric = [rubric]
        elif not isinstance(rubric, list):
            rubric = []
        
        # Format rubric as a string for the prompt
        rubric_text = "\n".join([f"- {r}" for r in rubric])
        data["rubric"] = rubric_text

        try:
            # Initialize evaluation engine
            engine = EvaluationEngine(
                model_name=model_name,
                global_model=models.ai_model
            )
            
            # Perform evaluation
            result = engine.model_evaluator(data)

            # Validate result structure
            if not isinstance(result, dict):
                raise ValueError("Evaluation result is not a dictionary")

            required_keys = [
                "score",
                "strengths",
                "weakness",
                "justification",
                "suggested_improvement"
            ]

            missing_keys = [k for k in required_keys if k not in result]
            if missing_keys:
                raise ValueError(f"Missing keys in result: {missing_keys}")
            
            # Ensure score is within bounds
            max_marks = data.get("max_marks", 10)
            result["score"] = max(0, min(int(result.get("score", 0)), int(max_marks)))
            
            # Ensure lists are not None
            result["strengths"] = result.get("strengths", []) or []
            result["weakness"] = result.get("weakness", []) or []
            
            # Ensure strings are not None
            result["justification"] = result.get("justification", "") or ""
            result["suggested_improvement"] = result.get("suggested_improvement", "") or ""

        except Exception as e:
            print(f"Evaluation error: {e}")
            traceback.print_exc()
            
            return {
                "question_id": payload.question_id,
                "score": 0,
                "strengths": [],
                "weakness": ["Evaluation failed due to technical error"],
                "justification": f"System error: {str(e)[:200]}",
                "suggested_improvement": "Please try again or contact support"
            }

        result["question_id"] = payload.question_id
        return result


evaluator_service = EvaluationService()