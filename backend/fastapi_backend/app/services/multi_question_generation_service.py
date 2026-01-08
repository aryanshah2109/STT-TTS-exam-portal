from app.schemas.multi_question_generation import MultiQuestionGenerationRequest
from app.core import models
from ai_ml.MultiQuestionsGenerator import MultiQuestionsGenerator
from app.config import settings

model_name = settings.HF_EVAL_MODEL_NAME

class MultiQuestionGenerationService:

    def generate(self, payload: MultiQuestionGenerationRequest):

        data = payload.model_dump()

        try:

            # Use models.ai_model loaded during lifespan

            result = MultiQuestionsGenerator(model_name=model_name, global_model=models.ai_model).create_questions(data)

            required_keys = ["questions"]

            if (
                not result 
                or not isinstance(result, dict)
                or any(k not in result for k in required_keys)
            ):
                raise ValueError("Model returned invalid output.")
            
        except Exception as e:
            
            print("Generation error: ", e)

            return {
                "questions": ["No questions could be generated due to model error"]
            }

        return result

generation_service = MultiQuestionGenerationService()