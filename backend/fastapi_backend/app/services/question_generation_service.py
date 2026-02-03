from app.schemas.question_generation import QuestionGenerationRequest
from app.core import models
from ai_ml.QuestionsGenerator import QuestionsGenerator
from app.config import settings

model_name = settings.HF_EVAL_MODEL_NAME


class QuestionGenerationService:

    def generate(self, payload: QuestionGenerationRequest):

        final_output = {}

        generator = QuestionsGenerator(
            model_name=model_name,
            global_model=models.ai_model
        )

        for topic in payload.topics:
            questions = generator.create_questions(
                topic=topic,
                num_questions=payload.num_questions,
                difficulty=payload.difficulty
            )
            

            final_output[topic] = {
                f"question {i + 1}": q
                for i, q in enumerate(questions)
            }

        return {
            "topics": final_output
        }


generation_service = QuestionGenerationService()
