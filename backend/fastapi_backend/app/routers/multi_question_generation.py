from fastapi import APIRouter, HTTPException
from app.schemas.multi_question_generation import MultiQuestionGenerationRequest, MultiQuestionGenerationResponse
from app.services.multi_question_generation_service import generation_service

router = APIRouter(
    prefix="/questions_generate_multi",
    tags = ["questions_generation_multi"]
)

@router.post("/generate", response_model= MultiQuestionGenerationResponse)
async def generate_route(payload: MultiQuestionGenerationRequest):
    
    try:
        questions = generation_service.generate(payload)

        if not questions:
            raise HTTPException(
                status_code=500,
                detail="Model failed to generate questions"
            )

        return questions

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate questions due to error. Details: {e}"
        )


