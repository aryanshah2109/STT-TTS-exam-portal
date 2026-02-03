from fastapi import APIRouter, HTTPException
from app.schemas.question_generation import (
    QuestionGenerationRequest,
    QuestionGenerationResponse
)
from app.services.question_generation_service import generation_service

router = APIRouter(
    prefix="/questions_generate",
    tags=["Questions Generation"]
)


@router.post("/generate", response_model=QuestionGenerationResponse)
async def generate_route(payload: QuestionGenerationRequest):

    result = generation_service.generate(payload)

    if not result["topics"]:
        raise HTTPException(
            status_code=400,
            detail="No questions could be generated"
        )

    return result

