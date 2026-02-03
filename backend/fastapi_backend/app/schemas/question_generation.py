from typing import List, Dict, Literal
from pydantic import BaseModel, Field


class QuestionGenerationRequest(BaseModel):
    topics: List[str]
    num_questions: int = Field(ge=1, le=100)
    difficulty: Literal["easy", "medium", "hard"]


class QuestionGenerationResponse(BaseModel):
    topics: Dict[str, Dict[str, str]]

