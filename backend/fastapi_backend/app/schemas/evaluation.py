from typing import List, Annotated
from pydantic import BaseModel, Field, StringConstraints, field_validator


class EvaluateAnswer(BaseModel):
    question_id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1)
    ]

    question_text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=5, max_length=3000)
    ]

    student_answer: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)
    ]

    rubric: List[str] = Field(..., min_length=1)

    max_marks: Annotated[float, Field(ge=1, le=100)] = 10

    @field_validator("rubric", mode="before")
    @classmethod
    def validate_rubric(cls, v):
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            cleaned = [str(item).strip() for item in v if str(item).strip()]
            if not cleaned:
                raise ValueError("Rubric must contain at least one non-empty item.")
            return cleaned
        raise TypeError("Rubric must be a string or list of strings.")


class EvaluateAnswerResponse(BaseModel):
    question_id: str
    score: int = Field(ge=0, le=100)
    strengths: List[str]
    weakness: List[str]
    justification: str
    suggested_improvement: str

    @field_validator("strengths", "weakness", mode="before")
    @classmethod
    def validate_lists(cls, v):
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item) for item in v]
        return [str(v)]
