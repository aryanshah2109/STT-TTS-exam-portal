from typing import List, Annotated
from pydantic import BaseModel, Field, StringConstraints, field_validator


class RubricsRequest(BaseModel):
    question_id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1)
    ]

    question_text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=3000)
    ]

    max_marks: Annotated[int, Field(ge=1, le=100)]


class RubricsResponse(BaseModel):
    question_id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1)
    ]

    question_text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1)
    ]

    rubrics: List[str] = Field(..., min_length=1)

    @field_validator("rubrics", mode="before")
    @classmethod
    def validate_rubrics(cls, v):
        if isinstance(v, str):
            import ast
            try:
                v = ast.literal_eval(v)
            except Exception:
                v = [v]

        if isinstance(v, list):
            cleaned = [str(item).strip() for item in v if str(item).strip()]
            if not cleaned:
                raise ValueError("Rubrics must contain at least one non-empty item.")
            return cleaned

        raise TypeError("Rubrics must be a string or list of strings.")
