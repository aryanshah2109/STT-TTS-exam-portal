from typing import List, Dict, Annotated, Optional
from pydantic import BaseModel, Field, StringConstraints

class MultiQuestionGenerationRequest(BaseModel):

    topic_list: Annotated[
        List[
            Annotated[
                str,
                StringConstraints(strip_whitespace=True, min_length=1, to_lower=True)
            ]
        ],
        Field(min_length=1)
    ]
    
    subject_list: Annotated[
        List[
            Annotated[
                str,
                StringConstraints(strip_whitespace=True, min_length=1, to_lower=True)
            ]
        ],
        Field(min_length=1)
    ]
    
    num_questions: Annotated[int,
                             Field(ge=1, le=100)]
    

class MultiQuestionGenerationResponse(BaseModel):
    
    questions: Annotated[List[str],
                         Field(min_length=1)]
    
    

