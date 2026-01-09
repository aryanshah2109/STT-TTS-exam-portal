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
    

class Question(BaseModel):
    text: Annotated[
        str, 
        Field(
            title="Question text", 
            description="A single question of a topic"
        )
    ]

class TopicQuestions(BaseModel):
    topic: Annotated[
        str,
        Field(
            title="Topic name",
            description="Topic name from the provided topic_list"
        )
    ]
    questions: Annotated[
        List[Question],
        Field(
            title="Questions for the topic",
            min_items=1
        )
    ]


class MultiQuestionGenerationResponse(BaseModel):
    
    topics: Annotated[
        List[TopicQuestions],
        Field(
            title="Output response",
            min_items=1
        )
    ]
    
    

