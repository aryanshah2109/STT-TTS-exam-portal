from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pydantic import BaseModel, Field
from typing import List, Annotated

from ai_ml.ModelCreator import HFModelCreation

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


class OutputResponse(BaseModel):
    topics: Annotated[
        List[TopicQuestions],
        Field(
            title="Output response",
            min_items=1
        )
    ]

class MultiQuestionsGenerator:

    def __init__(self, model_name: str, global_model=None):
        self.model_name = model_name
        self.model = global_model

    def get_model(self):
        if self.model is None:
            self.model = HFModelCreation.hf_model_creator(self.model_name)
        return self.model

    def chain_creator(self):
        parser = JsonOutputParser(pydantic_object=OutputResponse)

        template = """
You are an exam question generator.

Your task is to generate exactly {num_questions} questions
using ONLY the subjects and topics provided below.

SUBJECT-TOPIC MAPPING RULES:
- Each topic belongs to exactly one subject.
- Infer the most reasonable subject for each topic based on academic context.
- DO NOT invent new topics.
- DO NOT use topics outside the given topic list.
- If a subject has no matching topic, generate ZERO questions for that subject.

GENERATION RULES:
- Every question MUST clearly belong to one subject AND one topic.
- Questions must be well-structured, clear, and exam-appropriate.
- The total number of generated questions MUST be exactly {num_questions}.
- Do NOT exceed or fall short of the requested number.

DISTRIBUTION RULES:
- Distribute questions as evenly as possible across topics.
- If exact division is not possible, assign extra questions
  starting from the first topic in the topic list.

OUTPUT FORMAT RULES:
- Return ONLY valid JSON.
- Do NOT include explanations, markdown, or extra text.
- Follow the JSON schema strictly.

Topic List:
{topic_list}

Subject List:
{subject_list}

{format_instructions}
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["num_questions", "topic_list", "subject_list"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )

        chain = prompt | self.get_model() | parser
        return chain

    def create_questions(self, input_request: dict) -> OutputResponse:
        chain = self.chain_creator()

        result: OutputResponse = chain.invoke(input_request)

        self.validate_count(result, input_request["num_questions"])

        return result

    def validate_count(self, result: OutputResponse, expected: int):
        total = sum(len(t.questions) for t in result.topics)
        if total != expected:
            raise ValueError(
                f"Expected {expected} questions, got {total}"
            )
