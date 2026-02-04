from langchain_core.prompts import PromptTemplate
from ai_ml.ModelCreator import GeminiModelCreation

from pydantic import BaseModel
from typing import List
import re
import json


class EvalSchema(BaseModel):
    score: int
    strengths: List[str]
    weakness: List[str]
    justification: str
    suggested_improvement: str


class EvaluationEngine:
    def __init__(self, model_name: str, global_model=None):
        self.model_name = model_name
        self.model = global_model

    def get_model(self):
        if self.model is None:
            self.model = GeminiModelCreation.gemini_model_creator()
        return self.model

    def sanitize_json(self, text: str) -> str:
        text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found")
        text = match.group(0)
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        return text

    def create_evaluation_chain(self):
        template = """
You are a very strict exam evaluation engine.

Rules:
- Return ONLY valid JSON
- If student says "I don't know", score MUST be 0

Rubric:
{rubric}

Question:
{question_text}

Student Answer:
{student_answer}

Maximum Marks: {max_marks}

Return format:
{{
  "score": 0,
  "strengths": [],
  "weakness": [],
  "justification": "",
  "suggested_improvement": ""
}}
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=[
                "rubric",
                "question_text",
                "student_answer",
                "max_marks"
            ],
        )

        return prompt | self.get_model()

    def model_evaluator(self, input_features: dict):
        chain = self.create_evaluation_chain()
        raw = chain.invoke(input_features)

        if isinstance(raw, dict) and "text" in raw:
            output = raw["text"]
        elif hasattr(raw, "generations"):
            output = raw.generations[0][0].text
        else:
            output = str(raw)

        cleaned = self.sanitize_json(output)
        data = json.loads(cleaned)

        EvalSchema(**data)
        return data
