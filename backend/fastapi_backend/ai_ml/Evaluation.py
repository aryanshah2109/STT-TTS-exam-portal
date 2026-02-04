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

    def extract_and_fix_json(self, text: str) -> dict:
        """
        Robust JSON repair for LLM output (Gemini-safe)
        """
        # remove markdown fences
        text = text.replace("```json", "").replace("```", "").strip()

        # extract first {...} block
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in model output")

        json_like = match.group(0)

        # normalize quotes (Gemini often uses single quotes)
        json_like = json_like.replace("'", '"')

        # remove trailing commas
        json_like = re.sub(r",\s*}", "}", json_like)
        json_like = re.sub(r",\s*]", "]", json_like)

        return json.loads(json_like)

    def create_evaluation_chain(self):
        template = """
You are a very strict exam evaluation engine.

Rules:
- Return ONLY a JSON object
- Do NOT include explanations or text outside JSON
- If student says "I don't know", score MUST be 0

Rubric:
{rubric}

Question:
{question_text}

Student Answer:
{student_answer}

Maximum Marks: {max_marks}

Return JSON ONLY in this format:
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

        data = self.extract_and_fix_json(output)

        # strict validation
        EvalSchema(**data)
        return data
