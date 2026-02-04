from langchain_core.prompts import PromptTemplate
from ai_ml.ModelCreator import GeminiModelCreation

from pydantic import BaseModel, Field
from typing import List
import re
import json


class RubricsResponse(BaseModel):
    question_text: str = Field(min_length=1)
    rubrics: List[str] = Field(min_length=1)


class RubricsEngine:
    def __init__(self, model_name: str, global_model=None):
        self.model_name = model_name
        self.model = global_model

    def get_model(self):
        if self.model is None:
            self.model = GeminiModelCreation.gemini_model_creator()
        return self.model

    def extract_and_fix_json(self, text: str) -> dict:
        text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found")
        json_like = match.group(0)
        json_like = json_like.replace("'", '"')
        json_like = re.sub(
            r'(?<!")(\b[a-zA-Z_][a-zA-Z0-9_]*\b)\s*:',
            r'"\1":',
            json_like
        )
        json_like = re.sub(r",\s*}", "}", json_like)
        json_like = re.sub(r",\s*]", "]", json_like)
        return json.loads(json_like)

    def create_rubrics_chain(self):
        template = """
You are an exam evaluator.

TASK:
Generate marking rubrics for the given question.

RULES:
- Return ONLY a JSON object
- Do NOT include explanations or extra text
- Each rubric must be a clear evaluative point

Return JSON ONLY in this format:
{{
  "question_text": "{question_text}",
  "rubrics": []
}}

Question:
{question_text}

Total Marks: {max_marks}
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["question_text", "max_marks"]
        )

        return prompt | self.get_model()

    def create_rubrics(self, input_features: dict):
        if "question_text" not in input_features:
            raise KeyError("question_text is required")
        if "max_marks" not in input_features:
            raise KeyError("max_marks is required")

        chain = self.create_rubrics_chain()
        raw = chain.invoke(input_features)

        if isinstance(raw, dict) and "text" in raw:
            output = raw["text"]
        elif hasattr(raw, "generations"):
            output = raw.generations[0][0].text
        else:
            output = str(raw)

        data = self.extract_and_fix_json(output)
        RubricsResponse(**data)
        return data
