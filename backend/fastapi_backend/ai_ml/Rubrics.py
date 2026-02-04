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

    def sanitize_json(self, text: str) -> str:
        text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("No JSON object found in model output")
        text = match.group(0)
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        return text

    def create_rubrics_chain(self):
        template = """
You are an exam evaluator.
Generate marking rubrics for the given question.

Return ONLY valid JSON in the following format:
{{
  "question_text": "{question_text}",
  "rubrics": []
}}

Question: {question_text}
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

        cleaned = self.sanitize_json(output)
        data = json.loads(cleaned)

        RubricsResponse(**data)
        return data
