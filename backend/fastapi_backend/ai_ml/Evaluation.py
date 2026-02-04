from langchain_core.prompts import PromptTemplate
from ai_ml.ModelCreator import GeminiModelCreation
from pydantic import BaseModel, field_validator
from typing import List
import re
import json
from json.decoder import JSONDecodeError


class EvalSchema(BaseModel):
    score: int
    strengths: List[str]
    weakness: List[str]
    justification: str
    suggested_improvement: str
    
    @field_validator("score", mode="before")
    @classmethod
    def convert_score_to_int(cls, v):
        """Convert float scores to int by rounding"""
        if isinstance(v, float):
            return int(round(v))
        return int(v)


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
        Extract JSON from model output with robust error handling.
        """
        # Clean the text
        text = text.strip()
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Try to find JSON object
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        
        if not json_match:
            # Try to find JSON array as fallback
            json_match = re.search(r'(\[.*\])', text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object or array found in the response")
        
        json_str = json_match.group(1).strip()
        
        # First, try to parse it as-is
        try:
            return json.loads(json_str)
        except JSONDecodeError:
            pass  # Continue with fixing attempts
        
        # Fix common JSON issues
        fixed_json = json_str
        
        # Replace single quotes with double quotes 
        # Only replace single quotes at the boundaries of strings
        fixed_json = re.sub(
            r':\s*\'(.*?)\'\s*([,}])',
            r': "\1"\2',
            fixed_json
        )
        fixed_json = re.sub(
            r'\{\s*\'(.*?)\'\s*:',
            r'{ "\1":',
            fixed_json
        )
        
        # Ensure property names are quoted
        fixed_json = re.sub(
            r'(?<!["\w])(\b[a-zA-Z_][a-zA-Z0-9_]*\b)\s*:',
            r'"\1":',
            fixed_json
        )
        
        # Remove trailing commas
        fixed_json = re.sub(r',\s*([}\]])', r'\1', fixed_json)
        
        # Try parsing again
        try:
            return json.loads(fixed_json)
        except JSONDecodeError as e:
            # For debugging
            print(f"Original text: {text[:500]}")
            print(f"JSON attempt: {json_str[:500]}")
            print(f"Fixed JSON: {fixed_json[:500]}")
            raise ValueError(f"Invalid JSON format: {str(e)}")

    def create_evaluation_chain(self):
        template = """
You are a strict exam evaluation engine.

Rules:
- Return ONLY a valid JSON object
- Use double quotes for all property names and string values
- Do NOT include any explanations or text outside the JSON
- If student says "I don't know", score MUST be 0

Rubric:
{rubric}

Question:
{question_text}

Student Answer:
{student_answer}

Maximum Marks: {max_marks}

Return JSON ONLY in this EXACT format:
{{
  "score": 0,
  "strengths": [],
  "weakness": [],
  "justification": "",
  "suggested_improvement": ""
}}

IMPORTANT: Your response must be valid JSON that can be parsed by json.loads().
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

        # Properly extract content from LangChain response
        if hasattr(raw, 'content'):
            output = raw.content
        elif isinstance(raw, dict) and "text" in raw:
            output = raw["text"]
        elif hasattr(raw, "generations"):
            output = raw.generations[0][0].text
        else:
            output = str(raw)
        
        # Log the raw output
        print(f"DEBUG - Raw model output: {output[:500]}...")
        
        try:
            data = self.extract_and_fix_json(output)
            # Validate with Pydantic schema
            validated_data = EvalSchema(**data)
            return validated_data.model_dump()
        except Exception as e:
            print(f"DEBUG - Error during JSON parsing: {e}")
            print(f"DEBUG - Full output: {output}")
            raise