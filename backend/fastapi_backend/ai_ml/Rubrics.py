from langchain_core.prompts import PromptTemplate
from ai_ml.ModelCreator import GeminiModelCreation
from pydantic import BaseModel, Field
from typing import List
import re
import json
from json.decoder import JSONDecodeError


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
        
        #  common JSON issues
        fixed_json = json_str
        
        #  1: Replace single quotes with double quotes (careful approach)
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
        
        #  2: Ensure property names are quoted
        fixed_json = re.sub(
            r'(?<!["\w])(\b[a-zA-Z_][a-zA-Z0-9_]*\b)\s*:',
            r'"\1":',
            fixed_json
        )
        
        #  3: Remove trailing commas
        fixed_json = re.sub(r',\s*([}\]])', r'\1', fixed_json)
        
        #  4: Handle escaped quotes
        fixed_json = fixed_json.replace('\\"', '\\\\"')
        
        # Try parsing again
        try:
            return json.loads(fixed_json)
        except JSONDecodeError as e:
            # For debugging
            print(f"Original text: {text[:500]}")
            print(f"JSON attempt: {json_str[:500]}")
            print(f"Fixed JSON: {fixed_json[:500]}")
            raise ValueError(f"Invalid JSON format: {str(e)}")

    def create_rubrics_chain(self):
        template = """
You are an exam evaluator.

TASK:
Generate marking rubrics for the given question.

RULES:
- Return ONLY a valid JSON object
- Use double quotes for all property names and string values
- Do NOT include any explanations or text outside the JSON
- Each rubric must be a clear evaluative point
- Generate appropriate number of rubrics based on marks (approximately one rubric per 2-3 marks)

Return JSON ONLY in this EXACT format:
{{
  "question_text": "{question_text}",
  "rubrics": ["rubric 1", "rubric 2", "rubric 3"]
}}

IMPORTANT: Your response must be valid JSON that can be parsed by json.loads().

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
        
        # DEBUG: Log the raw output
        print(f"DEBUG - Raw model output: {output[:500]}...")
        
        try:
            data = self.extract_and_fix_json(output)
            # Validate with Pydantic schema
            validated_data = RubricsResponse(**data)
            return validated_data.dict()
        except Exception as e:
            print(f"DEBUG - Error during JSON parsing: {e}")
            print(f"DEBUG - Full output: {output}")
            raise