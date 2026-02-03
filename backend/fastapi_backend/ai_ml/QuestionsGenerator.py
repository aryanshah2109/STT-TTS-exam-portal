from langchain_core.prompts import PromptTemplate
import re
import json

from ai_ml.ModelCreator import GeminiModelCreation
from ai_ml.AIExceptions import (
    ChainCreationException,
    QuestionsGenerationException
)


class QuestionsGenerator:
    def __init__(self, model_name: str, global_model=None):
        self.model_name = model_name
        self.model = global_model

    def get_model(self):
        if self.model is None:
            self.model = GeminiModelCreation.gemini_model_creator()
        return self.model

    def chain_creator(self):
        try:
            template = """
You are an academic exam question setter.

TASK:
Generate EXACTLY {num_questions} theory-based exam questions based strictly on the given TOPIC.

GENERAL RULES:
- Questions must be verbally answerable
- NO code, NO programs, NO algorithms
- Language must be clear and suitable for written exams
- Stay strictly within the TOPIC
- Each question must be a complete sentence

DIFFICULTY:
EASY: definitions, meanings, purposes
MEDIUM: explanations, reasoning, simple examples
HARD: critical thinking, limitations, applications, justification

Before producing the final answer, internally decide all {num_questions} questions.
Do NOT output this reasoning.

OUTPUT RULES (MANDATORY):
- Return ONLY valid JSON
- Do NOT include markdown, comments, or explanations
- "questions" MUST be a list of strings
- The list MUST contain EXACTLY {num_questions} items
- The list MUST NOT be empty

JSON SCHEMA (DO NOT DEVIATE):

{{
  "topic": "{topic}",
  "questions": ["<string>", "<string>"]
}}

TOPIC: {topic}
DIFFICULTY: {difficulty}
"""

            prompt = PromptTemplate(
                template=template,
                input_variables=["num_questions", "topic", "difficulty"]
            )

            return prompt | self.get_model()

        except Exception as e:
            raise ChainCreationException(f"Could not create chain: {str(e)}")

    def sanitize_json(self, text: str) -> str:
        text = text.replace("```json", "").replace("```", "").strip()

        decoder = json.JSONDecoder()

        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, _ = decoder.raw_decode(text[i:])
                    return json.dumps(obj)
                except json.JSONDecodeError:
                    continue

        raise ValueError("No valid JSON object found in model output")

    def create_questions(self, topic: str, num_questions: int, difficulty: str):
        try:
            chain = self.chain_creator()

            raw = chain.invoke({
                "topic": topic,
                "num_questions": num_questions,
                "difficulty": difficulty
            })

            if isinstance(raw, dict) and "text" in raw:
                output = raw["text"]
            elif hasattr(raw, "content"):
                output = raw.content
            elif hasattr(raw, "generations"):
                output = raw.generations[0][0].text
            else:
                output = str(raw)

            cleaned = self.sanitize_json(output)
            data = json.loads(cleaned)

            questions = data.get("questions", [])

            if not isinstance(questions, list):
                raise QuestionsGenerationException("Invalid questions format returned")

            normalized = []
            for q in questions:
                if isinstance(q, str):
                    q = re.sub(r"^\s*\d+[\.\)]\s*", "", q).strip()
                    if q:
                        normalized.append(q)

            if not normalized:
                raise QuestionsGenerationException(
                    "Model returned empty questions list. Prompt compliance failed."
                )

            if len(normalized) < num_questions:
                raise QuestionsGenerationException(
                    f"Expected {num_questions} questions, got {len(normalized)}"
                )

            return normalized[:num_questions]

        except Exception as e:
            raise QuestionsGenerationException(f"Generation failed: {str(e)}")
