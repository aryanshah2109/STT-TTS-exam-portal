from langchain_core.prompts import PromptTemplate
import re
import json

from ai_ml.ModelCreator import HFModelCreation
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
            self.model = HFModelCreation.hf_model_creator(self.model_name)
        return self.model

    def chain_creator(self):
        try:
            template = """
You are an academic viva and interview exam question setter.

TASK:
Generate EXACTLY {num_questions} questions for the given TOPIC.

STRICT RULES (MANDATORY):
- Generate ONLY theory-based, verbally answerable questions.
- DO NOT ask for code, programs, functions, implementations, or pseudocode.
- DO NOT include "write", "implement", "design", "analyze", "compare", "evaluate".
- Questions must be suitable for written exams and viva exams.
- Keep words of the questions simple to understand. 
- STRICTLY do not ask questions that require a written solution like write the code based questions.

DIFFICULTY RULES (VERY IMPORTANT):

If difficulty is EASY:
- Ask ONLY definition-based or basic explanation questions
- Questions should be answerable in 2-4 sentences


If difficulty is MEDIUM:
- Ask explanation and comparison questions
- Allow reasoning and examples


If difficulty is HARD:
- Ask critical discussion, limitations, real-world relevance
- Higher-order thinking questions

STRICTLY NO code or implementation
Stay strictly within the given TOPIC.
Do NOT include unrelated concepts.

Return ONLY valid JSON in this exact format:
{{
  "topic": "{topic}",
  "questions": [
    "question 1",
    "question 2"
  ]
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
        matches = re.findall(r"\{[\s\S]*?\}", text)
        if not matches:
            raise ValueError("No JSON object found in model output")
        return matches[-1]

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
            elif hasattr(raw, "generations"):
                output = raw.generations[0][0].text
            else:
                output = str(raw)

            cleaned = self.sanitize_json(output)
            data = json.loads(cleaned)

            questions = data.get("questions", [])
            if not isinstance(questions, list):
                questions = []

            return questions[:num_questions]

        except Exception as e:
            raise QuestionsGenerationException(f"Generation failed: {str(e)}")

