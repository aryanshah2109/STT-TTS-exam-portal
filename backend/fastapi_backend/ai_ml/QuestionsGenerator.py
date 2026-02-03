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
You are generating academic exam questions.

TASK:
Generate EXACTLY {num_questions} questions strictly based on the given TOPIC.

GENERAL RULES (NON-NEGOTIABLE):
- Questions must be theory-based and verbally answerable
- NO code, NO programs, NO algorithms, NO implementations
- NO problem-solving or step-by-step reasoning questions
- Language must be clear, simple, and exam-oriented
- Stay strictly within the given TOPIC

DIFFICULTY CONTROL (CRITICAL):

You MUST strictly follow the cognitive level defined by DIFFICULTY.
Before finalizing each question, internally verify that it matches the allowed cognitive level.
If it exceeds the allowed level, you MUST simplify or regenerate it.

EASY difficulty:
- ONLY factual recall and basic understanding
- ONLY definitions, meanings, purposes, or simple descriptions
- Questions must NOT require reasoning, judgment, or justification
- Answerable in 1-2 short factual statements
- If a question can have multiple viewpoints, it is INVALID
- If a question requires explanation beyond basics, it is INVALID

MEDIUM difficulty:
- Conceptual understanding and explanation allowed
- Simple reasoning and illustrative examples allowed
- Limited comparison allowed (only when concepts are directly related)
- No critical evaluation or real-world impact analysis

HARD difficulty:
- Deep conceptual understanding required
- Critical thinking, limitations, assumptions, and applications allowed
- Real-world relevance and trade-offs allowed
- Questions may require structured, multi-paragraph answers

FINAL SELF-CHECK (MANDATORY):
- Re-read each generated question
- If it fits a higher difficulty than specified, downgrade it
- Ensure ALL questions strictly match the given difficulty level

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON in exactly this format:
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

