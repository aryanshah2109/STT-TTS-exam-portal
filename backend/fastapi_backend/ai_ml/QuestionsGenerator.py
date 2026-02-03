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
Generate EXACTLY {num_questions} questions based strictly on the given TOPIC.

GENERAL RULES:
- Questions must be theory-based and verbally answerable
- NO code, NO programs, NO algorithms, NO implementations
- Language must be clear, simple, and suitable for written exams and viva
- Stay strictly within the given TOPIC

DIFFICULTY GUIDELINES (FOLLOW STRICTLY):

EASY difficulty:
- Generate simple textbook-style questions
- Focus on definitions, meanings, purposes, or basic descriptions
- Questions should be directly answerable from standard textbooks
- No deep thinking or analysis required
- Answers should be short and straightforward

MEDIUM difficulty:
- Generate questions that require understanding of concepts
- Allow explanation, reasoning, or simple examples
- Questions may connect related ideas within the topic
- Moderate thinking required, but not critical analysis

HARD difficulty:
- Generate questions that require critical thinking
- Allow discussion of limitations, implications, applications, or deeper insights
- Questions may require justification or structured explanation
- Suitable for long-answer or higher-mark questions

FINAL CHECK:
Ensure every question clearly matches the specified difficulty level.

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

        decoder = json.JSONDecoder()

        for i, ch in enumerate(text):
            if ch == "{":
                try:
                    obj, end = decoder.raw_decode(text[i:])
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
            elif hasattr(raw, "generations"):
                output = raw.generations[0][0].text
            else:
                output = str(raw)

            cleaned = self.sanitize_json(output)
            data = json.loads(cleaned)

            questions = data.get("questions", [])
            if not isinstance(questions, list):
                questions = []

            if len(questions) < num_questions:
                raise QuestionsGenerationException(
                    f"Expected {num_questions} questions, got {len(questions)}"
                )


            return questions[:num_questions]

        except Exception as e:
            raise QuestionsGenerationException(f"Generation failed: {str(e)}")

