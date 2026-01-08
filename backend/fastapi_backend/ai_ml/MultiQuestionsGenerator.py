from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from pydantic import BaseModel, Field
from typing import List, Dict, Annotated, Optional
import re

from ai_ml.ModelCreator import HFModelCreation
from ai_ml.AIExceptions import *


class OutputResponse(BaseModel):
    questions: Annotated[List[str], Field(title="Questions", description="The questions created by the model")]



class MultiQuestionsGenerator:
    def __init__(self, model_name: str, global_model = None):
        self.model_name = model_name
        self.model = global_model

    def get_model(self):
        if self.model is None:
            self.model = HFModelCreation.hf_model_creator(self.model_name)
        return self.model

    
    def chain_creator(self):
        parser = JsonOutputParser(pydantic_object=OutputResponse)

        template = """
You are an exam question generator.

Your task is to generate exactly {num_questions} questions
using ONLY the subjects and topics provided below.

SUBJECT-TOPIC MAPPING RULES:
- Each topic belongs to exactly one subject.
- Infer the most reasonable subject for each topic based on academic context.
- DO NOT invent new topics.
- DO NOT use topics outside the given topic list.
- If a subject has no matching topic, generate ZERO questions for that subject.

GENERATION RULES:
- Every question MUST clearly belong to one subject AND one topic.
- Questions must be well-structured, clear, and exam-appropriate.
- The total number of generated questions MUST be exactly {num_questions}.
- Do NOT exceed or fall short of the requested number.

OUTPUT FORMAT RULES:
- Return ONLY valid JSON.
- Do NOT include explanations, markdown, or extra text.
- Follow the JSON schema strictly.

Topic List:
{topic_list}

Subject List:
{subject_list}

{format_instructions}

"""

        prompt = PromptTemplate(
            template=template,
            input_variables= ["num_questions", "topic_list", "subject_list"],
            partial_variables= {
                "format_instructions": parser.get_format_instructions()
            }
        )


        chain = prompt | self.get_model()

        return chain, parser

    def sanitize_json(self, text: str) -> str:
        text = text.replace("```json", "").replace("```", "").strip()
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            text = m.group(0)
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        return text


    def create_questions(self, input_request: dict):
        if "topic_list" not in input_request:
            raise KeyError("Input request must contain the topic list related to which you want questions")
        
        elif "subject_list" not in input_request:
            raise KeyError("Input request must contain the subject list related to which you want questions")

        elif "num_questions" not in input_request:
            raise KeyError("Input request must contain the number of questions you want related to the topic")
        
        try:
            
            chain, parser = self.chain_creator()

            raw = chain.invoke(input_request)

            # Extract actual text reliably
            if isinstance(raw, dict) and "text" in raw:
                output = raw["text"]

            elif isinstance(raw, dict) and "generated_text" in raw:
                output = raw["generated_text"]

            elif hasattr(raw, "generations"):
                output = raw.generations[0][0].text

            elif isinstance(raw, list) and isinstance(raw[0], dict) and "generated_text" in raw[0]:
                output = raw[0]["generated_text"]

            else:
                output = str(raw)

            cleaned = self.sanitize_json(output)
            return parser.parse(cleaned)

        except Exception as e:
            print(f"Some error occured! Details: {e}")