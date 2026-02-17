from importlib.metadata import metadata
import torch
import logging
import json
from langsmith import traceable
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator


class QueryMetadata(BaseModel):
    id: Optional[List[Optional[int]]] = Field(
        default=None,
        description="The unique identifier of the story or document. Should be an integer."
    )
    genre: Optional[List[Optional[str]]] = Field(
        default=None,
        description="The genre of the story, e.g., 'sci-fi', 'romance', 'politics'. Lowercase string."
    )
    title: Optional[List[Optional[str]]] = Field(
        default=None,
        description="The title of the story or document. Return the exact title if present, otherwise null."
    )

class GenerativeModel:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device: str = None,
        default_max_tokens: int = 1000,
        default_temperature: float = 0.1,
    ):
        self.device = "cpu"

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.model_name = model_name
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None
            ).to("cpu")
            self.logger.info(f"Loaded {model_name} on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

    # ------------------------------------------------------------------
    # Internal unified chat runner
    # ------------------------------------------------------------------
    @traceable(run_type="llm", name="run_chat")
    def run_chat(
        self,
        messages: list,
        max_tokens: int = None,
        temperature: float = None
    ) -> str | None:

        if not self.model or not self.tokenizer:
            self.logger.error("Model not initialized")
            return None

        max_tokens = max_tokens or self.default_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature
            )

            return self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return None

    # ------------------------------------------------------------------
    # LLM #1 — Structured metadata extraction
    # ------------------------------------------------------------------
    def parse_query_metadata(
        self,
        user_query: str,
        schema: type[QueryMetadata] = QueryMetadata
    ) -> QueryMetadata:

        system_prompt = """
            You are a strict JSON metadata extractor.

            Your task is to extract metadata fields from a user query.

            You MUST return ONLY valid JSON.
            Do NOT include explanations.
            Do NOT include markdown.
            Do NOT include text before or after JSON.

            The JSON structure MUST be exactly:

            {
            "id": [integer] | null,
            "genre": [string] | null,
            "title": [string] | null
            }

            RULES:
            1. All fields MUST be present.
            2. If a field is not mentioned → return null.
            3. If one value is found → return a list with one element.
            4. If multiple values are found → return a list of them.
            5. DO NOT fix typos.
            6. DO NOT invent values.
            7. Titles and genres must match EXACT text from the query.
            8. id must be integer and must present in the query.
            9. Return null instead of empty list.
            10. Output must be pure JSON only.
            11. After extracting metadata, check that every data extracted is actually present in the query.

            EXAMPLES:

            Query: what is the id of story 458,541
            Output:
            {
            "id": [458541],
            "genre": null,
            "title": null
            }

            Query: show me sci-fi and horror stories
            Output:
            {
            "id": null,
            "genre": ["sci-fi", "horror"],
            "title": null
            }

            Query: what is the id of story 'The walking dead'
            Output:
            {
            "id": null,
            "genre": null,
            "title": ["The walking dead"]
            }
            """

        user_prompt = f"Now extract metadata from this query:\n{user_query}\n\nReturn JSON only."

        messages = [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_prompt}
                ]

        raw_output = self.run_chat(
            messages=messages,
            max_tokens=256,
            temperature=0.1
        )

        try:
            return schema.model_validate_json(raw_output)
        except Exception as e:
            self.logger.error(f"Parsing failed: {e}")
            self.logger.info(f"Raw LLM output: {raw_output}")
            return schema()

    # ------------------------------------------------------------------
    # LLM #2 — Final answer generation
    # ------------------------------------------------------------------
    def answer_with_context(
    self,
    user_query: str,
    context: str,
    metadata: list[dict] | None = None,
    max_tokens: int = None,
    temperature: float = 0.1
        ) -> str | None:
            """
            Ask the LLM to answer using the context and optionally include metadata.
            Each metadata entry corresponds to the chunk at the same index in the context.
            """

            # Prepare context with metadata
            if metadata:
                context_lines = []
                for i, meta in enumerate(metadata):
                    meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
                    context_lines.append(f" {i+1} Metadata: {meta_str}\n{context[i]}")
                full_context = "\n\n".join(context_lines)
            else:
                full_context = context

            messages = [
                {
                    "role": "system",
                    "content": "\n".join([
"You are a helpful assistant that answers the user's questions using the provided context.", 
"The context include both metadata about each story and the story text.", 
"IMPORTANT INSTRUCTIONS:",
"1) Metadata fields (id, title, genre) are factual. If the question asks about them, return the metadata value directly.", 
"2) The story text contains narrative or plot details. Use it when the question is about story itself.",
"3) Do NOT use information outside the provided context or metadata.", 
"4) If the question asks for story content (e.g., summary, characters, events, plots, themes), you may combine information from multiple chunks.", 
"5) You are allowed to summarize, paraphrase, and synthesize information from the provided chunks.", 
"6)Take care that the answer you provided is related to the question and the answer is from the 'context with metadata' " 
"7) Answer the user question and you will always find the answer in the context with metadata.",
"8) The context information are raw data you should first extract the related information from it to answer the user question  "
"9) Never giving the user the context as it is use should always handle the context first and to see what is related to the user and answer."
"",


                    ])
                },
                {
                    "role": "user",
                    "content": "\n".join([
                        f"Question:\n{user_query}\n\n",
                        f"Context with metadata:\n{full_context}"
                    ])
                }
            ]

            return self.run_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )

