import torch
import logging
from langsmith import traceable
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
from pydantic import BaseModel, Field


# ==========================================================
# Metadata Schema (UNCHANGED)
# ==========================================================

class QueryMetadata(BaseModel):
    id: Optional[List[Optional[int]]] = Field(default=None)
    genre: Optional[List[Optional[str]]] = Field(default=None)
    title: Optional[List[Optional[str]]] = Field(default=None)


# ==========================================================
# Generative Model
# ==========================================================

class GenerativeModelHistory:

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        device: str = None,
        default_max_tokens: int = 1000,
        default_temperature: float = 0.3,
    ):

        self.device = "cpu"
        self.model_name = model_name
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # -----------------------------
        # Load model
        # -----------------------------
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None
            ).to(self.device)

            self.logger.info(f"Loaded {model_name} on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None

        # -----------------------------
        # Simple short-term memory
        # -----------------------------
        self.history = []
        self.max_history_turns = 2


    # ==========================================================
    # Unified Chat Runner
    # ==========================================================

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


    # ==========================================================
    # LLM #1 — Structured Metadata Extraction (UNCHANGED)
    # ==========================================================

    def parse_query_metadata(
        self,
        user_query: str,
        schema: type[QueryMetadata] = QueryMetadata
    ) -> QueryMetadata:

        system_prompt = """
            You are a strict metadata query parser.

            Your task is to extract structured metadata from the user's query.

            Return ONLY a valid JSON object with EXACTLY these fields:

            {
            "id": [integer] | null,
            "genre": [string] | null,
            "title": [string] | null
            }

            STRICT RULES:
            1. Return ONLY JSON. No text before or after.
            2. All present values MUST be lists.
            3. If only one value exists, return a list with one element.
            4. If a field is not mentioned, return null .
            5. Do not generate anything you are an extractor not a generator.
            6. Return exactly the text mentioned in the user query.

            Example 1:
            Query: summarize the story with id 12
            Output:
            {"id": [12], "genre": null, "title": null}

            Example 2:
            Query: summarize the stories titled The Lost World and Hidden City in the adventure genre
            Output:
            {"id": null, "genre": ["adventure"], "title": ["Lost World and Hidden City", "Hidden City"]}
            """

        user_prompt = f"User Query:\n{user_query}\n\nReturn JSON only."

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


    # ==========================================================
    # LLM #2 — Answer With Context + Short-Term Memory
    # ==========================================================

    def answer_with_context(
        self,
        user_query: str,
        context: list[str],
        metadata: list[dict] | None = None,
        max_tokens: int = None,
        temperature: float = 0.3
    ) -> str | None:

        # --------------------------------------------------
        # Build formatted context
        # --------------------------------------------------
        if metadata:
            context_lines = []
            for i, meta in enumerate(metadata):
                meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
                context_lines.append(
                    f"Chunk {i+1} Metadata: {meta_str}\n{context[i]}"
                )
            full_context = "\n\n".join(context_lines)
        else:
            full_context = "\n\n".join(context)

        # --------------------------------------------------
        # Build conversation history (last 2 turns)
        # --------------------------------------------------
        history_text = ""
        for turn in self.history[-self.max_history_turns:]:
            history_text += f"User: {turn['user']}\n"
            history_text += f"Assistant: {turn['assistant']}\n\n"

        # --------------------------------------------------
        # Construct messages
        # --------------------------------------------------
        messages = [
            {
                "role": "system",
                "content": (
                            "You are a helpful assistant that answers the user's questions using the provided context.",
                            "The context may include both metadata about each story and the story text.",
                            "IMPORTANT INSTRUCTIONS:",
                            "1) Metadata fields (id, title, genre) are factual. If the question asks about them, return the metadata value directly.",
                            "2) The story text contains narrative or plot details. Use it when the question is about story content.",
                            "3) Do NOT use information outside the provided context or metadata.",
                            "4) If the question asks for story content (e.g., summary, characters, events), you may combine information from multiple chunks that share the same id.",
                            "5) You are allowed to summarize, paraphrase, and synthesize information from the provided chunks.",
                            "",
                            "Below you will be provided with multiple chunks. Each chunk starts with:",
                            "Chunk X Metadata: id: <id>, title: <title>, genre: <genre> on one line, followed by the story text.",
                            "",
                            "Format:",
                            "Context with metadata:",
                            "Chunk 1 Metadata: …",
                            "<story text…>",
                            "",
                            "Chunk 2 Metadata: …",
                            "<story text…>"
                )
            },
            {
                "role": "user",
                "content": "\n".join([
                    f"Conversation History:\n{history_text}",
                    f"\nQuestion:\n{user_query}",
                    f"\nContext:\n{full_context}"
                ])
            }
        ]

        # --------------------------------------------------
        # Generate response
        # --------------------------------------------------
        response = self.run_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # --------------------------------------------------
        # Save to memory (keep only last 2 turns)
        # --------------------------------------------------
        if response:
            self.history.append({
                "user": user_query,
                "assistant": response
            })

            if len(self.history) > self.max_history_turns:
                self.history = self.history[-self.max_history_turns:]

        return response
