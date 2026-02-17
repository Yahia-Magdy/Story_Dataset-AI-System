from RAG.stores.llm.providers import GenerativeModel, E5SentenceEmbedder, quantized_model
from RAG.stores.vectordb.Providers import QdrantDBProvider
from typing import List, Optional, Dict
from setfit import SetFitModel
from rapidfuzz import process, fuzz
import json
import torch
import os
import re

class RagPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline with metadata filtering.
    """

    def __init__(
        self,
        qdrant_db_path: str,
        collection_name: str,
        llm_model_name: str,
        embed_device: Optional[str] = None,
    ):
        # Device selection
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize LLM
        self.llm = quantized_model(model_name=llm_model_name, device=device)

        # Initialize embedding model
        embed_device = embed_device or device
        self.embedder = E5SentenceEmbedder(device=embed_device)

        # Initialize vector DB
        self.qdrantdb = QdrantDBProvider(qdrant_db_path, distance_method="cosine")
        self.qdrantdb.connect()
        self.collection_name = collection_name
        # ----- Classification model -----
        self.classifier_model = self.load_classification_model()
        self.genres = self.load_genres()
        self.id2genre = {i: g for i, g in enumerate(self.genres)}

        
    def load_classification_model(self):
        """
        Automatically detect and load the fine-tuned SetFit model for CPU.
        """
        # Determine project root (src/)
        #project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        #model_path = os.path.join(project_root, "stores", "llm", "local_models", "sbert_setfit")

        #print(f"Loading classification model from {model_path}...")
        model = SetFitModel.from_pretrained("Yahia-123/Setfit_model")
        return model
    
    def load_genres(self):
        """
        Load genre list from genres.json located in the same directory
        as rag_pipeline.py
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        genres_path = os.path.join(current_dir, "genres.json")

        if not os.path.exists(genres_path):
            raise FileNotFoundError(f"Cannot find genres.json at {genres_path}")

        with open(genres_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        genres = data["genres"]

        if not isinstance(genres, list):
            raise ValueError("Invalid genres.json format. Expected {'genres': [...]}")

        return genres


    # ------------------------
    # Internal retrieval
    # ------------------------
    def fuzzy_match(self, value: str, valid_set: set, threshold: int = 85) -> str | None:
        match = process.extractOne(value, valid_set, scorer=fuzz.ratio)
        if match and match[1] >= threshold:  # match[1] is the score
            return match[0]
        return None
    
    def validate_metadata(
        self,
        raw_metadata: Dict,
        valid_titles: set,
        valid_genres: set,
        valid_ids: set
    ) -> Dict:
        """Validate and fuzzy-correct metadata extracted from query."""
        validated = {}

        # ---- Titles ----
        raw_titles = raw_metadata.get("title") or []
        resolved_titles = []
        for t in raw_titles:
            if not isinstance(t, str):
                continue
            t_clean = t.strip().lower()  # <-- lowercase here
            if t_clean in valid_titles:
                resolved_titles.append(t_clean)
            else:
                match = self.fuzzy_match(t_clean, valid_titles)
                if match:
                    resolved_titles.append(match)
        if resolved_titles:
            validated["title"] = list(set(resolved_titles))  # remove duplicates

        # ---- Genres ----
        raw_genres = raw_metadata.get("genre") or []
        resolved_genres = []
        for g in raw_genres:
            if not isinstance(g, str):
                continue
            g_clean = g.strip().lower()  # <-- lowercase here
            if g_clean in valid_genres:
                resolved_genres.append(g_clean)
            else:
                match = self.fuzzy_match(g_clean, valid_genres)
                if match:
                    resolved_genres.append(match)
        if resolved_genres:
            validated["genre"] = list(set(resolved_genres))

        # ---- IDs ----
        raw_ids = raw_metadata.get("id") or []
        resolved_ids = []
        for i in raw_ids:
            cleaned = re.sub(r"\D", "", str(i))

            if cleaned == "":   # prevents int("") crash
                continue

            i = int(cleaned)
            if i in valid_ids:
                resolved_ids.append(i)
        if resolved_ids:
            validated["id"] = list(set(resolved_ids))

        return validated
    
    
    def _retrieve_chunks(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve top-k relevant chunks using vector similarity and optional metadata.
        Returns a list of dicts: [{'text': ..., 'metadata': {...}}, ...]
        """
        if not hasattr(self, "valid_titles") or not hasattr(self, "valid_genres"):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
            valid_metadata_path = os.path.join(project_root, "assets", "validation_metadata", "valid_metadata.json")
            with open(valid_metadata_path, "r", encoding="utf-8") as f:
                valid_metadata = json.load(f)
            self.valid_titles = set(valid_metadata.get("titles", []))
            self.valid_genres = set(valid_metadata.get("genres", []))
            self.valid_ids = set(valid_metadata.get("ids", []))

        # 1. Parse metadata from query
        query_metadata = self.llm.parse_query_metadata(query)
        raw_metadata = query_metadata.model_dump()
        
        '''
        validated_metadata = {}
        # ---- TITLE (list now) ----
        titles = raw_metadata.get("title")
        if titles:
            normalized_titles = [
                t.strip().lower()
                for t in titles
                if isinstance(t, str) and t.strip()
            ]

            valid_titles = [
                t for t in normalized_titles
                if t in self.valid_titles
            ]

            if valid_titles:
                validated_metadata["title"] = valid_titles


        # ---- GENRE (list now) ----
        genres = raw_metadata.get("genre")
        if genres:
            normalized_genres = [
                g.strip().lower()
                for g in genres
                if isinstance(g, str) and g.strip()
            ]

            valid_genres = [
                g for g in normalized_genres
                if g in self.valid_genres
            ]

            if valid_genres:
                validated_metadata["genre"] = valid_genres


        # ---- ID (list now) ----
        ids = raw_metadata.get("id")
        if ids:
            # Ensure list of integers only
            valid_ids = [
                i for i in ids
                if isinstance(i, int)
            ]

            if valid_ids:
                validated_metadata["id"] = valid_ids
          '''
        validated_metadata = self.validate_metadata(raw_metadata, self.valid_titles,self.valid_genres, self.valid_ids)
        print(validated_metadata)
        # 2. Embed the query
        query_vector = self.embedder.embed_query(query).cpu().tolist()

        # 3. Search Qdrant with vector + metadata filter
        results = self.qdrantdb.search_by_vector(
            collection_name=self.collection_name,
            vector=query_vector,
            limit=top_k,
            metadata=validated_metadata if validated_metadata else None
        )

        if not results:
            return []

        # 4. Return both text and metadata
        return [{"text": r.text, "metadata": r.metadata} for r in results]


    # ------------------------
    # Public API
    # ------------------------
    def ask(
        self,
        query: str,
        top_k: int ,
        max_tokens: int = 1000,
        temperature: float = 0.3,
         ) -> str:
        
        chunks = self._retrieve_chunks(query, top_k=top_k)

        if chunks:
            context_texts = [c["text"] for c in chunks]
            context_metadata = [c["metadata"] for c in chunks]
        else:
            context_texts = []
            context_metadata = []

        return self.llm.answer_with_context(
            user_query=query,
            context=context_texts,
            metadata=context_metadata,
            max_tokens=max_tokens
        )
        #print("Context Texts")
        
    
    def classify_genre(self, text: str) -> str:
        
        prediction = self.classifier_model.predict([text])[0]
        prediction = int(prediction)
        return self.id2genre.get(prediction, "Unknown")

    def close(self):
        """Cleanup vector DB connection if needed."""
        self.qdrantdb.disconnect()
