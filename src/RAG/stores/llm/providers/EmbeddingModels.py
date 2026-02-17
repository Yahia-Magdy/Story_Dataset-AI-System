from sentence_transformers import SentenceTransformer
import torch
from typing import List

class E5SentenceEmbedder:
    def __init__(self, model_name: str = "intfloat/e5-small", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def embed_documents(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Embed document chunks (stories, passages).
        """
        passages = [f"passage: {text}" for text in texts]
        return self.model.encode(
            passages,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )

    def embed_query(self, query: str, normalize: bool = True) -> torch.Tensor:
        """
        Embed a user query for retrieval.
        """
        query_text = f"query: {query}"
        return self.model.encode(
            query_text,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )
