import torch
from typing import List
from tqdm import tqdm
from RAG.models.data_preprocessing import StoryDatasetProcessor
from RAG.stores.llm.providers import E5SentenceEmbedder
from .Providers.QdrantDBProvider import QdrantDBProvider

class QdrantChunkIngestor:
    """
    Handles end-to-end ingestion of normalized text chunks into Qdrant:
    - extract text + metadata
    - embed chunks
    - create collection
    - batch insert into Qdrant
    """

    def __init__(
        self,
        qdrant_db_path: str,
        collection_name: str,
        batch_size: int = 128,
        embed_device: str = None,
    ):
        """
        qdrant_provider : QdrantDBProvider
        embedder        : E5SentenceEmbedder
        collection_name : str
        batch_size      : int (embedding + upsert batch size)
        """
        self.qdrantdb = QdrantDBProvider(qdrant_db_path, distance_method="cosine")
        self.collection_name = collection_name
        self.batch_size = batch_size
        # Initialize dataset processor
        self.processor = StoryDatasetProcessor()
        # Preprocess dataset immediately
        self._chunked_documents = None
        # Initialize E5 embedder with device safety
        device = embed_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = E5SentenceEmbedder(device=device)

    @property
    def chunked_documents(self):
        if self._chunked_documents is None:
            self._chunked_documents = self.processor.process_all()
        return self._chunked_documents


    # --------------------------------------------------
    # Main public API
    # --------------------------------------------------
    def ingest(self, reset_collection: bool) -> bool:
        """
        Full ingestion pipeline.
        """

        chunked_documents = self.chunked_documents
        if not chunked_documents:
            raise ValueError("No chunks found for ingestion.")
        num_chunks = len(chunked_documents)
        
        texts = [doc.page_content for doc in chunked_documents]
        metadata = [doc.metadata or {} for doc in chunked_documents]

        # 1. Embed all chunks
        vectors = self._embed_texts(texts)
        if not vectors:
            raise ValueError("Embedding failed. No vectors produced.")
        num_vectors = len(vectors)  

        # 3. Create collection (once)
        embedding_dim = len(vectors[0])

        self.qdrantdb.connect()
        self.qdrantdb.create_collection(
            collection_name=self.collection_name,
            embedding_size=embedding_dim,
            do_reset=reset_collection,
        )

        # 4. Insert into Qdrant
        success = self.qdrantdb.insert_many(
            collection_name=self.collection_name,
            texts=texts,
            vectors=vectors,
            metadata=metadata,
            batch_size=self.batch_size,
        )
        collection_info = self.qdrantdb.get_collection_info(self.collection_name)
        total_points = collection_info.points_count

        self.qdrantdb.disconnect()
        print("\nIngestion summary")
        print("-" * 40)
        print(f"Chunks created           : {num_chunks}")
        print(f"Vectors embedded         : {num_vectors}")
        print(f"Points inserted this run : {num_vectors}")
        print(f"Total points in Qdrant   : {total_points}")
        print("-" * 40)
        
        return success

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts in batches and return List[List[float]]
        """

        all_vectors = []

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding chunks"):
            batch_texts = texts[i : i + self.batch_size]

            with torch.no_grad():
                batch_embeddings = self.embedder.embed_documents(batch_texts)

            all_vectors.extend(batch_embeddings.cpu().tolist())

        return all_vectors
