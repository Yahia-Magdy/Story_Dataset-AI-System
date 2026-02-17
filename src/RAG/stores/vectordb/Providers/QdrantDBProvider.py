from importlib import metadata
from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
import logging
from typing import List
from RAG.models.db_schemes import RetrievedDocument

class QdrantDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str):

        self.client = None
        self.db_path = db_path
        self.distance_method = None

        if distance_method == "cosine":
            self.distance_method = models.Distance.COSINE
        elif distance_method == "dot":
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = QdrantClient(path=self.db_path)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)
    
    def list_all_collections(self) -> List:
        return self.client.get_collections()
    
    def get_collection_info(self, collection_name: str):
        return self.client.get_collection(collection_name=collection_name)
    
    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            return self.client.delete_collection(collection_name=collection_name)
        
    def create_collection(self, collection_name: str, 
                                embedding_size: int,
                                do_reset: bool = False):
        if do_reset:
            _ = self.delete_collection(collection_name=collection_name)
        
        if not self.is_collection_existed(collection_name):
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method,
                    hnsw_config=models.HnswConfigDiff(
                        m=16,
                        ef_construct=200
                        )
                )
            )

            return True
        
        return False
    
    def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False
        
        try:
            _ = self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=record_id,
                        vector=vector,
                        payload={
                            "text": text,
                            **(metadata or {}) 
                        }
                    )
                ]
            )
        except Exception as e:
            self.logger.error(f"Error while inserting batch: {e}")
            return False

        return True
    
    def insert_many(self, collection_name: str, texts: list, 
                          vectors: list, metadata: list = None, 
                          record_ids: list = None, batch_size: int = 128):
        
        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = list(range(0, len(texts)))

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_points  = [
                models.PointStruct(
                    id=batch_record_ids[x],
                    vector=batch_vectors[x],
                    payload={
                        "text": batch_texts[x],
                        **(batch_metadata[x] or {})
                    }
                )

                for x in range(len(batch_texts))
            ]

            try:
                _ = self.client.upsert(
                    collection_name=collection_name,
                    points=batch_points,
                )
            except Exception as e:
                self.logger.error(f"Error while inserting batch: {e}")
                return False

        return True
    


    def build_metadata_filter(self, metadata: dict | None):
        """
        Create a Qdrant Filter.
        Supports multiple values per field using MatchAny.
        """

        if not metadata:
            return None

        must_conditions = []

        for key, value in metadata.items():
            if not value:
                continue

            # Convert single value to list
            if not isinstance(value, list):
                value = [value]

            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=value)
                )
            )

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)


        
    def search_by_vector(self, collection_name: str, vector: list,
                         limit: int , ef: int = 128,  metadata: dict = None):
        filter_obj = self.build_metadata_filter(metadata)  
                             
        results = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            query_filter=filter_obj,
            search_params=models.SearchParams(
              hnsw_ef=ef
           )
        )

        if not results:
            return None
        
        return [
            RetrievedDocument(**{
                "score": result.score,
                "text": result.payload["text"],
                "metadata": {k: v for k, v in result.payload.items() if k != "text"}
            })
            for result in results.points
        ]