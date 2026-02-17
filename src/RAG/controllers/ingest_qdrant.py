import os
from RAG.stores.vectordb import QdrantChunkIngestor


def run_ingestion():
    # Get absolute path to src/
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(CURRENT_DIR)

    # assets/qdrant path
    qdrant_path = os.path.join(SRC_DIR, "assets", "qdrant")

    # Ensure directory exists (safe)
    os.makedirs(qdrant_path, exist_ok=True)

    ingestor = QdrantChunkIngestor(
        qdrant_db_path=qdrant_path,
        collection_name="stories",
        batch_size=128,
    )

    success = ingestor.ingest(reset_collection=True)

    if success:
        print("Qdrant ingestion completed successfully.")
    else:
        print("Qdrant ingestion failed.")


if __name__ == "__main__":
    run_ingestion()
