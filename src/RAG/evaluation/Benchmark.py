"""
import time
import statistics
from RAG.controllers.RagPipeline import RagPipeline
from RAG.helpers.config import get_settings
from main import COLLECTION_NAME, QDRANT_PATH, model_name

def load_rag():
    return RagPipeline(
        qdrant_db_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        llm_model_name=model_name
    )

rag = load_rag()
def benchmark_chatbot(rag, query, runs=50):
    latencies = []

    # Warm-up
    for _ in range(10):
        rag.ask(query=query, top_k=4)

    # Actual benchmark
    for _ in range(runs):
        start = time.perf_counter()
        rag.ask(query=query, top_k=4)
        end = time.perf_counter()
        latencies.append(end - start)

    return {
        "average": statistics.mean(latencies),
        "min": min(latencies),
        "max": max(latencies),
    
    }
    """
