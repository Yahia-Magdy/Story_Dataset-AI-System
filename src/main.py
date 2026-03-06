from fastapi import FastAPI
from pydantic import BaseModel
import logging
import os
import time  # <-- add this at the top
import asyncio

from langsmith import Client
from langchain_core.tracers.context import tracing_v2_enabled

from RAG.controllers.RagPipeline import RagPipeline
from RAG.helpers.config import get_settings
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import prometheus_client

REQUEST_COUNT = Counter(
    "fastapi_requests_total",
    "Total number of requests",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "fastapi_request_latency_seconds",
    "Latency per endpoint",
    ["endpoint"]
)



# -------------------------------
# 1. App Initialization
# -------------------------------
app = FastAPI(title="Story Dataset AI System API")
settings = get_settings()
# -------------------------------
# 2. Paths & collection settings
# ------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_DIR,"RAG","assets")
QDRANT_PATH = os.path.join(ASSETS_DIR, "qdrant")

os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

COLLECTION_NAME = settings.COLLECTION_NAME
model_name = settings.model_name_quant

client = Client()

# -------------------------------
# 3. Request Schema
# -------------------------------
class QueryRequest(BaseModel):
    text: str
    
# -------------------------------
# 4. Load Pipeline (on startup)
# -------------------------------
@app.on_event("startup")
async def load_pipeline():
    global rag
    
    rag = RagPipeline(
        qdrant_db_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        llm_model_name=model_name
    )
    print("✅ RAG Pipeline Loaded")


# -------------------------------
# 5. Endpoints (ASYNC + THREAD SAFE)
# -------------------------------

@app.post("/ask")
async def ask_question(request: QueryRequest):
    start_time = time.time()
    
    REQUEST_COUNT.labels(endpoint="ask").inc()

    def task():
        with tracing_v2_enabled(project_name="Story-Retrieval", client=client):
            return rag.ask(query=request.text, top_k=4)

    answer = await run_in_threadpool(task)

    latency = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint="ask").observe(latency)
    print(f"[Chatbot] Response Time: {latency:.4f} seconds")

    return {
        "answer": answer,
    }


@app.post("/classify")
async def classify_text(request: QueryRequest):
    start_time = time.time()
    REQUEST_COUNT.labels(endpoint="classify").inc()

    def task():
        with tracing_v2_enabled(project_name="Story-Retrieval", client=client):
            return rag.classify_genre(request.text)

    genre = await run_in_threadpool(task)

    latency = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint="classify").observe(latency)
    print(f"[Classifier] Response Time: {latency:.4f} seconds")

    return {
        "genre": genre,
    }

@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# -------------------------------
# 6. Shutdown
# -------------------------------
@app.on_event("shutdown")
async def shutdown():
    await run_in_threadpool(rag.close)
    print("🛑 RAG Pipeline Closed")
