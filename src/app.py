# main.py
from RAG.controllers.RagPipeline import RagPipeline
from RAG.helpers.config import get_settings
from langsmith import Client
from langchain_core.tracers.context import tracing_v2_enabled
import os
import logging
from groq import Groq
import os
import streamlit as st
import time  # <-- add this at the top


# -------------------------------
# 1. Paths & collection settings
# -------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_DIR,"RAG","assets")
QDRANT_PATH = os.path.join(ASSETS_DIR, "qdrant")

settings = get_settings()

os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

COLLECTION_NAME = settings.COLLECTION_NAME
model_name = settings.model_name_qwen

client = Client()

# logs


# -------------------------------
# 2. Initialize RAG pipeline (cached)
# -------------------------------
@st.cache_resource
def load_rag():
    return RagPipeline(
        qdrant_db_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        llm_model_name=model_name
    )

rag = load_rag()

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.title("Story Dataset AI System")

mode = st.sidebar.selectbox(
    "Select mode",
    ["Ask a question", "Classify text"]
)

user_input = st.text_area("Enter your text here:")

if st.button("Run") and user_input.strip():

    with tracing_v2_enabled(project_name="Story-Retrieval", client=client):

        if mode == "Ask a question":
            with st.spinner("Fetching answers..."):
                start_time = time.time()
                answer = rag.ask(query=user_input, top_k=4)
                end_time = time.time()
                latency = end_time - start_time
                print(f"[Chatbot] Response Time: {latency:.4f} seconds")               
                st.write(answer)

        elif mode == "Classify text":
            with st.spinner("Classifying text..."):
                start_time = time.time()
                genre_name = rag.classify_genre(user_input)
                end_time = time.time()
                latency = end_time - start_time
                print(f"[Classifier] Response Time: {latency:.4f} seconds")
                st.write(genre_name)

# -------------------------------
# 4. Shutdown
# -------------------------------
if st.button("Shutdown RAG"):
    with st.spinner("Cleaning up resources..."):
        rag.close()
        st.cache_resource.clear()
        st.success("RAG pipeline closed and cache cleared.")
