# app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("Story Dataset AI System")

mode = st.sidebar.selectbox(
    "Select mode",
    ["Ask a question", "Classify text"]
)

user_input = st.text_area("Enter your text here:")

if st.button("Run") and user_input.strip():

    if mode == "Ask a question":
        with st.spinner("Fetching answers..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"text": user_input}
            )
            result = response.json()["answer"]
            st.write(result)

    elif mode == "Classify text":
        with st.spinner("Classifying text..."):
            response = requests.post(
                f"{API_URL}/classify",
                json={"text": user_input}
            )
            result = response.json()["genre"]
            st.write(result)
        

