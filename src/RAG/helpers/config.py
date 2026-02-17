from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):

    LANGSMITH_TRACING: str
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str

    COLLECTION_NAME: str
    model_name_qwen: str
    model_name_quant: str
    GROQ_API_KEY: str

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()