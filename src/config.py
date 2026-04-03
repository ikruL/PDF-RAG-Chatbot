from pathlib import Path
import os


class Config:

    EMBEDDING_MODEL = "gemini-embedding-2-preview"
    LLM_MODEL = "gemini-3.1-flash-lite-preview"

    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 150

    RETRIEVER_K = 5

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")


config = Config()
