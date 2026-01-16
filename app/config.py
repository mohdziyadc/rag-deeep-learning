import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    es_host: str = os.getenv("ES_HOST")
    es_index: str = os.getenv("ES_INDEX")
    database_url: str = os.getenv("DATABASE_URL")
    embedding_model: str = os.getenv("EMBEDDING_MODEL")
    embedding_dimension: int = os.getenv("EMBEDDING_DIMENSION")
    bm25_weight: float = os.getenv("BM25_WEIGHT")
    vector_weight: float = os.getenv("VECTOR_WEIGHT")
    top_k: int = os.getenv("TOP_K")


@lru_cache
def get_settings() -> Settings:
    return Settings()


























































