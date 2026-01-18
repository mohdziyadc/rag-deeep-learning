import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    es_host: str
    es_index: str 
    database_url: str
    embedding_model: str
    embedding_dimension: int
    bm25_weight: float
    vector_weight: float
    top_k: int
    database_url: str

    class Config:
        env_file = '.env'


@lru_cache
def get_settings() -> Settings:
    return Settings()


























































