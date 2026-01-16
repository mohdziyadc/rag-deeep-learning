from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

from app.config import get_settings

app = FastAPI(title="RAG Deep Learning API")


@app.get("/")
def root():
    return {"message": "RAG Deep Learning API is running"}


@app.get("/config")
def show_config():
    settings = get_settings()
    return {
        "es_host": settings.es_host,
        "es_index": settings.es_index,
        "embedding_model": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "bm25_weight": settings.bm25_weight,
        "vector_weight": settings.vector_weight,
        "top_k": settings.top_k,
    }
