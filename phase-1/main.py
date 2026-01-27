from dotenv import load_dotenv

from core.embedder import embedder
from core.searcher import searcher
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI

from core.indexer import indexer
from core.reranker import reranker
from app.api.routes import documents
from app.api.routes import search

from app.config import get_settings
import logging
logging.basicConfig(level=logging.INFO, force=True)


logger = logging.getLogger(__name__)
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await indexer.connect()
    await indexer.create_index()
    await searcher.connect()
    embedder.load()
    reranker.load()
    logger.info("Mini RAG ready!")
    
    yield

    logger.info("Shutting down...")
    await indexer.close()
    await searcher.close()

app = FastAPI(
    title="Mini RAG", 
    lifespan=lifespan,
    version="0.0.0",
)

app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(search.router, prefix="/api/search", tags=["Search"] )


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
        "db_url": settings.db_url
    }


@app.get("/health/es")
async def es_health():
    try:
        stats = await indexer.get_stats()
        return {"status": "healthy", "stats": stats}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get('/embedder_test')
def embed_test():
    vectors, tokens = embedder.encode(["zee 4 lyf", "lowkey tha goat"])
    return {"shape": list(vectors.shape), "tokens": tokens, "embeddings": vectors.tolist()}


# @app.get('/es-search')
# def es_search_test():
#     es_searcher = searcher._build_hybrid_query("", [0.1], 10)
#     return {"es_searcher": es_searcher}