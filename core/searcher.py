

from elasticsearch import AsyncElasticsearch
from app.config import get_settings
from core.embedder import embedder
from core.tokenizer import Tokenizer
import numpy as np
from models.dtos import SearchResult
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


class HybridSearcher:

    def __init__(self) -> None:
        self.client: AsyncElasticsearch | None = None
        self.indexname = settings.es_index
        self.tokenizer = Tokenizer()

        self.bm25_weight = settings.bm25_weight
        self.vector_weight = settings.vector_weight

    
    async def connect(self):
        if not self.client or self.client is None:
            self.client = AsyncElasticsearch(
                hosts=[settings.es_host],
                request_timeout=30
            )

    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None

    async def search(
        self, 
        query:str,
        top_k: int = 10,
        similarity_threshold: float = 0.2
    ) -> tuple[list[SearchResult], list[float]]:

        # for BM25
        query_tokens = self.tokenizer.tokenize_query(query)
        query_text = " ".join(query_tokens)
        
        # for KNN
        embedder.load()
        query_vector, _ = embedder.encode_query(query)

        search_query = self._build_hybrid_query(
            query_text=query_text,
            query_vector=query_vector.tolist(),
            top_k=top_k * 2
        )
        
        response = await self.client.search(
            index=self.index_name,
            body=search_query
        )


        results = self._process_results(
            response, 
            query_vector, 
            similarity_threshold
        )

        return results[:top_k], query_vector.tolist()


    def _build_hybrid_query(
        self, 
        query_text: str, 
        query_vector: list[float],
        top_k: int
    ) -> dict:

        pass

    # fusion
    def _process_results(
        self, 
        response: dict, 
        query_vector: np.ndarray,
        similarity_threshold: float        
    ) -> list[SearchResult]:
        pass
