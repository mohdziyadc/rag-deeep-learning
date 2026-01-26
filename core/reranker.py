
from app.config import get_settings
from sentence_transformers import CrossEncoder
import logging

from models.dtos import SearchResult


settings = get_settings()
logger = logging.getLogger(__name__)

class CrossEncoderReranker:


    def __init__(self) -> None:
        self.modelname = settings.rerank_model
        self.model = None

    def load(self):
        if self.model is None:
            logger.info(f"Loading cross-encoder: {self.modelname}")
            self.model = CrossEncoder(self.modelname)
            logger.info(f"Loaded the reranker model: {self.modelname}")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        **kwargs
    ) -> list[SearchResult]:
        
        if not results or self.model is None:
            return results

        pairs = [(query, r.content) for r in results]

        scores = self.model.predict(pairs)

        for i, result in enumerate(results):
            result.score = float(scores[i])
        
        results.sort(key=lambda x: x.score, reverse=True)

        return results



reranker = CrossEncoderReranker()

