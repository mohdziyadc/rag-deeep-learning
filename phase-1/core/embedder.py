from app.config import get_settings
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

class Embedder:

    def __init__(self, model_name: str = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.model: SentenceTransformer | None = None
        self.dimension = settings.embedding_dimension

    
    def load(self):
        if not self.model:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self.dimension}")
    
    # convert tokens to embeddings
    def encode(self, tokens: list[str]) -> tuple[np.ndarray, int]:
        if self.model is None:
            self.load()

        embeddings = self.model.encode(
            tokens,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        token_count = sum(len(t.split()) for t in tokens)

        return np.array(embeddings), token_count

    def encode_query(self, query: str) -> tuple[np.ndarray, int]:
        embeddings, token_count = self.encode([query])
        return embeddings[0], token_count


embedder = Embedder() 