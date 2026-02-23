from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:

    def __init__(self, model_name:str) -> None:
        self.modelname = model_name
        self.model: SentenceTransformer | None = None
        self.dimension = 384

    def load(self):
        if not self.model:
            self.model = SentenceTransformer(self.modelname)
            self.dimension = self.model.get_sentence_embedding_dimension()


    def encode(self, tokens: list[str]) -> tuple[np.ndarray, int]:
        if not self.model:
            self.load()
        
        embeddings = self.model.encode(
            tokens,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        token_count = sum(len(t.split()) for t in tokens)
        
        return np.array(embeddings), token_count

    
embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")