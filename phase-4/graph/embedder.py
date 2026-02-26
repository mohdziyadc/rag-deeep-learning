from sentence_transformers import SentenceTransformer
import numpy as np

model_name = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    def __init__(self) -> None:
        self.model = SentenceTransformer(model_name)
        # dimensions are needed to pass into the ES client for index
        # The key rule: the ES dense_vector dims must exactly match the model’s embedding size.
        self.dimensions = self.model.get_sentence_embedding_dimension()

    
    def embed(self, tokens: list[str]) -> np.ndarray:
        return self.model.encode(tokens, normalize_embeddings=True)