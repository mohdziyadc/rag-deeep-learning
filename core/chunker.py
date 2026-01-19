from datetime import datetime
import logging
import uuid

from core.embedder import embedder
from core.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

class NaiveChunker:

    def __init__(
        self, 
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.tokenizer = Tokenizer()

    
    def chunk_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        metadata: dict = None
    ) -> list[dict]:

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + self.chunk_size
            # Try to break at a sentence or paragraph boundary
            for punct in ['. ', '! ', '? ', '\n\n', '\n']:
                pos = content.rfind(punct, start, end)
                if pos > start + self.min_chunk_size:
                    end = pos + len(punct)
                    break
            chunk_text = content[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "content": chunk_text,
                    "content_tokens": self.tokenizer.tokenize(chunk_text),
                    "title": title,
                    "metadata": metadata or {},
                    "created_at": datetime.utcnow().isoformat()
                })

                chunk_idx += 1

            start = end - self.chunk_overlap

        logger.info(f"Created {len(chunks)} chunks from document {doc_id}")
        return chunks

    def chunk_and_embed(
        self,
        doc_id: str,
        title: str,
        content: str,
        metadata: dict = None
    ) -> list[dict]:

        chunks = self.chunk_document(doc_id, title, content, metadata)

        if not chunks:
            return chunks
        
        embedder.load()
        # We are finding the embeddings for the content in each chunk
        chunk_texts = [c["content"] for c in chunks]
        embeddings, _ = embedder.encode(chunk_texts)

        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        return chunks

chunker = NaiveChunker()



