from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str
    content: str
    metadata: dict = Field(default_factory=dict)

class SearchQuery(BaseModel):
    question: str
    top_k: int = 10
    similarity_threshold: float = 0.2
    use_rerank: bool = True


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    title: str
    score: float
    bm25_score: float
    vector_score: float

class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SearchResult]