from pydantic import BaseModel, Field
from typing import Optional

class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    title: str
    metadata: dict = Field(default_factory=dict)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    messages: list[ChatMessage] = Field(default_factory=list)
    top_k: int = 5
    stream: bool = True
    max_tokens: int = 4096
    qoute: bool = True


class ReferenceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    content: str
    metadata: dict = Field(default_factory=dict)


class ReferenceBundle(BaseModel):
    chunks: list[ReferenceChunk]
    doc_aggs: list[dict] = Field(default_factory=list)

class ChatResponse(BaseModel):
    answer: str
    reference: ReferenceBundle | None = None
    prompt: str | None = None



# ReferenceChunk / ReferenceBundle note
# ReferenceChunk represents a single source chunk returned to the client (id, doc_id, title, content, metadata).
# ReferenceBundle is the response wrapper that groups all reference chunks (and optional doc-level aggregates in doc_aggs).
# Server-side attribution: after generation, used chunk IDs (used_ids) are determined (e.g., via citation insertion),
# then ReferenceBundle is built from those used chunks so the UI can map citations like [ID:x] to sources.