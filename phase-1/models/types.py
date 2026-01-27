
from datetime import datetime
from typing import Any, TypedDict


class Chunk(TypedDict):
    chunk_id: str
    doc_id: str
    content_ltks: str
    content: str
    title_tks: str
    chunk_index: int
    created_at: datetime | None
    metadata: dict[str, Any]
    # Need a dynamic field which would take 
    # in the vectotr embeddings