from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ParsedSection(BaseModel):
    text: str
    section_type: str = "text"  # text, table, image, metadata
    content_format: str # text, json, html
    page: Optional[int] = None
    title: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    doc_id: str
    source_name: str
    file_type: str
    title: str
    sections: list[ParsedSection]
    raw_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkedDocument(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    source_name: str
    file_type: str
    section_type: str
    content_format: str
    page: Optional[int] = None
    title: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
