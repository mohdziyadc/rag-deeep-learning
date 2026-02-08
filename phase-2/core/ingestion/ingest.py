

from typing import Any
import uuid
from core.parsers.registry import ParserRegistry
from core.normalization.normalizer import DocumentNormalizer
from core.chunking.chunker import DocumentChunker
from models.metadata import File, Document
from models.parsed import ParsedDocument
from db.metadata_store import metadata_store
from core.vectorstore.parsed_index import indexer


# Where everything happens
# Docs come, they get parsed, normalized, chunked, metadata captured, indexed.

class IngestionService:
    def __init__(self) -> None:
        self.registry = ParserRegistry()
        self.normalizer = DocumentNormalizer()
        self.chunker = DocumentChunker()


    async def ingest(
        self,
        file_name: str, 
        data: bytes, 
        metadata: dict[str, Any]
    ):
        doc_id = str(uuid.uuid4())
        file_id = str(uuid.uuid4())
        metadata = dict(metadata or {}) 

        await metadata_store.create_file(
            File(
                id=file_id,
                name=file_name,
                source=metadata.get("source", "local"),
                mime_type=metadata.get("mime_type"),
                size_bytes=len(data),
                metadata=metadata
            )
        )
        metadata["doc_id"] = doc_id
        metadata["file_id"] = file_id
        ext = file_name.split(".")[-1].lower()
        parser = self.registry.get_parser(ext)
        parsed_doc = parser.parse(file_name, data, metadata)
        normalized = self.normalizer.normalize(parsed_doc)

        chunks = self.chunker.chunk(normalized)

        await indexer.bulk_insert(chunks=chunks)

        await metadata_store.create_document(
            Document(
                id=doc_id,
                file_id=file_id,
                status="INDEXED",
                title=parsed_doc.title,
                file_type=parsed_doc.file_type,
                chunk_count=len(chunks),
                metadata=metadata
            )
        )

        return {
            "file_name": file_name,
            "doc_id": doc_id,
            "status": "INDEXED",
            "chunks_created": len(chunks),
            "file_type": parsed_doc.file_type
        }
        
