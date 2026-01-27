import uuid
import logging
from fastapi import APIRouter, HTTPException

from models.dtos import DocumentCreate
from core.chunker import chunker
from core.indexer import indexer


logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ingest")
async def ingest_document(document: DocumentCreate):

    """
    Ingest a document into the RAG system.
    
    Steps:
    1. Generate unique doc_id
    2. Chunk the document
    3. Generate embeddings
    4. Index into Elasticsearch
    """

    doc_id = str(uuid.uuid4())
    try:

        chunks = chunker.chunk_and_embed(
            doc_id=doc_id,
            title=document.title,
            content=document.content,
            metadata=document.metadata
        )

        if not chunks:
            raise HTTPException(status_code=500, detail="Error occured during chunking and embedding process")

        await indexer.bulk_index(chunks)

        return {
            "doc_id": doc_id,
            "title": document.title,
            "chunks_count": len(chunks),
            "is_indexed": True 
        }

    except Exception as e:
        logger.error(f'Error ingesting document: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@router.delete('/{doc_id}')
async def delete_doc(doc_id: str):
    try:
        await indexer.delete_doc_by_id(doc_id)
        return {"message": f"Document {doc_id} deleted"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get index statistics"""
    try:
        stats = await indexer.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


