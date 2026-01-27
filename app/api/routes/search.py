from fastapi import APIRouter, HTTPException
import logging
from models.dtos import SearchResponse, SearchQuery
from core.searcher import searcher
from core.reranker import reranker



logger = logging.getLogger(__name__)
router = APIRouter()


@router.post('/', response_model=SearchResponse)
async def hybrid_search(query: SearchQuery):

    try:
        results, _ = await searcher.search(
            query=query.question,
            top_k=query.top_k * 2 if query.use_rerank else query.top_k
        )

        if query.use_rerank:
            results = reranker.rerank(query.question, results)
            results = results[:query.top_k]
            
        return SearchResponse(
            query=query.question,
            total=len(results),
            results=results
        )
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))