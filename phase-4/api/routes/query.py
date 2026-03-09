from fastapi import APIRouter
from graph.store import GraphDocStore
from graph.embedder import Embedder
from graph.search import GraphSearcher
from models.schemas import GraphQuery
from config.config import settings

router = APIRouter()
vector_store = GraphDocStore(settings.es_index, settings.embedding_dims)
searcher = GraphSearcher(vector_store, Embedder())

@router.post('/graphrag/query')
async def graphrag_query(q: GraphQuery):
    kb_id = q.kb_id or "kb_demo"
    llm_formatted_prompt = await searcher.query(kb_id, q)
    return {"graph_context": llm_formatted_prompt}