from fastapi import APIRouter
from pathlib import Path
from datetime import datetime, UTC
import json
import re
from graph.store import GraphDocStore
from graph.embedder import Embedder
from graph.search import GraphSearcher
from models.schemas import GraphQuery
from config.config import settings

router = APIRouter()
vector_store = GraphDocStore(settings.es_index, settings.embedding_dims)
searcher = GraphSearcher(vector_store, Embedder())

RETRIEVAL_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "retrieval-prompts"
RETRIEVAL_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:80] or "query"


def _store_retrieval_prompt(q: GraphQuery, kb_id: str, graph_context: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    file_name = f"{ts}_{_slugify(q.question)}.json"
    file_path = RETRIEVAL_PROMPTS_DIR / file_name
    payload = {
        "created_at": ts,
        "kb_id": kb_id,
        "request": q.model_dump(),
        "graph_context": graph_context,
    }
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return str(file_path)


@router.post("/graphrag/query")
async def graphrag_query(q: GraphQuery):
    kb_id = q.kb_id or "kb_demo"
    llm_formatted_prompt = await searcher.query(kb_id, q)
    stored_path = _store_retrieval_prompt(q, kb_id, llm_formatted_prompt)
    return {"graph_context": llm_formatted_prompt, "prompt_file": stored_path}
