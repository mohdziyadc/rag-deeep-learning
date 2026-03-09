from fastapi import APIRouter
import networkx as nx
from config.config import settings
from llm.client import LLMClient
from graph.embedder import Embedder
from graph.extractor import GraphExtractor
from graph.community import CommunityReportBuilder
from graph.merge import graph_merge
from graph.resolve import resolve_entities
from graph.store import GraphDocStore
from models.schemas import GraphExtractionResult

router = APIRouter()
llm = LLMClient(api_key=settings.llm_api_key, model=settings.llm_model)
embedder = Embedder()
vector_store = GraphDocStore(index_name=settings.es_index, embedding_dims=settings.embedding_dims)
community_builder = CommunityReportBuilder(llm=llm)



@router.post('/graphrag/build')
async def build_graph(payload: dict):
    kb_id = payload.get("kb_id") or "kb_demo"
    chunks = payload["chunks"]
    # this is stored at a KB level in ragflow's DB. If not present, defaulted to a set of entity types
    entity_types = payload.get("entity_types", ["organization", "person", "geo", "event", "category"])
    reset = payload.get("reset", False)

    extractor = GraphExtractor(llm, entity_types)
    base = nx.Graph() if reset else await vector_store.load_graph(kb_id)

    if reset:
        await vector_store.delete_kb_docs(kb_id=kb_id)

    base = await merge_chunks(kb_id, chunks, extractor, base)
    base = resolve_entities(base)

    apply_pagerank(base)

    await index_graph_snapshot(kb_id, base)
    await index_entities(kb_id, base)
    await index_relations(kb_id, base)
    await index_community_reports(kb_id, base)

    return {"nodes": base.number_of_nodes(), "edges": base.number_of_edges(), "kb_id": kb_id}



async def merge_chunks(
    kb_id: str, 
    chunks: list[dict], 
    extractor: GraphExtractor,
    base: nx.Graph
) -> nx.Graph:

    for chunk in chunks:
        doc_id = chunk.get("doc_id", "doc_unknown")
        result = extractor.extract(chunk["content"])
        subgraph = build_subgraph(doc_id, result)
        await vector_store.upsert_subgraph(kb_id, doc_id, subgraph)
        base = graph_merge(base, subgraph)
    return base



def build_subgraph(doc_id: str, result: GraphExtractionResult) -> nx.Graph:
    subgraph = nx.Graph()
    subgraph.graph["source_id"] = [doc_id]

    for ent in result.entities:
        data = ent.model_dump()
        data["source_id"] = [doc_id]
        subgraph.add_node(ent.entity_name, **data)
    
    for rel in result.relations:
        data = rel.model_dump()
        data["source_id"] = [doc_id]
        data["weight"] = data.pop("strength", 0)
    
        if subgraph.has_node(rel.src_id) and subgraph.has_node(rel.tgt_id):
            subgraph.add_edge(rel.src_id, rel.tgt_id, **data)
    return subgraph

def apply_pagerank(graph: nx.Graph) -> None:
    pr = nx.pagerank(graph) if len(graph) else {}

    for node, score in pr.items():
        graph.nodes[node]["pagerank"] = score


async def index_graph_snapshot(kb_id:str, graph: nx.Graph) -> None:
    await vector_store.delete_graph_docs(kb_id)
    await vector_store.upsert_graph(kb_id, graph)



async def index_entities(kb_id: str, graph: nx.Graph) -> None:
    entity_names = list(graph.nodes())
    entity_texts = [graph.nodes[n].get("description", "") for n in entity_names]
    entity_vecs = embedder.embed(entity_texts) if entity_texts else []

    for name, vec in zip(entity_names, entity_vecs):
        ent = graph.nodes[name]
        await vector_store.index_entity(kb_id, {
            "entity_name": name,
            "entity_type": ent.get("entity_type", "-"),
            "description": ent.get("description", ""),
            "pagerank": ent.get("pagerank", 0),
        }, vec.tolist())

async def index_relations(kb_id: str, graph: nx.Graph) -> None:
    relations = list(graph.edges(data=True))
    rel_texts = [r[2].get("description", "") for r in relations]
    rel_vecs = embedder.embed(rel_texts) if rel_texts else []

    for (src, tgt, data), vec in zip(relations, rel_vecs):
        await vector_store.index_relation(
            kb_id, 
            {
                "src_id": src,
                "tgt_id": tgt,
                "description": data.get("description", ""),
                "strength": data.get("weight", 0)
            },
            vec.tolist()
        )

async def index_community_reports(kb_id: str, graph: nx.Graph) -> None:
    
    communities = community_builder.build_communities(graph)
    for _, nodes in communities.items():
        weight = len(nodes) / max(1, graph.number_of_nodes())
        report = community_builder.build_report(graph, nodes, weight)

        await vector_store.index_community_report(kb_id, report)



    

