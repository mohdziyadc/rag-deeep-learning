from fastapi import APIRouter
import networkx as nx
import logging
from time import perf_counter
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
logger = logging.getLogger(__name__)
llm = LLMClient(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    model=settings.llm_model,
)
embedder = Embedder()
vector_store = GraphDocStore(
    index_name=settings.es_index, embedding_dims=settings.embedding_dims
)
community_builder = CommunityReportBuilder(llm=llm)


@router.post("/graphrag/build")
async def build_graph(payload: dict):
    build_started_at = perf_counter()
    kb_id = payload.get("kb_id") or "kb_demo"
    chunks = payload["chunks"]
    # this is stored at a KB level in ragflow's DB. If not present, defaulted to a set of entity types
    entity_types = payload.get(
        "entity_types", ["organization", "person", "geo", "event", "category"]
    )
    reset = payload.get("reset", False)
    logger.info(
        "Build started for kb_id=%s chunks=%d reset=%s", kb_id, len(chunks), reset
    )
    logger.info("Entity types: %s", ", ".join(entity_types))

    extractor = GraphExtractor(llm, entity_types)
    base = nx.Graph() if reset else await vector_store.load_graph(kb_id)
    logger.info(
        "Base graph loaded nodes=%d edges=%d",
        base.number_of_nodes(),
        base.number_of_edges(),
    )

    if reset:
        logger.info("Reset enabled; deleting KB graph docs for kb_id=%s", kb_id)
        await vector_store.delete_kb_docs(kb_id=kb_id)

    logger.info("Merging chunk subgraphs")
    merge_started_at = perf_counter()
    base = await merge_chunks(kb_id, chunks, extractor, base)
    logger.info("merge_chunks completed in %.2fs", perf_counter() - merge_started_at)
    logger.info(
        "Post-merge graph nodes=%d edges=%d",
        base.number_of_nodes(),
        base.number_of_edges(),
    )

    logger.info("Running entity resolution")
    resolve_started_at = perf_counter()
    base = resolve_entities(base)
    logger.info(
        "resolve_entities completed in %.2fs", perf_counter() - resolve_started_at
    )
    logger.info(
        "Post-resolution graph nodes=%d edges=%d",
        base.number_of_nodes(),
        base.number_of_edges(),
    )

    logger.info("Applying pagerank")
    pagerank_started_at = perf_counter()
    apply_pagerank(base)
    logger.info(
        "apply_pagerank completed in %.2fs", perf_counter() - pagerank_started_at
    )

    logger.info("Indexing graph snapshot")
    snapshot_started_at = perf_counter()
    await index_graph_snapshot(kb_id, base)
    logger.info(
        "index_graph_snapshot completed in %.2fs", perf_counter() - snapshot_started_at
    )
    logger.info("Indexing entity docs")
    entities_started_at = perf_counter()
    await index_entities(kb_id, base)
    logger.info(
        "index_entities completed in %.2fs", perf_counter() - entities_started_at
    )
    logger.info("Indexing relation docs")
    relations_started_at = perf_counter()
    await index_relations(kb_id, base)
    logger.info(
        "index_relations completed in %.2fs", perf_counter() - relations_started_at
    )
    logger.info("Indexing community reports")
    communities_started_at = perf_counter()
    await index_community_reports(kb_id, base)
    logger.info(
        "index_community_reports completed in %.2fs",
        perf_counter() - communities_started_at,
    )

    logger.info(
        "Build completed for kb_id=%s nodes=%d edges=%d",
        kb_id,
        base.number_of_nodes(),
        base.number_of_edges(),
    )
    logger.info(
        "Total build pipeline time for kb_id=%s: %.2fs",
        kb_id,
        perf_counter() - build_started_at,
    )

    return {
        "nodes": base.number_of_nodes(),
        "edges": base.number_of_edges(),
        "kb_id": kb_id,
    }


async def merge_chunks(
    kb_id: str, chunks: list[dict], extractor: GraphExtractor, base: nx.Graph
) -> nx.Graph:
    logger.info("Starting merge of %d chunks", len(chunks))
    for idx, chunk in enumerate(chunks, start=1):
        doc_id = chunk.get("doc_id", "doc_unknown")
        logger.info(
            "[%d/%d] Extracting graph tuples for doc_id=%s", idx, len(chunks), doc_id
        )
        result = await extractor.extract(chunk["content"])
        logger.info(
            "[%d/%d] Extracted entities=%d relations=%d",
            idx,
            len(chunks),
            len(result.entities),
            len(result.relations),
        )
        subgraph = build_subgraph(doc_id, result)
        await vector_store.upsert_subgraph(kb_id, doc_id, subgraph)
        logger.info(
            "[%d/%d] Subgraph upserted for doc_id=%s nodes=%d edges=%d",
            idx,
            len(chunks),
            doc_id,
            subgraph.number_of_nodes(),
            subgraph.number_of_edges(),
        )
        base = graph_merge(base, subgraph)
    logger.info("Finished merge_chunks for kb_id=%s", kb_id)
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
    logger.debug(
        "Built subgraph doc_id=%s nodes=%d edges=%d",
        doc_id,
        subgraph.number_of_nodes(),
        subgraph.number_of_edges(),
    )
    return subgraph


def apply_pagerank(graph: nx.Graph) -> None:
    pr = nx.pagerank(graph) if len(graph) else {}

    for node, score in pr.items():
        graph.nodes[node]["pagerank"] = score
    logger.info("Pagerank assigned to %d nodes", len(pr))


async def index_graph_snapshot(kb_id: str, graph: nx.Graph) -> None:
    logger.info("Refreshing graph snapshot docs for kb_id=%s", kb_id)
    await vector_store.delete_graph_docs(kb_id)
    await vector_store.upsert_graph(kb_id, graph)


async def index_entities(kb_id: str, graph: nx.Graph) -> None:
    entity_names = list(graph.nodes())
    entity_texts = [graph.nodes[n].get("description", "") for n in entity_names]
    entity_vecs = embedder.embed(entity_texts) if entity_texts else []
    logger.info("Indexing %d entities for kb_id=%s", len(entity_names), kb_id)

    for name, vec in zip(entity_names, entity_vecs):
        ent = graph.nodes[name]
        await vector_store.index_entity(
            kb_id,
            {
                "entity_name": name,
                "entity_type": ent.get("entity_type", "-"),
                "description": ent.get("description", ""),
                "pagerank": ent.get("pagerank", 0),
            },
            vec.tolist(),
        )
    logger.info("Entity indexing complete for kb_id=%s", kb_id)


async def index_relations(kb_id: str, graph: nx.Graph) -> None:
    relations = list(graph.edges(data=True))
    rel_texts = [r[2].get("description", "") for r in relations]
    rel_vecs = embedder.embed(rel_texts) if rel_texts else []
    logger.info("Indexing %d relations for kb_id=%s", len(relations), kb_id)

    for (src, tgt, data), vec in zip(relations, rel_vecs):
        await vector_store.index_relation(
            kb_id,
            {
                "src_id": src,
                "tgt_id": tgt,
                "description": data.get("description", ""),
                "strength": data.get("weight", 0),
            },
            vec.tolist(),
        )
    logger.info("Relation indexing complete for kb_id=%s", kb_id)


async def index_community_reports(kb_id: str, graph: nx.Graph) -> None:
    communities = community_builder.build_communities(graph)
    logger.info("Detected %d communities for kb_id=%s", len(communities), kb_id)
    for _, nodes in communities.items():
        weight = len(nodes) / max(1, graph.number_of_nodes())
        report = await community_builder.build_report(graph, nodes, weight)

        await vector_store.index_community_report(kb_id, report)
    logger.info("Community report indexing complete for kb_id=%s", kb_id)
