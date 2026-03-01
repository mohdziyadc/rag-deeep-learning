import logging
from elasticsearch import AsyncElasticsearch
from numpy import indices
from config.config import settings
import networkx as nx
import json

logger = logging.getLogger(__name__)

"""
GraphRAG storage layers:
- Graph: one global snapshot of the full knowledge graph (all nodes + edges), stored as node-link JSON. Preserves full structure for rebuilds.
- Subgraph: per-document snapshot (only nodes/edges derived from a single doc_id). Used for partial rebuilds and doc-level deletion.
- Entity: one doc per node, with name/type/description + embedding. Used for vector retrieval of relevant entities.
- Relation: one doc per edge, with src/tgt/description + embedding + weight. Used for vector retrieval of relevant relations.
- Community report: one doc per detected community (cluster). Stores summary + findings/evidence for higher-level context retrieval.

Example diagram
doc_1 ──> Subgraph(doc_1):  [A]──(r1)──[B]
doc_2 ──> Subgraph(doc_2):  [B]──(r2)──[C]
Global Graph (merge of subgraphs):
  Nodes: A, B, C
  Edges: (A-B), (B-C)
Entity docs:
  entity A, entity B, entity C  (each has embedding)
Relation docs:
  relation A-B, relation B-C    (each has embedding)
Community report:
  Community #1: {A, B, C} → summary + findings

Document: doc_1
  └─ Subgraph(doc_1)
       ├─ Nodes: [A], [B]
       └─ Edges : (A)--(B)
Document: doc_2
  └─ Subgraph(doc_2)
       ├─ Nodes: [B], [C]
       └─ Edges : (B)--(C)
Global Graph (merge of subgraphs)
  ├─ Nodes: [A], [B], [C]
  └─ Edges: (A)--(B), (B)--(C)
"""


class GraphDocStore:

    def __init__(self, index_name: str, embedding_dims: int) -> None:
        self.client: AsyncElasticsearch | None = None
        self.index_name = index_name
        self.embedding_dims = embedding_dims
    

    async def connect(self) -> None:
        if self.client is None:
            self.client = AsyncElasticsearch(
                hosts=[settings.es_host],
                request_timeout=30
            )
            logger.info(f"Connected to ES at {settings.es_host}")

    async def close(self) -> None:
        if self.client:
            await self.client.close()
            self.client = None
    
    async def create_index(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            return

        await self.client.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "knowledge_graph_kwd": {"type": "keyword"},
                    "kb_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "source_id": {"type": "keyword"},
                    "entity_kwd": {"type": "keyword"},
                    "entity_type_kwd": {"type": "keyword"},
                    "from_entity_kwd": {"type": "keyword"},
                    "to_entity_kwd": {"type": "keyword"},
                    "entities_kwd": {"type": "keyword"},
                    "content_with_weight": {"type": "text"},
                    "rank_flt": {"type": "float"},
                    "weight_int": {"type": "integer"},
                    "weight_flt": {"type": "float"},
                    "entity_vec": {"type": "dense_vector", "dims": self.embedding_dims,
                    "index": True, "similarity": "cosine"
                    },
                    "relation_vec": {"type": "dense_vector", "dims": self.embedding_dims,
                    "index": True, "similarity": "cosine"
                    },
                }
            }
        )

    async def delete_kb_docs(self, kb_id: str) -> None:
        # This is just a naive deletion, ragflow's deletion is much more selective
        # reference - set_graph in graphrag/utils.py
        await self.client.delete_by_query(
            index=self.index_name,
            query={
                "bool": {
                    "filter": [
                        {"term": {"kb_id": kb_id}},
                        {"terms": {
                            "knowledge_graph_kwd": ["graph", "subgraph", "entity", "relation", "community_report"]
                        }}
                    ]
                }
            }
        )
    

    async def delete_graph_docs(self, kb_id: str) -> None:
        await self.client.delete_by_query(
            index=self.index_name,
            query={
                "bool": {
                    "filter": [
                        {"term": {"kb_id": kb_id}},
                        {"terms": {"knowledge_graph_kwd": ["graph", "subgraph"]}}
                    ]
                }
            }
        )
    

    async def upsert_graph(self, kb_id: str, graph: nx.Graph) -> None:
        doc = {
            "knowledge_graph_kwd": "graph",
            "kb_id": kb_id,
            "source_id": graph.graph.get("source_id", []),
            "content_with_weight": json.dumps(nx.node_link_data(graph, edges="edges"), ensure_ascii=False) 
        }
        # graph.graph.source_id = is the provenance list for the graph: the set of document IDs
        # whose chunks contributed nodes/edges to this graph.
        await self.client.index(index=self.index_name, document=doc)

    async def upsert_subgraph(self, kb_id: str, doc_id: str, subgraph: nx.Graph) -> None:
        doc = {
            "knowledge_graph_kwd": "subgraph",
            "kb_id": kb_id,
            "source_id": [doc_id],
            "doc_id": doc_id,
            "content_with_weight": json.dumps(nx.node_link_data(subgraph, edges="edges"), ensure_ascii=False)
        }

        await self.client.index(index=self.index_name, document=doc)
    
    async def load_graph(self, kb_id: str) -> nx.Graph:
        res = await self.client.search(
            index=self.index_name,
            size = 1,
            query={
                "bool": {
                    "filter": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"knowledge_graph_kwd": "graph"}}
                    ]
                }
            }
        )

        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            return nx.Graph()
        
        raw = hits[0]["_source"]["content_with_weight"]
        return nx.node_link_graph(json.loads(raw), edges="edges")
    
    async def index_entity(self, kb_id: str, entity: dict, vec: list[float]) -> None:
        doc = {
            "knowledge_graph_kwd": "entity",
            "kb_id": kb_id,
            "entity_kwd": entity["entity_name"],
            "entity_type_kwd": entity["entity_type"],
            "content_with_weight": json.dumps({"description": entity["description"]}),
            "rank_flt": entity.get("pagerank", 0),
            "entity_vec": vec
        }

        await self.client.index(index=self.index_name, document=doc)
    
    async def index_relation(self, kb_id: str, relation: dict, vec: list[float]) -> None:
        doc = {
            "knowledge_graph_kwd": "relation",
            "kb_id": kb_id,
            "from_entity_kwd": relation["src_id"],
            "to_entity_kwd": relation["tgt_id"],
            "content_with_weight": json.dumps({"description": relation["description"]}),
            "weight_int": relation.get("strength", 0),
            "relation_vec": vec,
        }
        await self.client.index(index=self.index_name, document=doc)

    
    async def index_community_report(self, kb_id:str, report: dict) -> None:
        doc = {
            "knowledge_graph_kwd": "community_report",
            "kb_id": kb_id,
            "entities_kwd": report.get("entities", []),
            "weight_flt": report.get("weight", 0),
            "content_with_weight": json.dumps({
                "report": report.get("summary", ""),
                "evidences": "\n".join([f.get("explanation", "") for f in report.get("findings", [])]),
            }),
            
        }
        await self.client.index(index=self.index_name, document=doc)


    async def search_entities(self, kb_id:str, vec:list[float], top_k:int) -> list[dict]:
        res = await self.client.search(
            index= self.index_name,
            size=top_k,
            query={
                "bool": {
                    "filter": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"knowledge_graph_kwd": "entity"}}
                    ]
                }
            },
            knn={"field": "entity_vec", "query_vector": vec, "k": top_k, "num_candidates": max(50, top_k*5)}
        )
        return [h["_source"] for h in res.get("hits", {}).get("hits", [])]

    async def search_relations(self, kb_id:str, vec:list[float], top_k:int) -> list[dict]:
        res = await self.client.search(
            index= self.index_name,
            size=top_k,
            query={
                "bool": {
                    "filter": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"knowledge_graph_kwd": "relation"}}
                    ]
                }
            },
            knn={"field": "relation_vec", "query_vector": vec, "k": top_k, "num_candidates": max(50, top_k*5)}
        )
        return [h["_source"] for h in res.get("hits", {}).get("hits", [])]

    async def search_community_reports(self, kb_id:str, entities:list[str], top_k:int) -> list[dict]:
        res = await self.client.search(
            index= self.index_name,
            size=top_k,
            query={
                "bool": {
                    "filter": [
                        {"term": {"kb_id": kb_id}},
                        {"term": {"knowledge_graph_kwd": "community_report"}},
                        {"terms": {"entities_kwd": entities}}
                    ]
                }
            },
           sort=[{"weight_flt": "desc"}]
        )
        return [h["_source"] for h in res.get("hits", {}).get("hits", [])]
