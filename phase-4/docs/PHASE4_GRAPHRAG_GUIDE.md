# Phase 4: Knowledge Graph (GraphRAG) — RAGFlow-Faithful

## Goal

Build a **GraphRAG layer** that extracts entities and relations from your parsed chunks, merges them into a global graph, and uses that graph at query time to retrieve high-level, structured context. You will:

- Extract entities and relationships with an LLM prompt (GraphExtractor).
- Build per-document subgraphs and merge them into a knowledge graph.
- Resolve entity duplicates (Entity Resolution).
- Generate community reports that summarize clusters.
- Query the graph at inference time to provide graph-derived context.
- Expose this as a FastAPI app with end-to-end endpoints.

No toy snippets, no pseudo-code. This guide mirrors RAGFlow’s production flow and explains every “why” and “where in code”.

## Where Phase 4 Fits (Pipeline Placement)

Phase 4 runs in **two places**:

1. **Graph build (offline/async)** — after parsing, chunking, and indexing.
2. **Graph retrieval (online)** — during retrieval, in parallel with BM25/vector.

```
parser → chunker → indexing → GraphRAG build (Phase 4, offline)
query  → chunk retrieval (Phase 1) + graph retrieval (Phase 4) → generation (Phase 3)
```

Diagram view:

```
Parsed Chunks (parser + chunker output)
        │
        ▼
    Indexing
        │
        ▼
GraphRAG Build (Phase 4, offline)
        │
        ▼
Global Graph + Graph Index

Query
  │
  ├─▶ Chunk Retrieval (BM25 + Vector, Phase 1)
  │
  └─▶ Graph Retrieval (Phase 4)
                │
                ▼
           Combined Evidence
                │
                ▼
            LLM Answer
```

---

## High-Level Design (GraphRAG)

```
Parsed Chunks
   │
   ├─▶ Graph Build (offline)
   │      ├─ Extract entities + relations (LLM)
   │      ├─ Build per-doc subgraph
   │      ├─ Merge into global graph
   │      ├─ Entity resolution
   │      └─ Community reports
   │
   └─▶ Graph Index (ES)
          ├─ graph (global)
          ├─ subgraph (per doc)
          ├─ entity (vector)
          ├─ relation (vector)
          └─ community_report

Query
   │
   ├─▶ Chunk Retrieval (BM25 + vector)
   └─▶ Graph Retrieval (entity/rel/community)
                │
                ▼
           Combined Evidence
                │
                ▼
             LLM Answer
```

## Low-Level Design (Data + Control Flow)

```
POST /api/graphrag/build
  ├─ load chunks[]
  ├─ for each chunk:
  │    ├─ GraphExtractor.extract(text) -> entities, relations
  │    ├─ build subgraph (NetworkX)
  │    ├─ upsert subgraph into ES (knowledge_graph_kwd=subgraph)
  │    └─ merge into global graph (graph_merge)
  ├─ resolve_entities(global graph)
  ├─ pagerank update
  ├─ upsert global graph into ES (knowledge_graph_kwd=graph)
  ├─ embed entities + relations
  └─ index entity/relation vectors into ES

POST /api/graphrag/query
  ├─ embed query
  ├─ vector search entities (ES)
  ├─ vector search relations (ES)
  ├─ fetch community_report by entities
  └─ format graph_context block
```

---

## Why is the knowledge graph needed ?

You maintain the knowledge graph because it gives the generation layer structured, global context that chunk retrieval alone can’t reliably provide. It impacts output in three concrete ways:

1. Higher‑level synthesis  
   Chunk retrieval is local and snippet‑based. A graph captures cross‑document relationships (entity → policy → SLA → compliance), so the generator can produce coherent, policy‑level answers instead of stitching isolated sentences.
2. Better grounding + less hallucination  
   Graph retrieval surfaces entities and relationships that are explicitly connected in your corpus. When the LLM sees “Acme → SOC 2 Type II → audit frequency,” it’s less likely to invent missing links.
3. Structured evidence injection  
   The graph provides a compact, structured context block (entities, relations, community summaries). This reduces token waste and gives the generator a “map” of the knowledge space, improving answer consistency and coverage.
   In practice: the graph doesn’t replace chunk retrieval; it augments it. The generator gets:

- Local evidence (chunks with citations)
- Global structure (graph context)
  That combination yields answers that are both precise and well‑connected across documents.

## Table of Contents

1. The Big Picture
2. How RAGFlow Does It (Where To Look)
3. Phase 4 Architecture (rag-deep-learning/phase-4)
4. Data Prerequisites (Standalone)
5. Step-by-Step Implementation
6. Query-Time Graph Retrieval Flow
7. Testing Checklist
8. Phase 4 Outcomes (Ready for Agentic RFP)

---

## 1. The Big Picture

GraphRAG adds a second retrieval channel: **knowledge graph context**.
RAGFlow uses **parallel retrieval**:

- **Chunk retrieval** (BM25 + vector) from Phase 1
- **Graph retrieval** (entities/relations/communities) from Phase 4

Both are fed into the generator as evidence.

```
Parsed Chunks (parser + chunker output)
        │
        ▼
┌─────────────┐  ┌──────────────┐  ┌───────────────┐
│ Entity/Rel  │▶│ Subgraph per  │▶│ Merge + Clean  │
│ Extraction  │  │ Document      │  │ Global Graph  │
└─────────────┘  └──────────────┘  └───────────────┘
                                                │
                                                ▼
                                      ┌─────────────────┐
                                      │ Entity Resolve  │
                                      │ + Community     │
                                      │ Reports         │
                                      └─────────────────┘

Query
  │
  ├─▶ Chunk Retrieval (BM25 + Vector, Phase 1)
  │
  └─▶ Graph Retrieval (entities/relations/community, Phase 4)
                │
                ▼
           Combined Evidence
                │
                ▼
            LLM Answer
```

Why it matters:

- Chunk retrieval (BM25/vector over indexed chunks) captures **local evidence**.
- GraphRAG adds **global structure** (entities, relationships, communities).
- For RFPs, this enables “policy-level” answers and cross-document reasoning.

---

## 2. How RAGFlow Does It (Where To Look)

These are the exact files to study in RAGFlow:

- **Graph extraction**:
  - `graphrag/general/graph_extractor.py`
  - `graphrag/general/graph_prompt.py`
- **Graph indexing & merge**:
  - `graphrag/general/index.py`
- **Entity resolution**:
  - `graphrag/entity_resolution.py`
  - `graphrag/entity_resolution_prompt.py`
- **Community reports**:
  - `graphrag/general/community_reports_extractor.py`
  - `graphrag/general/community_report_prompt.py`
- **Graph retrieval at query time**:
  - `graphrag/search.py`
- **Chunk retrieval (BM25 + vector)**:
  - `rag/nlp/search.py`
  - `api/db/services/dialog_service.py` (retrieval + generation orchestration)

Parallel retrieval happens by **running both retrieval paths and merging
their outputs into a single prompt** before calling the generator.

This guide mirrors these components in a smaller, learnable system.

---

## 3. Phase 4 Architecture (rag-deep-learning/phase-4)

```
rag-deep-learning/phase-4/
├── app/
│   ├── main.py
│   ├── config.py
│   └── api/
│       └── routes/
│           ├── graphrag.py
│           └── query.py
├── core/
│   ├── graph/
│   │   ├── extractor.py
│   │   ├── prompts.py
│   │   ├── merge.py
│   │   ├── resolve.py
│   │   ├── community.py
│   │   ├── store.py
│   │   └── search.py
│   ├── llm/
│   │   └── client.py
│   └── retrieval/
│       └── chunk_bridge.py
├── models/
│   └── schemas.py
└── tests/
    └── test_graphrag.py
```

Design choices (aligned with your learning style):

- **Explicit file separation** to match RAGFlow’s modules.
- **GraphStore** uses Elasticsearch, mirroring RAGFlow's doc store.
- **Prompts** mirror RAGFlow’s tuple/record delimiter format.

---

## 4. Data Prerequisites (Standalone)

You can implement Phase 4 without Phase 1 or Phase 2 by using the
pre-built data artifacts in `opencode/data/phase-4`:

- `opencode/data/phase-4/chunks.json`: mock parsed chunks
- `opencode/data/phase-4/entity_types.json`: entity type list
- `opencode/data/phase-4/build_request.json`: payload for `/api/graphrag/build`
- `opencode/data/phase-4/query_request.json`: payload for `/api/graphrag/query`

How to use these artifacts:

1. Read `chunks.json` and pass chunk objects (doc_id, content, metadata) with `kb_id` to the build endpoint.
2. Use `entity_types.json` for the graph extraction prompt.
3. Run `query_request.json` against the query endpoint and inspect
   the returned `graph_context`.

These files are complete enough to build a graph and test query-time
GraphRAG with Elasticsearch as the graph store.

---

## 5. Step-by-Step Implementation

### Step 0: Dependencies

Add to `rag-deep-learning/phase-4/pyproject.toml`:

```toml
[project]
dependencies = [
    # API
    "fastapi>=0.128.0",
    "uvicorn[standard]>=0.40.0",
    "pydantic>=2.12.5",

    # LLM + token counting
    "openai>=1.50.0",
    "tiktoken>=0.7.0",

    # Elasticsearch (Graph store)
    "elasticsearch>=8.12.0",

    # Settings
    "pydantic-settings>=2.10.0",

    # Graph + community detection
    "networkx>=3.3",
    "python-louvain>=0.16",  # community detection (Leiden substitute)

    # Utils
    "numpy>=1.26.0",
    "pandas>=2.2.0",
]
```

Why this matches RAGFlow:

- `networkx` is used in RAGFlow for graph representation.
- RAGFlow uses Leiden; Louvain is a close proxy for learning.

Local Elasticsearch (for learning):

```
docker run --rm -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.12.2
```

---

### Step 1: Data Models

Create `models/schemas.py`:

```python
from pydantic import BaseModel, Field


class GraphEntity(BaseModel):
    entity_name: str
    entity_type: str
    description: str
    source_id: list[str] = Field(default_factory=list)


class GraphRelation(BaseModel):
    src_id: str
    tgt_id: str
    description: str
    strength: float
    source_id: list[str] = Field(default_factory=list)


class GraphExtractionResult(BaseModel):
    entities: list[GraphEntity]
    relations: list[GraphRelation]


class CommunityReport(BaseModel):
    title: str
    summary: str
    findings: list[dict]
    weight: float
    entities: list[str]


class GraphQuery(BaseModel):
    question: str
    kb_id: str | None = None
    top_entities: int = 6
    top_relations: int = 6
    top_communities: int = 1
```

Why this matters:

- RAGFlow uses structured entity/relation objects and stores them into ES.
- These models keep the graph layer explicit and inspectable.

---

### Step 2: Graph Extraction Prompt (RAGFlow Format)

Create `core/graph/prompts.py`:

```python
GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract:
  - entity_name (capitalized, in language of text)
  - entity_type (one of: {entity_types})
  - entity_description (comprehensive)
  Format each as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Identify all related pairs. For each pair, extract:
  - source_entity, target_entity
  - relationship_description
  - relationship_strength (numeric)
  Format each as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output as a single list separated by {record_delimiter}
4. When finished, output {completion_delimiter}
"""
```

Why this matches RAGFlow:

- This is a direct adaptation of `graphrag/general/graph_prompt.py`.
- The tuple/record delimiters make parsing reliable and deterministic.

---

### Step 3: LLM Client

Create `core/llm/client.py`:

```python
from openai import OpenAI


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
```

---

### Step 3.1: Embedding Client (Entity/Relation Vectors)

Create `core/graph/embeddings.py`:

```python
from openai import OpenAI


class EmbeddingClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]
```

---

### Step 4: Graph Extractor (Entities + Relations)

Create `core/graph/extractor.py`:

```python
import re
from core.graph.prompts import GRAPH_EXTRACTION_PROMPT
from core.llm.client import LLMClient
from models.schemas import GraphEntity, GraphRelation, GraphExtractionResult


DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"


class GraphExtractor:
    def __init__(self, llm: LLMClient, entity_types: list[str]) -> None:
        self.llm = llm
        self.entity_types = entity_types

    def extract(self, text: str) -> GraphExtractionResult:
        """
        Extract entities + relations from a chunk. Mirrors
        graphrag/general/graph_extractor.py parsing flow.
        """
        system = GRAPH_EXTRACTION_PROMPT.format(
            entity_types=", ".join(self.entity_types),
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
        )
        raw = self.llm.chat(system=system, user=text)
        records = raw.split(DEFAULT_RECORD_DELIMITER)

        entities: list[GraphEntity] = []
        relations: list[GraphRelation] = []

        for record in records:
            match = re.search(r"\((.*)\)", record)
            if not match:
                continue
            parts = match.group(1).split(DEFAULT_TUPLE_DELIMITER)
            if not parts:
                continue
            kind = parts[0].strip().strip('"')
            if kind == "entity" and len(parts) >= 4:
                entities.append(GraphEntity(
                    entity_name=parts[1].strip('" '),
                    entity_type=parts[2].strip('" '),
                    description=parts[3].strip('" '),
                ))
            if kind == "relationship" and len(parts) >= 5:
                relations.append(GraphRelation(
                    src_id=parts[1].strip('" '),
                    tgt_id=parts[2].strip('" '),
                    description=parts[3].strip('" '),
                    strength=float(parts[4]),
                ))

        return GraphExtractionResult(entities=entities, relations=relations)
```

---

### Step 5: Graph Store (Elasticsearch, RAGFlow-style)

Create `core/graph/store.py`:

```python
import json
from elasticsearch import Elasticsearch
import networkx as nx


class GraphDocStore:
    def __init__(self, url: str, index_name: str, embedding_dims: int) -> None:
        self.client = Elasticsearch(url)
        self.index_name = index_name
        self.embedding_dims = embedding_dims

    def ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            return
        self.client.indices.create(
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
                    "weight_int": {"type": "float"},
                    "weight_flt": {"type": "float"},
                    "entity_vec": {"type": "dense_vector", "dims": self.embedding_dims, "index": True, "similarity": "cosine"},
                    "relation_vec": {"type": "dense_vector", "dims": self.embedding_dims, "index": True, "similarity": "cosine"},
                }
            },
        )

    def delete_kb_docs(self, kb_id: str) -> None:
        self.client.delete_by_query(
            index=self.index_name,
            query={"bool": {"filter": [
                {"term": {"kb_id": kb_id}},
                {"terms": {"knowledge_graph_kwd": ["graph", "subgraph", "entity", "relation", "community_report"]}},
            ]}},
        )

    def delete_graph_docs(self, kb_id: str) -> None:
        self.client.delete_by_query(
            index=self.index_name,
            query={"bool": {"filter": [
                {"term": {"kb_id": kb_id}},
                {"terms": {"knowledge_graph_kwd": ["graph", "subgraph"]}},
            ]}},
        )

    def upsert_graph(self, kb_id: str, graph: nx.Graph) -> None:
        doc = {
            "knowledge_graph_kwd": "graph",
            "kb_id": kb_id,
            "source_id": graph.graph.get("source_id", []),
            "content_with_weight": json.dumps(nx.node_link_data(graph, edges="edges"), ensure_ascii=False),
        }
        self.client.index(index=self.index_name, document=doc)

    def upsert_subgraph(self, kb_id: str, doc_id: str, subgraph: nx.Graph) -> None:
        doc = {
            "knowledge_graph_kwd": "subgraph",
            "kb_id": kb_id,
            "source_id": [doc_id],
            "doc_id": doc_id,
            "content_with_weight": json.dumps(nx.node_link_data(subgraph, edges="edges"), ensure_ascii=False),
        }
        self.client.index(index=self.index_name, document=doc)

    def load_graph(self, kb_id: str) -> nx.Graph:
        res = self.client.search(
            index=self.index_name,
            size=1,
            query={"bool": {"filter": [
                {"term": {"kb_id": kb_id}},
                {"term": {"knowledge_graph_kwd": "graph"}},
            ]}},
        )
        hits = res.get("hits", {}).get("hits", [])
        if not hits:
            return nx.Graph()
        raw = hits[0]["_source"]["content_with_weight"]
        return nx.node_link_graph(json.loads(raw), edges="edges")

    def index_entity(self, kb_id: str, entity: dict, vec: list[float]) -> None:
        doc = {
            "knowledge_graph_kwd": "entity",
            "kb_id": kb_id,
            "entity_kwd": entity["entity_name"],
            "entity_type_kwd": entity["entity_type"],
            "content_with_weight": json.dumps({"description": entity["description"]}),
            "rank_flt": entity.get("pagerank", 0),
            "entity_vec": vec,
        }
        self.client.index(index=self.index_name, document=doc)

    def index_relation(self, kb_id: str, relation: dict, vec: list[float]) -> None:
        doc = {
            "knowledge_graph_kwd": "relation",
            "kb_id": kb_id,
            "from_entity_kwd": relation["src_id"],
            "to_entity_kwd": relation["tgt_id"],
            "content_with_weight": json.dumps({"description": relation["description"]}),
            "weight_int": relation.get("strength", 0),
            "relation_vec": vec,
        }
        self.client.index(index=self.index_name, document=doc)

    def index_community_report(self, kb_id: str, report: dict) -> None:
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
        self.client.index(index=self.index_name, document=doc)

    def search_entities(self, kb_id: str, vec: list[float], top_k: int) -> list[dict]:
        res = self.client.search(
            index=self.index_name,
            size=top_k,
            query={"bool": {"filter": [
                {"term": {"kb_id": kb_id}},
                {"term": {"knowledge_graph_kwd": "entity"}},
            ]}},
            knn={"field": "entity_vec", "query_vector": vec, "k": top_k, "num_candidates": max(50, top_k * 5)},
        )
        return [h["_source"] for h in res.get("hits", {}).get("hits", [])]

    def search_relations(self, kb_id: str, vec: list[float], top_k: int) -> list[dict]:
        res = self.client.search(
            index=self.index_name,
            size=top_k,
            query={"bool": {"filter": [
                {"term": {"kb_id": kb_id}},
                {"term": {"knowledge_graph_kwd": "relation"}},
            ]}},
            knn={"field": "relation_vec", "query_vector": vec, "k": top_k, "num_candidates": max(50, top_k * 5)},
        )
        return [h["_source"] for h in res.get("hits", {}).get("hits", [])]

    def search_community_reports(self, kb_id: str, entities: list[str], top_k: int) -> list[dict]:
        res = self.client.search(
            index=self.index_name,
            size=top_k,
            query={"bool": {"filter": [
                {"term": {"kb_id": kb_id}},
                {"term": {"knowledge_graph_kwd": "community_report"}},
                {"terms": {"entities_kwd": entities}},
            ]}},
            sort=[{"weight_flt": "desc"}],
        )
        return [h["_source"] for h in res.get("hits", {}).get("hits", [])]
```

Why this matches RAGFlow:

- RAGFlow stores `graph`, `subgraph`, `entity`, `relation`, and `community_report` docs in ES.
- The `knowledge_graph_kwd` field mirrors RAGFlow's routing mechanism.

---

### Step 6: Merge Subgraphs

Create `core/graph/merge.py`:

```python
import networkx as nx


GRAPH_FIELD_SEP = "<SEP>"


def graph_merge(base: nx.Graph, subgraph: nx.Graph) -> nx.Graph:
    """
    Merge subgraph into the global graph with description/source_id
    accumulation, mirroring graphrag/utils.py::graph_merge.
    """
    for node_name, attr in subgraph.nodes(data=True):
        if not base.has_node(node_name):
            base.add_node(node_name, **attr)
            continue
        node = base.nodes[node_name]
        node["description"] += GRAPH_FIELD_SEP + attr["description"]
        node["source_id"] += attr["source_id"]

    for source, target, attr in subgraph.edges(data=True):
        edge = base.get_edge_data(source, target)
        if edge is None:
            base.add_edge(source, target, **attr)
            continue
        edge["weight"] = edge.get("weight", 0) + attr.get("weight", 0)
        edge["description"] += GRAPH_FIELD_SEP + attr["description"]
        edge["source_id"] += attr["source_id"]

    for node_degree in base.degree:
        base.nodes[str(node_degree[0])]["rank"] = int(node_degree[1])
    base.graph.setdefault("source_id", [])
    base.graph["source_id"] += subgraph.graph.get("source_id", [])
    return base
```

---

### Step 7: Entity Resolution (De-dup)

Create `core/graph/resolve.py`:

```python
import re
import networkx as nx


def should_merge(a: str, b: str) -> bool:
    """
    Lightweight similarity heuristic (RAGFlow uses LLM + editdistance).
    For learning, this handles basic near-duplicates.
    """
    a_norm = re.sub(r"\W+", "", a.lower())
    b_norm = re.sub(r"\W+", "", b.lower())
    if a_norm == b_norm:
        return True
    if a_norm in b_norm or b_norm in a_norm:
        return True
    return False


def resolve_entities(graph: nx.Graph) -> nx.Graph:
    """
    Merge nodes that represent the same entity.
    Mirrors graphrag/entity_resolution.py in simplified form.
    """
    nodes = list(graph.nodes())
    merged = graph.copy()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[i] not in merged or nodes[j] not in merged:
                continue
            if should_merge(nodes[i], nodes[j]):
                nx.contracted_nodes(merged, nodes[i], nodes[j], self_loops=False, copy=False)
    return merged
```

Tradeoff note:

- RAGFlow uses LLM prompts + edit distance for better semantic resolution.
- You can later swap this to the LLM-based resolution prompt.

---

### Step 8: Community Reports

Create `core/graph/community.py`:

```python
import networkx as nx
import community as community_louvain


def build_communities(graph: nx.Graph) -> dict[int, list[str]]:
    """
    Detect communities (clusters). RAGFlow uses Leiden; Louvain is a close proxy.
    """
    if len(graph) == 0:
        return {}
    partition = community_louvain.best_partition(graph)
    communities: dict[int, list[str]] = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)
    return communities
```

To mirror RAGFlow, convert each community into a report (LLM summary + findings)
and store it in ES using `GraphDocStore.index_community_report(...)`.

---

### Step 9: Graph Query (GraphRAG Retrieval)

Create `core/graph/search.py`:

```python
import json
from core.graph.store import GraphDocStore
from core.graph.embeddings import EmbeddingClient
from models.schemas import GraphQuery


def format_graph_context(entities: list[dict], relations: list[dict], communities: list[dict]) -> str:
    ent_block = ["---- Entities ----"]
    for ent in entities:
        desc = json.loads(ent.get("content_with_weight", "{}") or "{}").get("description", "")
        ent_block.append(f"Entity: {ent.get('entity_kwd', '-')}")
        ent_block.append(f"Type: {ent.get('entity_type_kwd', '-')}")
        ent_block.append(f"Description: {desc}")

    rel_block = ["---- Relations ----"]
    for rel in relations:
        desc = json.loads(rel.get("content_with_weight", "{}") or "{}").get("description", "")
        rel_block.append(f"From: {rel.get('from_entity_kwd', '-')}")
        rel_block.append(f"To: {rel.get('to_entity_kwd', '-')}")
        rel_block.append(f"Description: {desc}")

    comm_block = ["---- Community Reports ----"]
    for rep in communities:
        obj = json.loads(rep.get("content_with_weight", "{}") or "{}")
        comm_block.append(obj.get("report", ""))
        comm_block.append(obj.get("evidences", ""))

    return "\n".join(ent_block + [""] + rel_block + [""] + comm_block)


class GraphSearcher:
    def __init__(self, store: GraphDocStore, embedder: EmbeddingClient) -> None:
        self.store = store
        self.embedder = embedder

    def query(self, kb_id: str, q: GraphQuery) -> str:
        """
        Retrieve graph context from ES using vector search over
        entity/relation embeddings, mirroring graphrag/search.py.
        """
        vec = self.embedder.embed([q.question])[0]
        entities = self.store.search_entities(kb_id, vec, q.top_entities)
        relations = self.store.search_relations(kb_id, vec, q.top_relations)
        entity_names = [e.get("entity_kwd", "") for e in entities if e.get("entity_kwd")]
        communities = self.store.search_community_reports(kb_id, entity_names, q.top_communities)
        return format_graph_context(entities, relations, communities)
```

---

### Step 10: FastAPI Endpoints

Create `app/api/routes/graphrag.py`:

```python
from fastapi import APIRouter
import networkx as nx
from app.config import get_settings
from core.llm.client import LLMClient
from core.graph.embeddings import EmbeddingClient
from core.graph.extractor import GraphExtractor
from core.graph.merge import graph_merge
from core.graph.resolve import resolve_entities
from core.graph.store import GraphDocStore


router = APIRouter()
settings = get_settings()
llm = LLMClient(settings.llm_api_key, settings.llm_base_url, settings.llm_model)
embedder = EmbeddingClient(settings.llm_api_key, settings.llm_base_url, settings.embedding_model)
store = GraphDocStore(settings.es_url, settings.es_index, settings.embedding_dims)
store.ensure_index()


@router.post("/graphrag/build")
async def build_graph(payload: dict):
    kb_id = payload.get("kb_id") or settings.default_kb_id
    chunks = payload["chunks"]
    entity_types = payload.get("entity_types", ["organization", "product", "policy", "contract"])
    reset = payload.get("reset", False)

    extractor = GraphExtractor(llm, entity_types)
    base = nx.Graph() if reset else store.load_graph(kb_id)
    if reset:
        store.delete_kb_docs(kb_id)

    for chunk in chunks:
        doc_id = chunk.get("doc_id", "doc_unknown")
        result = extractor.extract(chunk["content"])

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

        store.upsert_subgraph(kb_id, doc_id, subgraph)
        base = graph_merge(base, subgraph)

    base = resolve_entities(base)
    pr = nx.pagerank(base) if len(base) else {}
    for node, score in pr.items():
        base.nodes[node]["pagerank"] = score

    store.delete_graph_docs(kb_id)
    store.upsert_graph(kb_id, base)

    entity_names = list(base.nodes())
    entity_texts = [base.nodes[n].get("description", "") for n in entity_names]
    entity_vecs = embedder.embed(entity_texts) if entity_texts else []
    for name, vec in zip(entity_names, entity_vecs):
        ent = base.nodes[name]
        store.index_entity(kb_id, {
            "entity_name": name,
            "entity_type": ent.get("entity_type", "-"),
            "description": ent.get("description", ""),
            "pagerank": ent.get("pagerank", 0),
        }, vec)

    relations = list(base.edges(data=True))
    rel_texts = [r[2].get("description", "") for r in relations]
    rel_vecs = embedder.embed(rel_texts) if rel_texts else []
    for (src, tgt, data), vec in zip(relations, rel_vecs):
        store.index_relation(kb_id, {
            "src_id": src,
            "tgt_id": tgt,
            "description": data.get("description", ""),
            "strength": data.get("weight", 0),
        }, vec)

    return {"nodes": base.number_of_nodes(), "edges": base.number_of_edges(), "kb_id": kb_id}
```

Create `app/api/routes/query.py`:

```python
from fastapi import APIRouter
from core.graph.store import GraphDocStore
from core.graph.embeddings import EmbeddingClient
from core.graph.search import GraphSearcher
from models.schemas import GraphQuery
from app.config import get_settings


router = APIRouter()
settings = get_settings()
store = GraphDocStore(settings.es_url, settings.es_index, settings.embedding_dims)
searcher = GraphSearcher(store, EmbeddingClient(settings.llm_api_key, settings.llm_base_url, settings.embedding_model))


@router.post("/graphrag/query")
async def graphrag_query(q: GraphQuery):
    kb_id = q.kb_id or settings.default_kb_id
    return {"graph_context": searcher.query(kb_id, q)}
```

Create `app/config.py`:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_api_key: str
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    es_url: str = "http://localhost:9200"
    es_index: str = "graphrag_demo"
    default_kb_id: str = "kb_demo"

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()
```

Create `app/main.py`:

```python
from fastapi import FastAPI
from app.api.routes import graphrag, query


app = FastAPI(title="Phase 4 GraphRAG")
app.include_router(graphrag.router, prefix="/api", tags=["graphrag"])
app.include_router(query.router, prefix="/api", tags=["graphrag"])
```

---

## Elasticsearch Inspection (Graph Docs)

After running `/api/graphrag/build`, ES will contain multiple document types
distinguished by `knowledge_graph_kwd`. This mirrors RAGFlow’s storage pattern.

Inspect counts by type:

```bash
curl -s "http://localhost:9200/graphrag_demo/_count" -H "Content-Type: application/json" -d '{"query":{"term":{"knowledge_graph_kwd":"graph"}}}'
curl -s "http://localhost:9200/graphrag_demo/_count" -H "Content-Type: application/json" -d '{"query":{"term":{"knowledge_graph_kwd":"subgraph"}}}'
curl -s "http://localhost:9200/graphrag_demo/_count" -H "Content-Type: application/json" -d '{"query":{"term":{"knowledge_graph_kwd":"entity"}}}'
curl -s "http://localhost:9200/graphrag_demo/_count" -H "Content-Type: application/json" -d '{"query":{"term":{"knowledge_graph_kwd":"relation"}}}'
curl -s "http://localhost:9200/graphrag_demo/_count" -H "Content-Type: application/json" -d '{"query":{"term":{"knowledge_graph_kwd":"community_report"}}}'
```

Inspect a sample entity doc:

```bash
curl -s "http://localhost:9200/graphrag_demo/_search" -H "Content-Type: application/json" -d '{"size":1,"query":{"term":{"knowledge_graph_kwd":"entity"}}}'
```

Inspect the global graph payload:

```bash
curl -s "http://localhost:9200/graphrag_demo/_search" -H "Content-Type: application/json" -d '{"size":1,"query":{"term":{"knowledge_graph_kwd":"graph"}}}'
```

---

## 6. Query-Time Graph Retrieval Flow

RAGFlow’s `graphrag/search.py` does:

1. Query rewrite (extract entity types + candidate entities).
2. Fetch relevant entities + relations from ES.
3. Pull n-hop expansions and community reports.
4. Package into a “Graph context” chunk.

In your learning implementation:

1. Query `/api/graphrag/query` for graph context.
2. Insert `graph_context` into Phase 3 prompt as another knowledge block.
3. This gives the LLM global, structured context.

---

## 7. Testing Checklist

- Build a graph from 2–3 chunks; verify nodes and edges appear.
- Merge multiple chunk batches; ensure pagerank updates.
- Confirm entity resolution merges near-duplicates.
- Query the graph and inspect the returned entity/rel block.
- Inject `graph_context` into Phase 3 and test answer quality.

---

## 8. Phase 4 Outcomes (Ready for Agentic RFP)

After this phase, you will understand:

1. How GraphRAG converts chunks into a global graph.
2. How entity resolution prevents duplicate nodes.
3. How community detection generates higher-level summaries.
4. How graph retrieval becomes a second evidence channel for answers.
5. How to wire graph context into your Phase 3 generator.

This completes Phase 4 and prepares you for agentic RFP workflows (multi-step reasoning + tool use).

## Footnotes

Gotchas / workarounds vs RAGFlow:

- Entity resolution: guide uses a lightweight heuristic; RAGFlow uses LLM + editdistance (graphrag/entity_resolution.py).
- Community reports: guide uses Louvain clustering; RAGFlow uses Leiden + LLM summaries (graphrag/general/community_reports_extractor.py).
- Query rewrite: guide skips RAGFlow’s query rewrite step (graphrag/search.py::query_rewrite).
- N-hop expansion: guide doesn’t add n-hop entity path logic used in RAGFlow.
- Scoring blend: guide uses pure vector search; RAGFlow blends pagerank + sim + type hints.
- Graph change tracking: guide doesn’t mirror GraphChange or set_graph deletion semantics fully.
- Caching + concurrency: no Redis cache, no chat limiter, no task cancellation hooks.
- Doc store coupling: guide uses a direct ES client, not RAGFlow’s doc store abstraction.
