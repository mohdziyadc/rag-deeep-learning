from pydantic import BaseModel, Field

"""
GraphExtractionResult
├── entities: list[GraphEntity]
│   └── GraphEntity
│       ├── entity_name: str           (e.g., "Acme Corp")
│       ├── entity_type: str           (e.g., "organization")
│       ├── description: str           (e.g., "A SaaS vendor providing...")
│       └── source_id: list[str]       (e.g., ["doc_12", "doc_15"])
└── relations: list[GraphRelation]
    └── GraphRelation
        ├── src_id: str                (e.g., "Acme Corp")
        ├── tgt_id: str                (e.g., "SOC 2")
        ├── description: str           (e.g., "Acme Corp complies with SOC 2")
        ├── strength: float            (e.g., 0.87)
        └── source_id: list[str]       (e.g., ["doc_12"])
CommunityReport
├── title: str                         (e.g., "Security Compliance Cluster")
├── summary: str                       (e.g., "Entities related to audits...")
├── findings: list[dict]               (e.g., [{"summary":"SOC 2 required","explanation":"..."}])
├── weight: float                      (e.g., 0.92)
└── entities: list[str]                (e.g., ["Acme Corp", "SOC 2", "Audit"])
GraphQuery
├── question: str                      (e.g., "How does Acme handle audits?")
├── kb_id: str | None                  (e.g., "kb_demo")
├── top_entities: int                  (e.g., 6)
├── top_relations: int                 (e.g., 6)
└── top_communities: int               (e.g., 1)

"""

class GraphEntity(BaseModel):
    entity_name: str
    entity_type: str
    description: str
    source_id: list[str] = Field(default_factory=list) # optional field cuz we are defaulting to []
    #  When building a subgraph, each entity gets source_id=[doc_id] 
    # so you know which document it came from.
    # So think of it as “where did this node come from?” and it persists through merges.

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
    findings: list[dict] #mandatory - hence not defaulting using Field()
    weight: float
    entities: list[str] # mandatory

class GraphQuery(BaseModel):
    question: str
    kb_id: str | None = None
    top_entities: int = 6
    top_relations: int = 6
    top_communities: int = 1


"""
findings is the structured evidence inside a community report.
It's a list of items the LLM extracts to justify the community summary.
"""



