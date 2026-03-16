# Phase 5: Conversation Memory System (RAGFlow-Faithful)

## Goal

Build an end-to-end memory subsystem that mirrors how RAGFlow stores, extracts, retrieves, and manages long-term conversation memory.

By the end of this phase, you will deeply understand:

- How raw dialogue is persisted immediately.
- How semantic/episodic/procedural memories are extracted asynchronously by an LLM.
- How hybrid retrieval over memory works (BM25 + vector + weighted fusion).
- How memory lifecycle controls work (`status`, `forget_at`, FIFO deletion by size).
- How this plugs into an agentic RAG workflow for RFP/questionnaire answering.

This guide is written from a single-tenant learning view, with explicit notes on multi-tenant behavior exactly where it matters.

---

## Where Phase 5 Fits

Memory sits between online generation and future retrieval:

```
User/Agent Turn -> Save Raw Memory -> Async Extraction -> Store Structured Memories
                                                    |
                                                    v
                                      Future Queries Retrieve Memory
```

In your final RFP system, memory gives your agent continuity across sessions (vendor facts, commitments, prior answers, procedures).

---

## How RAGFlow Implements Phase 5 (Source Map)

These are the key files to study and mirror:

- `memory/services/query.py` - Hybrid query building for memory search.
- `memory/services/messages.py` - Message index CRUD and retrieval.
- `api/db/services/memory_service.py` - Memory metadata CRUD in MySQL (RAGFlow default relational store).
- `api/db/joint_services/memory_message_service.py` - Orchestration: queue, extract, embed, save, retrieve.
- `memory/utils/prompt_util.py` - Prompt assembly for memory extraction.
- `memory/utils/es_conn.py` and `memory/utils/infinity_conn.py` - Message-store backend adapters.
- `api/apps/sdk/memories.py` and `api/apps/sdk/messages.py` - HTTP endpoints.
- `rag/svr/task_executor.py` - Async worker path for memory tasks.

---

## High-Level Architecture (Single-Tenant Mental Model)

```mermaid
flowchart TB
    User[User]
    Agent[Agent]
    API[Messages API]
    MemAPI[Memories API]

    subgraph Orchestration
        MMS[MemoryMessageService]
        MS[MemoryService]
        Worker[Task Executor Worker]
    end

    subgraph Models
        LLM[Chat LLM for Extraction]
        EMB[Embedding Model]
    end

    subgraph Storage
        PG[(PostgreSQL memory table)]
        ES[(Message Store index: memory_tenant_id)]
        Redis[(Redis: id seq + queue + memory size cache)]
        TaskTbl[(PostgreSQL task table)]
    end

    User --> Agent
    Agent --> API
    User --> MemAPI

    MemAPI --> MS
    MS --> PG

    API --> MMS
    MMS --> EMB
    EMB --> ES
    MMS --> Redis
    MMS --> TaskTbl

    Redis --> Worker
    Worker --> MMS
    MMS --> LLM
    LLM --> MMS
    MMS --> EMB
    MMS --> ES
```

### Multi-tenant extension in real RAGFlow

- Relational ownership is scoped by `memory.tenant_id` (MySQL in RAGFlow, PostgreSQL in this guide).
- Message index name is `memory_{tenant_id}`.
- Retrieval can span multiple memory IDs (and tenant indices when permitted).
- LLM/embedding are resolved per tenant via `TenantLLMService` and `LLMBundle`.

---

## Low-Level Diagram 1: Add Message + Async Extraction

```mermaid
sequenceDiagram
    participant C as Client
    participant API as POST /api/v1/messages
    participant MMS as memory_message_service
    participant ES as MessageStore
    participant R as Redis Stream
    participant T as task table
    participant W as task_executor worker
    participant L as LLM extractor
    participant E as Embedding model

    C->>API: add message (memory_id[], user_input, agent_response)
    API->>MMS: queue_save_to_memory_task(...)
    MMS->>E: embed raw message
    E-->>MMS: raw vector
    MMS->>ES: insert RAW message immediately
    MMS->>T: insert memory task (digest=raw_message_id)
    MMS->>R: push task payload
    API-->>C: "All add to task."

    W->>R: consume task
    W->>MMS: handle_save_to_memory_task(...)
    MMS->>L: extract semantic/episodic/procedural JSON
    L-->>MMS: extracted memories
    MMS->>E: embed extracted messages
    E-->>MMS: vectors
    MMS->>ES: insert extracted messages
    MMS->>T: progress=1.0
```

Why RAGFlow does this in two stages:

- RAW memory becomes searchable fast.
- Expensive extraction is offloaded to workers.
- Task progress is observable from UI/API.

---

## Low-Level Diagram 2: Memory Retrieval (Hybrid Search)

```mermaid
sequenceDiagram
    participant Agent as Agent Retrieval Tool
    participant MMS as query_message
    participant Q as MsgTextQuery
    participant V as get_vector
    participant ES as MessageStore.search

    Agent->>MMS: query + memory_id[] + thresholds
    MMS->>Q: build MatchTextExpr (BM25 + synonyms)
    MMS->>V: build MatchDenseExpr (embedding vector)
    MMS->>MMS: build FusionExpr(weighted_sum)
    MMS->>ES: search(condition + [text,dense,fusion])
    ES-->>MMS: ranked messages
    MMS-->>Agent: top_n memory messages
```

Scoring intent:

- Dense similarity captures semantics.
- BM25 catches exact terms and identifiers.
- Fusion balances both for robust RFP memory recall.

---

## Storage Layout You Must Internalize

```mermaid
flowchart LR
    subgraph Relational
        M[(memory table)]
    end

    subgraph MessageStore
        IDX[(memory_tenantId index)]
        RAW[raw messages]
        SEM[semantic messages]
        EPI[episodic messages]
        PRO[procedural messages]
    end

    subgraph CacheQueue
        RID[id_generator:memory]
        MZ[memory_memoryId size cache]
        QS[rag_flow_svr_queue stream]
    end

    M --> IDX
    IDX --> RAW
    IDX --> SEM
    IDX --> EPI
    IDX --> PRO

    RID --> IDX
    MZ --> IDX
    QS --> IDX
```

Important fields in each message document:

- `message_id`: global incremental ID.
- `message_type_kwd`: `raw|semantic|episodic|procedural`.
- `source_id`: points extracted memory back to RAW message.
- `valid_at`, `invalid_at`, `forget_at`: temporal lifecycle.
- `status_int`: soft enable/disable retrieval.
- `content_ltks`: BM25 text.
- `q_*_vec`: dense embedding vector.

---

## Function-By-Function Deep Dive (RAGFlow)

You asked for each function and why it exists. This section is your exact map.

### 1) `memory/services/query.py`

- `get_vector(txt, emb_mdl, topk, similarity)`
  - Encodes query into embedding, picks correct vector field (`q_768_vec`, `q_1024_vec`, etc.), returns `MatchDenseExpr`.
  - Why: enables dense semantic retrieval without hardcoding embedding dimensions.
- `MsgTextQuery.__init__()`
  - Initializes term weighting and synonym dealers.
  - Why: same lexical enrichment strategy as core retrieval.
- `MsgTextQuery.question(txt, tbl="messages", min_match=0.6)`
  - Normalizes text, language-branches (Chinese vs non-Chinese), builds weighted BM25 query + synonyms + phrase boosts, returns `MatchTextExpr` and keywords.
  - Why: lexical recall robustness and multilingual support.
- Nested `need_fine_grained_tokenize(tk)`
  - Decides when to do aggressive Chinese token expansion.
  - Why: better token-level recall for CJK text.

### 2) `memory/services/messages.py`

- `index_name(uid)`
  - Builds tenant-scoped message index name.
  - Why: physical tenant isolation in message storage.
- `has_index/create_index/delete_index(...)`
  - Lifecycle for message index/table.
  - Why: lazy provisioning and cleanup.
- `insert_message(...)`
  - Applies canonical `id`, normalizes `status`, bulk inserts messages.
  - Why: deterministic document ID and indexing consistency.
- `update_message/delete_message(...)`
  - Updates or removes by conditions.
  - Why: status toggles, forgetting, cleanup.
- `list_message(...)`
  - Lists RAW messages, then joins extracted children by `source_id`.
  - Why: UI wants dialogue row + extracted memory children.
- `get_recent_messages(...)`
  - Time-desc fetch by agent/session.
  - Why: short-term context recall.
- `search_message(...)`
  - Applies condition + match expressions + top-N.
  - Why: shared entrypoint for hybrid memory retrieval.
- `calculate_message_size(...)`
  - Approximates memory footprint using text + embedding vector.
  - Why: enforce memory capacity.
- `calculate_memory_size(...)`
  - Recomputes aggregate size by memory ID.
  - Why: cache warmup/backfill.
- `pick_messages_to_delete_by_fifo(...)`
  - Chooses deletion candidates: forgotten first, then oldest `valid_at`.
  - Why: deterministic eviction policy.
- `get_by_message_id(...)`
  - Direct read by composite doc ID.
  - Why: inspect full message content/vector.
- `get_max_message_id(...)`
  - Finds largest existing `message_id`.
  - Why: initialize Redis ID sequence safely.

### 3) `api/db/services/memory_service.py`

- `get_by_memory_id/get_by_tenant_id/get_all_memory(...)`
  - Base metadata fetchers.
  - Why: routing and initialization.
- `get_with_owner_name_by_id(...)`
  - Joins owner nickname for UI config page.
  - Why: user-facing metadata enrichment.
- `get_by_filter(...)`
  - Filter/paginate memory list by tenant, type, storage, keyword.
  - Why: scalable list API.
- `create_memory(...)`
  - Deduplicates name, converts type bitmask, assembles default system prompt, inserts record.
  - Why: consistent bootstrapping.
- `update_memory(...)`
  - Normalizes fields, handles name dedupe, updates timestamps.
  - Why: safe mutation contract.
- `delete_memory(...)`
  - Deletes metadata row.
  - Why: lifecycle cleanup.

### 4) `api/utils/memory_utils.py`

- `format_ret_data_from_memory(...)`
  - Converts ORM row to API response dict.
  - Why: stable response shape.
- `get_memory_type_human(memory_type)`
  - Bitmask int -> readable list.
  - Why: API ergonomics.
- `calculate_memory_type(memory_type_name_list)`
  - Readable list -> bitmask int.
  - Why: compact storage and fast filtering.

### 5) `memory/utils/prompt_util.py` (`PromptAssembler`)

- `assemble_system_prompt(config)`
  - Builds extraction policy prompt for selected memory types + output schema.
  - Why: keep extraction format controlled and parseable.
- `_get_types_to_extract(...)`
  - Filters unsupported and excludes `raw` from extraction set.
  - Why: RAW is stored directly, not extracted.
- `_generate_type_instructions(...)`
  - Merges type-specific guidance.
  - Why: explicit behavior per memory class.
- `_generate_output_format(...)`
  - Builds exact JSON template.
  - Why: machine-readable outputs.
- `_generate_examples(...)`
  - Adds exemplars for better model conformance.
  - Why: reduces LLM format drift.
- `assemble_user_prompt(conversation, conversation_time, current_time)`
  - Injects dialogue and timing context.
  - Why: temporal grounding for episodic/procedural extraction.

### 6) `api/db/joint_services/memory_message_service.py`

- `save_to_memory(...)`
  - Synchronous path: raw + extraction + embed/save (mostly superseded by queue flow).
  - Why: simpler one-shot integration path.
- `save_extracted_to_memory_only(...)`
  - Extract-only follow-up for queued tasks.
  - Why: async stage 2.
- `extract_by_llm(...)`
  - Builds prompts, calls chat model, parses JSON extraction, normalizes timestamps.
  - Why: converts free-form dialogue into typed memories.
- `embed_and_save(memory, message_list, task_id=None)`
  - Embeds content, creates index if needed, enforces size policy, inserts docs, updates cache.
  - Why: core persistence transaction.
- `query_message(filter_dict, params)`
  - Builds hybrid text/dense/fusion expressions and searches messages.
  - Why: runtime memory retrieval API.
- `init_message_id_sequence()`
  - Seeds Redis auto-increment from existing max message ID.
  - Why: avoid ID collisions across restarts.
- `get_memory_size_cache/set_memory_size_cache/increase_memory_size_cache/decrease_memory_size_cache(...)`
  - Size cache layer around Redis.
  - Why: avoid expensive size recomputation per write.
- `init_memory_size_cache()`
  - Warms per-memory size cache at startup.
  - Why: predictable eviction behavior.
- `judge_system_prompt_is_default(...)`
  - Checks if prompt equals default assembler output.
  - Why: preserve user custom prompts during memory type updates.
- `queue_save_to_memory_task(memory_ids, message_dict)`
  - Saves RAW first, creates task row, pushes extraction tasks to stream.
  - Why: low-latency write + eventual structured extraction.
- `handle_save_to_memory_task(task_param)`
  - Worker execution entrypoint for queued extraction task.
  - Why: asynchronous extraction pipeline.

### 7) HTTP endpoint handlers (`api/apps/sdk/memories.py`, `api/apps/sdk/messages.py`)

- Memory endpoints: create, update, delete, list, get config/detail.
  - Why: memory admin and observability.
- Message endpoints: add, search, list recent, forget, status toggle, get content.
  - Why: runtime ingestion/retrieval controls.

---

## What You Will Build (Learning Implementation)

Build this structure in your learning project:

```
rag-deep-learning/phase-5/
├── app/
│   ├── main.py
│   ├── config.py
│   └── api/routes/
│       ├── memories.py
│       └── messages.py
├── core/
│   ├── enums.py
│   ├── schemas.py
│   ├── prompt_assembler.py
│   ├── memory_service.py
│   ├── message_service.py
│   ├── query_service.py
│   ├── memory_pipeline.py
│   └── worker.py
└── tests/
    └── test_phase5_memory.py
```

This mirrors RAGFlow's decomposition while staying runnable as a focused phase project.

---

## Seed Data (Generated For You)

Use these files under `opencode/data/phase-5`:

- `seed_manifest.json`
- `memory_create_raw_only.json`
- `memory_create_multi_type.json`
- `memory_update_small_fifo_limit.json`
- `prior_phase_context_outputs.json`
- `messages_add_batch.json`
- `mock_llm_extraction_output.json`
- `mock_queue_task_payload.json`
- `messages_search_requests.json`
- `messages_recent_request.json`
- `message_status_toggle_request.json`
- `forget_message_request.json`
- `expected_search_results.json`

These include synthetic outputs from prior phases (citation-rich responses and graph-context-aware answers) so you can run Phase 5 independently.

---

## Step-by-Step Implementation

### Step 0: Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "pydantic>=2.8.0",
  "pydantic-settings>=2.3.0",
  "sqlalchemy>=2.0.30",
  "asyncpg>=0.29.0",
  "elasticsearch>=8.14.0",
  "redis>=5.0.7",
  "openai>=1.40.0",
  "numpy>=1.26.0",
  "python-dateutil>=2.9.0",
]
```

Async I/O profile used in this guide:

- PostgreSQL: SQLAlchemy async engine (`asyncpg` driver)
- Elasticsearch: `AsyncElasticsearch`
- Redis: `redis.asyncio.Redis`

---

### Step 1: Enums and Bitmask Helpers

Create `core/enums.py`:

```python
from enum import Enum


class MemoryType(Enum):
    RAW = 0b0001
    SEMANTIC = 0b0010
    EPISODIC = 0b0100
    PROCEDURAL = 0b1000


def get_memory_type_human(memory_type: int) -> list[str]:
    return [m.name.lower() for m in MemoryType if memory_type & m.value]


def calculate_memory_type(memory_type_name_list: list[str]) -> int:
    type_value_map = {m.name.lower(): m.value for m in MemoryType}
    bits = 0
    for name in memory_type_name_list:
        if name in type_value_map:
            bits |= type_value_map[name]
    return bits
```

Why this exists:

- Keeps storage compact (`int` bitmask) but API human-readable (`list[str]`).
- Exactly how RAGFlow models `memory_type`.

---

### Step 2: Prompt Assembler (Extraction Contract)

Create `core/prompt_assembler.py`:

```python
from typing import Optional
from core.enums import MemoryType


class PromptAssembler:
    SYSTEM_BASE_TEMPLATE = """**Memory Extraction Specialist**
You are an expert at analyzing conversations to extract structured memory.

{type_specific_instructions}

**OUTPUT REQUIREMENTS:**
1. Output MUST be valid JSON
2. Follow the specified output format exactly
3. Each extracted item MUST have: content, valid_at, invalid_at
4. Timestamps in ISO 8601 format
5. Only extract memory types specified above
6. Maximum {max_items} items per type
"""

    TYPE_INSTRUCTIONS = {
        "semantic": """
**EXTRACT SEMANTIC KNOWLEDGE:**
- Facts, definitions, stable relationships.
""",
        "episodic": """
**EXTRACT EPISODIC KNOWLEDGE:**
- Time-bound events and experiences.
""",
        "procedural": """
**EXTRACT PROCEDURAL KNOWLEDGE:**
- Steps, methods, and actionable processes.
""",
    }

    OUTPUT_TEMPLATES = {
        "semantic": '"semantic": [{"content": "...", "valid_at": "...", "invalid_at": ""}]',
        "episodic": '"episodic": [{"content": "...", "valid_at": "...", "invalid_at": ""}]',
        "procedural": '"procedural": [{"content": "...", "valid_at": "...", "invalid_at": ""}]',
    }

    BASE_USER_PROMPT = """**CONVERSATION:**
{conversation}

**CONVERSATION TIME:** {conversation_time}
**CURRENT TIME:** {current_time}
"""

    @classmethod
    def _valid_types(cls, requested: list[str]) -> list[str]:
        all_types = {e.name.lower() for e in MemoryType}
        return [t for t in requested if t in all_types and t != "raw"]

    @classmethod
    def assemble_system_prompt(cls, config: dict) -> str:
        types_to_extract = cls._valid_types(config["memory_type"])
        instructions = "\n".join(cls.TYPE_INSTRUCTIONS[t] for t in types_to_extract)
        output_format = ",\n".join(cls.OUTPUT_TEMPLATES[t] for t in types_to_extract)
        prompt = cls.SYSTEM_BASE_TEMPLATE.format(
            type_specific_instructions=instructions,
            max_items=config.get("max_items_per_type", 5),
        )
        prompt += f"\n**REQUIRED OUTPUT FORMAT (JSON):**\n```json\n{{\n{output_format}\n}}\n```\n"
        return prompt

    @classmethod
    def assemble_user_prompt(
        cls,
        conversation: str,
        conversation_time: Optional[str],
        current_time: Optional[str],
    ) -> str:
        return cls.BASE_USER_PROMPT.format(
            conversation=conversation,
            conversation_time=conversation_time or "Not specified",
            current_time=current_time or "Not specified",
        )
```

Why this exists:

- Controls extraction schema so downstream parsing is deterministic.
- Prevents model drift into unstructured prose.

---

### Step 3: Memory Metadata + Task Tracking (PostgreSQL, Concrete)

Create `core/db.py`:

```python
import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_flow",
)

engine = create_async_engine(POSTGRES_DSN, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = async_sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


async def init_db():
    from core.models import MemoryModel, TaskModel  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session():
    async with SessionLocal() as db:
        yield db
```

Create `core/models.py`:

```python
from datetime import datetime
import uuid

from sqlalchemy import BIGINT, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from core.db import Base


class MemoryModel(Base):
    __tablename__ = "memory"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    memory_type: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_type: Mapped[str] = mapped_column(String(16), default="table")
    embd_id: Mapped[str] = mapped_column(String(256), nullable=False)
    llm_id: Mapped[str] = mapped_column(String(256), nullable=False)
    permissions: Mapped[str] = mapped_column(String(16), default="me")
    memory_size: Mapped[int] = mapped_column(BIGINT, default=5 * 1024 * 1024)
    forgetting_policy: Mapped[str] = mapped_column(String(16), default="FIFO")
    temperature: Mapped[float] = mapped_column(Float, default=0.5)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    user_prompt: Mapped[str] = mapped_column(Text, default="")
    create_time: Mapped[int] = mapped_column(BIGINT)
    update_time: Mapped[int] = mapped_column(BIGINT)
    create_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    update_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TaskModel(Base):
    __tablename__ = "task"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)  # memory_id
    task_type: Mapped[str] = mapped_column(String(16), default="memory")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    progress_msg: Mapped[str] = mapped_column(Text, default="")
    digest: Mapped[str] = mapped_column(String(64), default="")  # source raw message_id
    begin_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    create_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    update_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
```

Create `core/repositories.py`:

```python
from datetime import datetime
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.models import MemoryModel, TaskModel


class MemoryRepository:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def get(self, memory_id: str):
        return await self.db.get(MemoryModel, memory_id)

    async def list_by_tenant(self, tenant_id: str, page: int = 1, page_size: int = 20):
        stmt = (
            select(MemoryModel)
            .where(MemoryModel.tenant_id == tenant_id)
            .order_by(MemoryModel.update_time.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await self.db.scalars(stmt)
        return list(result)

    async def list_by_ids(self, memory_ids: list[str]):
        if not memory_ids:
            return []
        stmt = select(MemoryModel).where(MemoryModel.id.in_(memory_ids))
        result = await self.db.scalars(stmt)
        return list(result)

    async def list_all(self):
        result = await self.db.scalars(select(MemoryModel))
        return list(result)

    async def create(self, payload: dict):
        row = MemoryModel(**payload)
        self.db.add(row)
        await self.db.commit()
        await self.db.refresh(row)
        return row

    async def update(self, memory_id: str, update_dict: dict):
        row = await self.get(memory_id)
        if not row:
            return None
        for k, v in update_dict.items():
            setattr(row, k, v)
        row.update_time = int(datetime.utcnow().timestamp() * 1000)
        row.update_date = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(row)
        return row

    async def delete(self, memory_id: str):
        row = await self.get(memory_id)
        if not row:
            return False
        await self.db.delete(row)
        await self.db.commit()
        return True


class TaskRepository:
    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    async def get(self, task_id: str):
        return await self.db.get(TaskModel, task_id)

    async def create_memory_task(self, memory_id: str, source_id: int):
        task_id = uuid.uuid4().hex
        row = TaskModel(
            id=task_id,
            doc_id=memory_id,
            task_type="memory",
            progress=0.0,
            digest=str(source_id),
            progress_msg="Queued",
        )
        self.db.add(row)
        await self.db.commit()
        return row.id

    async def set_progress(self, task_id: str, progress: float, progress_msg: str):
        row = await self.get(task_id)
        if not row:
            return None
        row.progress = progress
        row.progress_msg = progress_msg
        row.update_date = datetime.utcnow()
        await self.db.commit()
        return row

    async def finish(self, task_id: str, msg: str):
        return await self.set_progress(task_id, 1.0, msg)

    async def fail(self, task_id: str, msg: str):
        return await self.set_progress(task_id, -1.0, msg)
```

Create `core/memory_service.py`:

```python
import uuid
from datetime import datetime

from core.enums import calculate_memory_type
from core.prompt_assembler import PromptAssembler


class MemoryService:
    def __init__(self, repo):
        self.repo = repo

    async def get_by_memory_id(self, memory_id: str):
        return await self.repo.get(memory_id)

    async def get_by_ids(self, memory_ids: list[str]):
        return await self.repo.list_by_ids(memory_ids)

    async def get_by_tenant_id(self, tenant_id: str, page: int = 1, page_size: int = 20):
        return await self.repo.list_by_tenant(tenant_id, page, page_size)

    async def get_all_memory(self):
        return await self.repo.list_all()

    async def create_memory(self, tenant_id: str, name: str, memory_type: list[str], embd_id: str, llm_id: str):
        now = datetime.utcnow()
        payload = {
            "id": uuid.uuid4().hex,
            "name": name,
            "tenant_id": tenant_id,
            "memory_type": calculate_memory_type(memory_type),
            "storage_type": "table",
            "embd_id": embd_id,
            "llm_id": llm_id,
            "permissions": "me",
            "memory_size": 5 * 1024 * 1024,
            "forgetting_policy": "FIFO",
            "temperature": 0.5,
            "system_prompt": PromptAssembler.assemble_system_prompt({"memory_type": memory_type}),
            "user_prompt": "",
            "create_time": int(now.timestamp() * 1000),
            "update_time": int(now.timestamp() * 1000),
            "create_date": now,
            "update_date": now,
        }
        return await self.repo.create(payload)

    async def update_memory(self, memory_id: str, update_dict: dict):
        return await self.repo.update(memory_id, update_dict)

    async def delete_memory(self, memory_id: str):
        return await self.repo.delete(memory_id)
```

Why this exists:

- Gives you a real PostgreSQL-backed metadata and task layer.
- Matches RAGFlow split: relational metadata + search/index store for messages.

---

### Step 4: Query Builder and Message Service

Create `core/query_service.py`:

```python
import re
import numpy as np


async def get_vector(query: str, embedding_model, topk: int = 10, similarity: float = 0.2):
    vec, _ = await embedding_model.encode_queries(query)
    if len(np.array(vec).shape) > 1:
        raise ValueError("Expected one embedding vector")
    vector_field = f"q_{len(vec)}_vec"
    return {
        "kind": "dense",
        "field": vector_field,
        "vector": vec,
        "topk": topk,
        "similarity": similarity,
    }


class MsgTextQuery:
    def question(self, txt: str, min_match: float = 0.6):
        normalized = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5 ]+", " ", txt.lower()).strip()
        tokens = [t for t in normalized.split() if t]
        if not tokens:
            tokens = [normalized]
        weighted_terms = [f"({t})^{1.0}" for t in tokens]
        query = " ".join(weighted_terms)
        return {
            "kind": "text",
            "fields": ["content"],
            "query": query,
            "minimum_should_match": min_match,
            "original_query": txt,
        }
```

Create `core/message_service.py`:

```python
import sys


def index_name(uid: str) -> str:
    return f"memory_{uid}"


class MessageService:
    def __init__(self, store_conn):
        self.store = store_conn

    async def has_index(self, uid: str, memory_id: str) -> bool:
        return await self.store.index_exist(index_name(uid), memory_id)

    async def create_index(self, uid: str, memory_id: str, vector_size: int):
        return await self.store.create_idx(index_name(uid), memory_id, vector_size)

    async def delete_index(self, uid: str, memory_id: str):
        return await self.store.delete_idx(index_name(uid), memory_id)

    async def insert_message(self, messages: list[dict], uid: str, memory_id: str):
        for m in messages:
            m["id"] = f"{memory_id}_{m['message_id']}"
            m["status"] = bool(m["status"])
        return await self.store.insert(messages, index_name(uid), memory_id)

    async def update_message(self, condition: dict, update_dict: dict, uid: str, memory_id: str):
        if "status" in update_dict:
            update_dict["status"] = 1 if update_dict["status"] else 0
        return await self.store.update(condition, update_dict, index_name(uid), memory_id)

    async def delete_message(self, condition: dict, uid: str, memory_id: str):
        return await self.store.delete(condition, index_name(uid), memory_id)

    async def search_message(self, memory_ids: list[str], condition: dict, uid_list: list[str], match_exprs: list[dict], top_n: int):
        index_names = [index_name(uid) for uid in uid_list]
        if "status" not in condition:
            condition["status"] = 1
        res, total_count = await self.store.search(
            select_fields=[
                "message_id", "message_type", "source_id", "memory_id", "user_id", "agent_id", "session_id",
                "valid_at", "invalid_at", "forget_at", "status", "content"
            ],
            highlight_fields=[],
            condition=condition,
            match_expressions=match_exprs,
            order_by={"valid_at": "desc"},
            offset=0,
            limit=top_n,
            index_names=index_names,
            memory_ids=memory_ids,
            agg_fields=[],
        )

        if not total_count:
            return []
        docs = self.store.get_fields(
            res,
            [
                "message_id", "message_type", "source_id", "memory_id", "user_id", "agent_id", "session_id",
                "valid_at", "invalid_at", "forget_at", "status", "content"
            ],
        )
        return list(docs.values())

    async def get_recent_messages(self, uid_list: list[str], memory_ids: list[str], agent_id: str, session_id: str, limit: int):
        index_names = [index_name(uid) for uid in uid_list]
        res, total_count = await self.store.search(
            select_fields=[
                "message_id", "message_type", "source_id", "memory_id", "user_id", "agent_id", "session_id",
                "valid_at", "invalid_at", "forget_at", "status", "content"
            ],
            highlight_fields=[],
            condition={"agent_id": agent_id, "session_id": session_id},
            match_expressions=[],
            order_by={"valid_at": "desc"},
            offset=0,
            limit=limit,
            index_names=index_names,
            memory_ids=memory_ids,
            agg_fields=[],
        )
        if not total_count:
            return []
        return list(
            self.store.get_fields(
                res,
                [
                    "message_id", "message_type", "source_id", "memory_id", "user_id", "agent_id", "session_id",
                    "valid_at", "invalid_at", "forget_at", "status", "content"
                ],
            ).values()
        )

    async def list_message(self, uid: str, memory_id: str, agent_ids: list[str] | None = None, session_keyword: str | None = None, page: int = 1, page_size: int = 50):
        condition = {"message_type": "raw"}
        if agent_ids:
            condition["agent_id"] = agent_ids
        if session_keyword:
            condition["session_id"] = session_keyword

        res, total_count = await self.store.search(
            select_fields=[
                "message_id", "message_type", "source_id", "memory_id", "user_id", "agent_id", "session_id",
                "valid_at", "invalid_at", "forget_at", "status", "content"
            ],
            highlight_fields=[],
            condition=condition,
            match_expressions=[],
            order_by={"valid_at": "desc"},
            offset=(page - 1) * page_size,
            limit=page_size,
            index_names=[index_name(uid)],
            memory_ids=[memory_id],
            agg_fields=[],
            hide_forgotten=False,
        )
        if not total_count:
            return {"message_list": [], "total_count": 0}

        raw_messages = list(
            self.store.get_fields(
                res,
                [
                    "message_id", "message_type", "source_id", "memory_id", "user_id", "agent_id", "session_id",
                    "valid_at", "invalid_at", "forget_at", "status", "content"
                ],
            ).values()
        )
        source_ids = [m["message_id"] for m in raw_messages]
        if not source_ids:
            return {"message_list": raw_messages, "total_count": total_count}

        child_res, _ = await self.store.search(
            select_fields=[
                "message_id", "message_type", "source_id", "memory_id", "user_id", "agent_id", "session_id",
                "valid_at", "invalid_at", "forget_at", "status", "content"
            ],
            highlight_fields=[],
            condition={"source_id": source_ids},
            match_expressions=[],
            order_by={"valid_at": "desc"},
            offset=0,
            limit=512,
            index_names=[index_name(uid)],
            memory_ids=[memory_id],
            agg_fields=[],
            hide_forgotten=False,
        )
        child_docs = list(
            self.store.get_fields(
                child_res,
                [
                    "message_id", "message_type", "source_id", "memory_id", "user_id", "agent_id", "session_id",
                    "valid_at", "invalid_at", "forget_at", "status", "content"
                ],
            ).values()
        ) if child_res else []

        grouped = {}
        for c in child_docs:
            grouped.setdefault(c["source_id"], []).append(c)
        for r in raw_messages:
            r["extract"] = grouped.get(r["message_id"], [])
        return {"message_list": raw_messages, "total_count": total_count}

    @staticmethod
    def calculate_message_size(message: dict) -> int:
        return sys.getsizeof(message["content"]) + sys.getsizeof(message["content_embed"][0]) * len(message["content_embed"])

    async def calculate_memory_size(self, memory_ids: list[str], uid_list: list[str]):
        index_names = [index_name(uid) for uid in uid_list]
        res, count = await self.store.search(
            select_fields=["memory_id", "content", "content_embed"],
            highlight_fields=[],
            condition={},
            match_expressions=[],
            order_by={"valid_at": "desc"},
            offset=0,
            limit=2048 * max(1, len(memory_ids)),
            index_names=index_names,
            memory_ids=memory_ids,
            agg_fields=[],
            hide_forgotten=False,
        )
        if count == 0:
            return {}

        docs = self.store.get_fields(res, ["memory_id", "content", "content_embed"])
        size_map = {}
        for d in docs.values():
            size = self.calculate_message_size(d)
            size_map[d["memory_id"]] = size_map.get(d["memory_id"], 0) + size
        return size_map

    async def pick_messages_to_delete_by_fifo(self, memory_id: str, uid: str, size_to_delete: int):
        idx = index_name(uid)
        selected_ids = []
        selected_size = 0

        forgotten_res = await self.store.get_forgotten_messages(["message_id", "content", "content_embed"], idx, memory_id)
        forgotten = self.store.get_fields(forgotten_res, ["message_id", "content", "content_embed"]) if forgotten_res else {}
        for d in forgotten.values():
            if selected_size >= size_to_delete:
                return selected_ids, selected_size
            selected_ids.append(d["message_id"])
            selected_size += self.calculate_message_size(d)

        res, total_count = await self.store.search(
            select_fields=["message_id", "content", "content_embed"],
            highlight_fields=[],
            condition={},
            match_expressions=[],
            order_by={"valid_at": "asc"},
            offset=0,
            limit=512,
            index_names=[idx],
            memory_ids=[memory_id],
            agg_fields=[],
            hide_forgotten=False,
        )
        if not total_count:
            return selected_ids, selected_size

        docs = self.store.get_fields(res, ["message_id", "content", "content_embed"])
        for d in docs.values():
            if selected_size >= size_to_delete:
                break
            if d["message_id"] in selected_ids:
                continue
            selected_ids.append(d["message_id"])
            selected_size += self.calculate_message_size(d)
        return selected_ids, selected_size

    async def get_by_message_id(self, memory_id: str, message_id: int, uid: str):
        return await self.store.get(f"{memory_id}_{message_id}", index_name(uid), [memory_id])

    async def get_max_message_id(self, uid_list: list[str], memory_ids: list[str]):
        index_names = [index_name(uid) for uid in uid_list]
        res, total_count = await self.store.search(
            select_fields=["message_id"],
            highlight_fields=[],
            condition={},
            match_expressions=[],
            order_by={"message_id": "desc"},
            offset=0,
            limit=1,
            index_names=index_names,
            memory_ids=memory_ids,
            agg_fields=[],
            hide_forgotten=False,
        )
        if not total_count:
            return 1
        docs = self.store.get_fields(res, ["message_id"])
        top = next(iter(docs.values()))
        return int(top["message_id"])
```

Why these exist:

- `query_service.py`: builds fusion-ready lexical+dense search expressions.
- `message_service.py`: central message CRUD and retrieval contract.

---

### Step 4B: Implement Elasticsearch Message Store (Concrete, Runnable)

You are right to standardize on ES for this phase. Below is a full adapter you can use directly in your code-along project.

Create `core/message_store_es.py`:

```python
import copy
import json
import re
from pathlib import Path

from elasticsearch import AsyncElasticsearch, NotFoundError
from elasticsearch.helpers import async_bulk
from elasticsearch_dsl import Q, Search


class ESMessageStore:
    """
    ES-backed message store for Phase 5 memory.
    Mirrors RAGFlow's message-store behavior, simplified only where noted.
    """

    def __init__(self, hosts: str, username: str = "", password: str = "", mapping_path: str = "conf/mapping.json"):
        auth = (username, password) if username else None
        self.es = AsyncElasticsearch(hosts=hosts, basic_auth=auth, verify_certs=False, request_timeout=60)
        self.mapping = json.loads(Path(mapping_path).read_text())

    @staticmethod
    def convert_field_name(field_name: str) -> str:
        if field_name == "message_type":
            return "message_type_kwd"
        if field_name == "status":
            return "status_int"
        if field_name == "content":
            return "content_ltks"
        return field_name

    @staticmethod
    def map_message_to_es_fields(message: dict) -> dict:
        return {
            "id": message.get("id"),
            "message_id": message["message_id"],
            "message_type_kwd": message["message_type"],
            "source_id": message["source_id"],
            "memory_id": message["memory_id"],
            "user_id": message.get("user_id", ""),
            "agent_id": message["agent_id"],
            "session_id": message["session_id"],
            "valid_at": message["valid_at"],
            "invalid_at": message.get("invalid_at"),
            "forget_at": message.get("forget_at"),
            "status_int": 1 if message["status"] else 0,
            "content_ltks": message["content"],
            f"q_{len(message['content_embed'])}_vec": message["content_embed"],
        }

    @staticmethod
    def get_message_from_es_doc(doc: dict) -> dict:
        emb_field = next((k for k in doc.keys() if re.match(r"q_\d+_vec", k)), None)
        return {
            "id": doc.get("id", ""),
            "message_id": doc["message_id"],
            "message_type": doc["message_type_kwd"],
            "source_id": doc.get("source_id"),
            "memory_id": doc["memory_id"],
            "user_id": doc.get("user_id", ""),
            "agent_id": doc.get("agent_id", ""),
            "session_id": doc.get("session_id", ""),
            "valid_at": doc.get("valid_at"),
            "invalid_at": doc.get("invalid_at"),
            "forget_at": doc.get("forget_at"),
            "status": bool(int(doc.get("status_int", 1))),
            "content": doc.get("content_ltks", ""),
            "content_embed": doc.get(emb_field, []) if emb_field else [],
        }

    async def create_idx(self, index_name: str, memory_id: str, vector_size: int):
        # ES index is tenant-scoped in this design; memory_id is filter field.
        if await self.index_exist(index_name, memory_id):
            return True
        await self.es.indices.create(
            index=index_name,
            settings=self.mapping["settings"],
            mappings=self.mapping["mappings"],
        )
        return True

    async def delete_idx(self, index_name: str, memory_id: str):
        # Match RAGFlow behavior: do not drop tenant index when deleting one memory_id.
        if memory_id:
            return True
        try:
            await self.es.indices.delete(index=index_name, allow_no_indices=True)
        except NotFoundError:
            return True
        return True

    async def index_exist(self, index_name: str, memory_id: str = "") -> bool:
        return bool(await self.es.indices.exists(index=index_name))

    async def insert(self, rows: list[dict], index_name: str, memory_id: str = None) -> list[str]:
        ops = []
        for row in rows:
            row_copy = copy.deepcopy(row)
            doc = self.map_message_to_es_fields(row_copy)
            doc["memory_id"] = memory_id
            doc_id = doc.pop("id")
            ops.append({"_op_type": "index", "_index": index_name, "_id": doc_id, "_source": doc})

        success, errors = await async_bulk(self.es, ops, raise_on_error=False, refresh=False)
        if not errors:
            return []
        return [str(e) for e in errors]

    def _build_filter_query(self, condition: dict, memory_ids: list[str], hide_forgotten: bool = True):
        q = Q("bool", must=[], filter=[], must_not=[])
        if hide_forgotten:
            q.must_not.append(Q("exists", field="forget_at"))

        q.filter.append(Q("terms", memory_id=memory_ids))
        for k, v in condition.items():
            if k == "memory_id":
                continue
            field = self.convert_field_name(k)
            if not v and v != 0:
                continue
            if isinstance(v, list):
                q.filter.append(Q("terms", **{field: v}))
            else:
                q.filter.append(Q("term", **{field: v}))
        return q

    async def search(
        self,
        select_fields: list[str],
        highlight_fields: list[str],
        condition: dict,
        match_expressions: list[dict],
        order_by,
        offset: int,
        limit: int,
        index_names: str | list[str],
        memory_ids: list[str],
        agg_fields: list[str] | None = None,
        rank_feature: dict | None = None,
        hide_forgotten: bool = True,
    ):
        if isinstance(index_names, str):
            index_names = [index_names]
        existing = []
        for i in index_names:
            if await self.index_exist(i):
                existing.append(i)
        if not existing:
            return None, 0

        base_filter = self._build_filter_query(condition, memory_ids, hide_forgotten=hide_forgotten)

        text_expr = next((m for m in match_expressions if m.get("kind") == "text"), None)
        dense_expr = next((m for m in match_expressions if m.get("kind") == "dense"), None)
        fusion_expr = next((m for m in match_expressions if m.get("kind") == "fusion"), None)

        s = Search()

        # weighted_sum is carried as a config knob; ES scoring blend here is approximate but practical.
        vector_weight = 0.5
        if fusion_expr and "weights" in fusion_expr:
            vector_weight = float(fusion_expr["weights"][1])

        if text_expr:
            min_match = text_expr.get("minimum_should_match", 0.0)
            if isinstance(min_match, float):
                min_match = f"{int(min_match * 100)}%"
            text_query = Q(
                "query_string",
                fields=[self.convert_field_name(f) for f in text_expr.get("fields", ["content"])],
                query=text_expr["query"],
                minimum_should_match=min_match,
                boost=(1.0 - vector_weight),
            )
            base_filter.must.append(text_query)

        if dense_expr:
            similarity = dense_expr.get("similarity", 0.0)
            s = s.knn(
                self.convert_field_name(dense_expr["field"]),
                dense_expr.get("topk", limit),
                dense_expr.get("topk", limit) * 2,
                query_vector=list(dense_expr["vector"]),
                filter=base_filter.to_dict(),
                similarity=similarity,
            )

        s = s.query(base_filter)

        if order_by and isinstance(order_by, dict):
            sort_list = []
            for field, direction in order_by.items():
                sort_list.append({self.convert_field_name(field): {"order": direction}})
            s = s.sort(*sort_list)

        if limit > 0:
            s = s[offset : offset + limit]

        res = await self.es.search(index=existing, body=s.to_dict(), track_total_hits=True, _source=True)
        total = res["hits"]["total"]["value"] if isinstance(res["hits"]["total"], dict) else res["hits"]["total"]
        return res, total

    async def get(self, data_id: str, index_name: str, memory_ids: list[str]) -> dict | None:
        try:
            res = await self.es.get(index=index_name, id=data_id, source=True)
        except NotFoundError:
            return None
        doc = res["_source"]
        doc["id"] = data_id
        return self.get_message_from_es_doc(doc)

    async def update(self, condition: dict, new_value: dict, index_name: str, memory_id: str) -> bool:
        mapped_update = {self.convert_field_name(k): v for k, v in new_value.items()}
        mapped_condition = {self.convert_field_name(k): v for k, v in condition.items()}
        mapped_condition["memory_id"] = memory_id

        if "id" in mapped_condition and isinstance(mapped_condition["id"], str):
            doc_id = mapped_condition["id"]
            await self.es.update(index=index_name, id=doc_id, doc=mapped_update)
            return True

        q = self._build_filter_query(mapped_condition, [memory_id], hide_forgotten=False)
        params = {}
        scripts = []
        for k, v in mapped_update.items():
            p = f"p_{k}"
            scripts.append(f"ctx._source.{k}=params.{p};")
            params[p] = v

        await self.es.update_by_query(
            index=index_name,
            refresh=True,
            conflicts="proceed",
            body={
                "query": q.to_dict(),
                "script": {"source": "".join(scripts), "params": params},
            },
        )
        return True

    async def delete(self, condition: dict, index_name: str, memory_id: str) -> int:
        mapped_condition = {self.convert_field_name(k): v for k, v in condition.items()}
        if "id" in mapped_condition:
            values = mapped_condition["id"]
            if not isinstance(values, list):
                values = [values]
            q = Q("bool", must=[Q("ids", values=values), Q("term", memory_id=memory_id)])
        else:
            q = self._build_filter_query(mapped_condition, [memory_id], hide_forgotten=False)

        try:
            res = await self.es.delete_by_query(index=index_name, body=Search().query(q).to_dict(), refresh=True)
            return int(res.get("deleted", 0))
        except NotFoundError:
            return 0

    def get_fields(self, res, fields: list[str]) -> dict[str, dict]:
        if not res:
            return {}
        out = {}
        for hit in res["hits"]["hits"]:
            src = hit["_source"]
            src["id"] = hit["_id"]
            msg = self.get_message_from_es_doc(src)
            out[hit["_id"]] = {k: msg[k] for k in fields if k in msg}
        return out

    async def get_forgotten_messages(self, select_fields: list[str], index_name: str, memory_id: str, limit: int = 512):
        q = Q("bool", must=[Q("term", memory_id=memory_id), Q("exists", field="forget_at")])
        s = Search().query(q).sort({"forget_at": {"order": "asc"}})[:limit]
        return await self.es.search(index=index_name, body=s.to_dict(), track_total_hits=False, _source=True)

    async def close(self):
        await self.es.close()
```

Create `app/config.py` wiring:

```python
import os

from core.message_store_es import ESMessageStore


DOC_ENGINE = os.getenv("DOC_ENGINE", "elasticsearch")
ES_HOSTS = os.getenv("ES_HOSTS", "http://localhost:1200")
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "infini_rag_flow")

if DOC_ENGINE != "elasticsearch":
    raise RuntimeError("This phase-5 learning path expects DOC_ENGINE=elasticsearch")

MESSAGE_STORE = ESMessageStore(
    hosts=ES_HOSTS,
    username=ES_USERNAME,
    password=ES_PASSWORD,
    mapping_path="conf/mapping.json",
)
```

Wire into your service container:

```python
from app.config import MESSAGE_STORE
from core.message_service import MessageService

message_service = MessageService(MESSAGE_STORE)
```

What this means for your specific question:

- If you use real RAGFlow code: this is already implemented end-to-end.
- If you use the learning project from this guide: add this file and wiring once.
- You do NOT manually pass an index name per request.
  - Index name is derived as `memory_{tenant_id}`.
  - `memory_id` is passed as filter (and storage partition key).

ES setup checklist:

- `DOC_ENGINE=elasticsearch`
- ES host + credentials configured.
- `conf/mapping.json` available.
- Embedding dimensions should match mapping templates (512/768/1024/1536 unless you extend mapping).

---

### Step 5: Memory Orchestration Pipeline (Core of Phase 5)

Create `core/memory_pipeline.py`:

```python
import json
from datetime import datetime

from dateutil import parser

from core.enums import MemoryType, get_memory_type_human
from core.prompt_assembler import PromptAssembler
from core.query_service import get_vector, MsgTextQuery


def now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def to_ymd_hms(iso_value: str) -> str:
    if not iso_value:
        return ""
    try:
        _ = parser.isoparse(iso_value)
        return datetime.fromisoformat(iso_value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_value


class MemoryPipeline:
    def __init__(
        self,
        memory_service,
        message_service,
        embedding_factory,
        chat_factory,
        redis_client,
        task_repo,
        stream_name: str = "rag_flow_svr_queue",
    ):
        self.memory_service = memory_service
        self.message_service = message_service
        self.embedding_factory = embedding_factory
        self.chat_factory = chat_factory
        self.redis = redis_client
        self.task_repo = task_repo
        self.stream_name = stream_name

    @staticmethod
    def _size_cache_key(memory_id: str) -> str:
        return f"memory_{memory_id}"

    async def get_memory_size_cache(self, memory_id: str, uid: str):
        k = self._size_cache_key(memory_id)
        cached = await self.redis.get(k)
        if cached is not None:
            return int(cached)

        size_map = await self.message_service.calculate_memory_size([memory_id], [uid])
        sz = int(size_map.get(memory_id, 0))
        await self.redis.set(k, sz)
        return sz

    async def set_memory_size_cache(self, memory_id: str, size: int):
        return await self.redis.set(self._size_cache_key(memory_id), int(size))

    async def increase_memory_size_cache(self, memory_id: str, size: int):
        return await self.redis.incrby(self._size_cache_key(memory_id), int(size))

    async def decrease_memory_size_cache(self, memory_id: str, size: int):
        return await self.redis.decrby(self._size_cache_key(memory_id), int(size))

    async def init_message_id_sequence(self):
        redis_key = "id_generator:memory"
        current = await self.redis.get(redis_key)
        if current is not None:
            return int(current)

        memories = await self.memory_service.get_all_memory()
        if not memories:
            await self.redis.set(redis_key, 1)
            return 1

        max_id = await self.message_service.get_max_message_id(
            [m.tenant_id for m in memories],
            [m.id for m in memories],
        )
        await self.redis.set(redis_key, int(max_id))
        return int(max_id)

    async def init_memory_size_cache(self):
        memories = await self.memory_service.get_all_memory()
        for m in memories:
            await self.get_memory_size_cache(m.id, m.tenant_id)

    async def extract_by_llm(
        self,
        tenant_id: str,
        llm_id: str,
        extract_conf: dict,
        memory_type: list[str],
        user_input: str,
        agent_response: str,
        system_prompt: str = "",
        user_prompt: str = "",
    ) -> list[dict]:
        if not system_prompt:
            system_prompt = PromptAssembler.assemble_system_prompt({"memory_type": memory_type})
        conversation = f"User Input: {user_input}\nAgent Response: {agent_response}"
        conv_time = now_str()
        if not user_prompt:
            user_prompt = PromptAssembler.assemble_user_prompt(conversation, conv_time, conv_time)

        llm = self.chat_factory(tenant_id, llm_id)
        raw = await llm.async_chat(system_prompt, [{"role": "user", "content": user_prompt}], extract_conf)

        try:
            parsed = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        except Exception:
            parsed = {}

        flattened = []
        for msg_type, arr in parsed.items():
            for item in arr:
                flattened.append({
                    "message_type": msg_type,
                    "content": item["content"],
                    "valid_at": to_ymd_hms(item.get("valid_at", "")) or now_str(),
                    "invalid_at": to_ymd_hms(item.get("invalid_at", "")),
                })
        return flattened

    async def embed_and_save(self, memory, message_list: list[dict]):
        embedder = self.embedding_factory(memory.tenant_id, memory.embd_id)
        vectors, _ = await embedder.encode([m["content"] for m in message_list])
        for i, msg in enumerate(message_list):
            msg["content_embed"] = vectors[i]

        vector_dim = len(vectors[0])
        if not await self.message_service.has_index(memory.tenant_id, memory.id):
            await self.message_service.create_index(memory.tenant_id, memory.id, vector_dim)

        current_size = await self.get_memory_size_cache(memory.id, memory.tenant_id)
        incoming_size = sum(self.message_service.calculate_message_size(m) for m in message_list)
        if current_size + incoming_size > memory.memory_size:
            overflow = current_size + incoming_size - memory.memory_size
            if memory.forgetting_policy != "FIFO":
                return False, "Memory size exceeded and forgetting policy is not FIFO."

            ids_to_remove, removed_size = await self.message_service.pick_messages_to_delete_by_fifo(
                memory.id,
                memory.tenant_id,
                overflow,
            )
            if ids_to_remove:
                await self.message_service.delete_message({"message_id": ids_to_remove}, memory.tenant_id, memory.id)
                await self.decrease_memory_size_cache(memory.id, removed_size)
                current_size = max(0, current_size - removed_size)

            if current_size + incoming_size > memory.memory_size:
                return False, "Memory size exceeded even after FIFO deletion."

        fail_cases = await self.message_service.insert_message(message_list, memory.tenant_id, memory.id)
        if fail_cases:
            return False, "; ".join(fail_cases)

        await self.increase_memory_size_cache(memory.id, incoming_size)
        return True, "Message saved successfully."

    async def queue_save_to_memory_task(self, memory_ids: list[str], message_dict: dict):
        not_found = []
        failed = []
        for memory_id in memory_ids:
            memory = await self.memory_service.get_by_memory_id(memory_id)
            if not memory:
                not_found.append(memory_id)
                continue

            raw_message_id = int(await self.redis.incrby("id_generator:memory", 1))
            raw_message = {
                "message_id": raw_message_id,
                "message_type": MemoryType.RAW.name.lower(),
                "source_id": 0,
                "memory_id": memory_id,
                "user_id": "",
                "agent_id": message_dict["agent_id"],
                "session_id": message_dict["session_id"],
                "content": f"User Input: {message_dict['user_input']}\nAgent Response: {message_dict['agent_response']}",
                "valid_at": now_str(),
                "invalid_at": None,
                "forget_at": None,
                "status": True,
            }

            ok, msg = await self.embed_and_save(memory, [raw_message])
            if not ok:
                failed.append({"memory_id": memory_id, "error": msg})
                continue

            task_id = await self.task_repo.create_memory_task(memory_id, source_id=raw_message_id)
            await self.redis.xadd(self.stream_name, {
                "message": json.dumps({
                    "id": task_id,
                    "task_id": task_id,
                    "task_type": "memory",
                    "memory_id": memory_id,
                    "source_id": raw_message_id,
                    "message_dict": message_dict,
                })
            })

        if not_found or failed:
            msg = ""
            if not_found:
                msg += f"Memory not found: {not_found}. "
            if failed:
                msg += " ".join([f"Memory {f['memory_id']} failed: {f['error']}" for f in failed])
            return False, msg.strip()

        return True, "All add to task."

    async def handle_save_to_memory_task(self, task_param: dict):
        memory = await self.memory_service.get_by_memory_id(task_param["memory_id"])
        if not memory:
            await self.task_repo.fail(task_param["task_id"], "Memory not found")
            return False, "Memory not found"

        if memory.memory_type == MemoryType.RAW.value:
            await self.task_repo.finish(task_param["task_id"], "RAW-only memory. No extraction required.")
            return True, "RAW-only memory"

        await self.task_repo.set_progress(task_param["task_id"], 0.15, "Prepared prompts and extractor")

        extracted = await self.extract_by_llm(
            tenant_id=memory.tenant_id,
            llm_id=memory.llm_id,
            extract_conf={"temperature": memory.temperature},
            memory_type=get_memory_type_human(memory.memory_type),
            user_input=task_param["message_dict"]["user_input"],
            agent_response=task_param["message_dict"]["agent_response"],
            system_prompt=memory.system_prompt or "",
            user_prompt=memory.user_prompt or "",
        )

        if not extracted:
            await self.task_repo.finish(task_param["task_id"], "No memory extracted from raw message.")
            return True, "No extraction"

        await self.task_repo.set_progress(task_param["task_id"], 0.5, f"Extracted {len(extracted)} memory items")

        rows = []
        for item in extracted:
            message_id = int(await self.redis.incrby("id_generator:memory", 1))
            rows.append({
                "message_id": message_id,
                "message_type": item["message_type"],
                "source_id": task_param["source_id"],
                "memory_id": memory.id,
                "user_id": "",
                "agent_id": task_param["message_dict"]["agent_id"],
                "session_id": task_param["message_dict"]["session_id"],
                "content": item["content"],
                "valid_at": item["valid_at"],
                "invalid_at": item["invalid_at"] or None,
                "forget_at": None,
                "status": True,
            })

        ok, msg = await self.embed_and_save(memory, rows)
        if ok:
            await self.task_repo.finish(task_param["task_id"], msg)
            return True, msg

        await self.task_repo.fail(task_param["task_id"], msg)
        return False, msg

    async def query_message(self, filter_dict: dict, params: dict):
        memory_ids = filter_dict["memory_id"]
        memories = []
        for mid in memory_ids:
            m = await self.memory_service.get_by_memory_id(mid)
            if m:
                memories.append(m)
        memories = [m for m in memories if m]
        if not memories:
            return []

        memory = memories[0]
        embd_model = self.embedding_factory(memory.tenant_id, memory.embd_id)
        dense = await get_vector(params["query"].strip(), embd_model, similarity=params["similarity_threshold"])
        text = MsgTextQuery().question(params["query"].strip(), min_match=params["similarity_threshold"])
        ksw = params.get("keywords_similarity_weight", 0.7)
        fusion = {"kind": "fusion", "method": "weighted_sum", "weights": [1 - ksw, ksw], "topn": params["top_n"]}

        condition = {k: v for k, v in filter_dict.items() if v}
        uid_list = [m.tenant_id for m in memories]
        return await self.message_service.search_message(memory_ids, condition, uid_list, [text, dense, fusion], params["top_n"])

    async def list_memory_messages(self, memory_id: str, tenant_id: str, page: int, page_size: int, agent_ids: list[str] | None = None, session_keyword: str | None = None):
        return await self.message_service.list_message(tenant_id, memory_id, agent_ids=agent_ids, session_keyword=session_keyword, page=page, page_size=page_size)

    async def get_recent_messages(self, memory_ids: list[str], agent_id: str, session_id: str, limit: int):
        memories = await self.memory_service.get_by_ids(memory_ids)
        if not memories:
            return []
        uid_list = [m.tenant_id for m in memories]
        return await self.message_service.get_recent_messages(uid_list, memory_ids, agent_id, session_id, limit)

    async def set_message_status(self, memory_id: str, message_id: int, status: bool):
        memory = await self.memory_service.get_by_memory_id(memory_id)
        if not memory:
            return False
        return bool(await self.message_service.update_message({"id": f"{memory_id}_{message_id}"}, {"status": status}, memory.tenant_id, memory_id))

    async def forget_message(self, memory_id: str, message_id: int):
        memory = await self.memory_service.get_by_memory_id(memory_id)
        if not memory:
            return False
        return bool(await self.message_service.update_message({"id": f"{memory_id}_{message_id}"}, {"forget_at": now_str()}, memory.tenant_id, memory_id))

    async def get_message_content(self, memory_id: str, message_id: int):
        memory = await self.memory_service.get_by_memory_id(memory_id)
        if not memory:
            return None
        return await self.message_service.get_by_message_id(memory_id, message_id, memory.tenant_id)
```

Why this file is the heart of Phase 5:

- It implements the exact ingestion/retrieval orchestration boundary used in RAGFlow.
- Every endpoint and worker path eventually routes through these functions.

---

### Step 6: API Endpoints

Create `app/model_clients.py`:

```python
import os

from openai import AsyncOpenAI


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class EmbeddingClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    async def encode(self, texts: list[str]):
        res = await self.client.embeddings.create(model=self.model_name, input=texts)
        vectors = [d.embedding for d in res.data]
        return vectors, len(vectors)

    async def encode_queries(self, query: str):
        vecs, _ = await self.encode([query])
        return vecs[0], 1


class ChatClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    async def async_chat(self, system_prompt: str, messages: list[dict], extract_conf: dict):
        payload = [{"role": "system", "content": system_prompt}, *messages]
        res = await self.client.chat.completions.create(
            model=self.model_name,
            messages=payload,
            temperature=extract_conf.get("temperature", 0.2),
        )
        return res.choices[0].message.content


def embedding_factory(tenant_id: str, model_name: str):
    return EmbeddingClient(model_name)


def chat_factory(tenant_id: str, model_name: str):
    return ChatClient(model_name)
```

Create `app/dependencies.py`:

```python
import os

from fastapi import Depends
from redis.asyncio import Redis

from app.config import MESSAGE_STORE
from app.model_clients import embedding_factory, chat_factory
from core.db import get_db_session
from core.memory_pipeline import MemoryPipeline
from core.memory_service import MemoryService
from core.message_service import MessageService
from core.repositories import MemoryRepository, TaskRepository


REDIS_URL = os.getenv("REDIS_URL", "redis://:infini_rag_flow@localhost:6379/1")
redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
message_service_singleton = MessageService(MESSAGE_STORE)


def build_pipeline(db):
    memory_service = MemoryService(MemoryRepository(db))
    task_repo = TaskRepository(db)
    pipeline = MemoryPipeline(
        memory_service=memory_service,
        message_service=message_service_singleton,
        embedding_factory=embedding_factory,
        chat_factory=chat_factory,
        redis_client=redis_client,
        task_repo=task_repo,
    )
    return pipeline


async def get_pipeline(db=Depends(get_db_session)):
    return build_pipeline(db)
```

Create `app/api/routes/memories.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_pipeline
from core.enums import get_memory_type_human


router = APIRouter()


@router.post("/memories")
async def create_memory(req: dict, pipeline=Depends(get_pipeline)):
    required = ["name", "memory_type", "embd_id", "llm_id"]
    missing = [k for k in required if k not in req]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing: {missing}")

    tenant_id = req.get("tenant_id", "tenant_acme_single_001")
    row = await pipeline.memory_service.create_memory(
        tenant_id=tenant_id,
        name=req["name"],
        memory_type=req["memory_type"],
        embd_id=req["embd_id"],
        llm_id=req["llm_id"],
    )
    return {"code": 0, "data": {"id": row.id}, "message": "success"}


@router.get("/memories")
async def list_memories(
    tenant_id: str = Query(default="tenant_acme_single_001"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    pipeline=Depends(get_pipeline),
):
    rows = await pipeline.memory_service.get_by_tenant_id(tenant_id, page, page_size)
    data = [
        {
            "id": r.id,
            "name": r.name,
            "memory_type": get_memory_type_human(r.memory_type),
            "embd_id": r.embd_id,
            "llm_id": r.llm_id,
            "memory_size": r.memory_size,
            "forgetting_policy": r.forgetting_policy,
            "update_time": r.update_time,
        }
        for r in rows
    ]
    return {"code": 0, "data": {"memory_list": data}, "message": "success"}


@router.get("/memories/{memory_id}")
async def get_memory(memory_id: str, pipeline=Depends(get_pipeline)):
    row = await pipeline.memory_service.get_by_memory_id(memory_id)
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {
        "code": 0,
        "data": {
            "id": row.id,
            "name": row.name,
            "memory_type": get_memory_type_human(row.memory_type),
            "embd_id": row.embd_id,
            "llm_id": row.llm_id,
            "memory_size": row.memory_size,
            "forgetting_policy": row.forgetting_policy,
            "system_prompt": row.system_prompt,
            "user_prompt": row.user_prompt,
        },
        "message": "success",
    }


@router.get("/memories/{memory_id}/config")
async def get_memory_config(memory_id: str, pipeline=Depends(get_pipeline)):
    row = await pipeline.memory_service.get_by_memory_id(memory_id)
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {
        "code": 0,
        "data": {
            "memory_type": get_memory_type_human(row.memory_type),
            "embd_id": row.embd_id,
            "llm_id": row.llm_id,
            "temperature": row.temperature,
            "system_prompt": row.system_prompt,
            "user_prompt": row.user_prompt,
        },
        "message": "success",
    }


@router.put("/memories/{memory_id}")
async def update_memory(memory_id: str, req: dict, pipeline=Depends(get_pipeline)):
    row = await pipeline.memory_service.get_by_memory_id(memory_id)
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")
    updated = await pipeline.memory_service.update_memory(memory_id, req)
    return {"code": 0, "data": {"id": updated.id}, "message": "success"}


@router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str, pipeline=Depends(get_pipeline)):
    row = await pipeline.memory_service.get_by_memory_id(memory_id)
    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")

    await pipeline.message_service.delete_message({}, row.tenant_id, memory_id)
    await pipeline.message_service.delete_index(row.tenant_id, memory_id)
    ok = await pipeline.memory_service.delete_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Delete failed")
    return {"code": 0, "message": "success"}
```

Create `app/api/routes/messages.py`:

```python
from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_pipeline


router = APIRouter()


def parse_message_key(message_key: str):
    try:
        memory_id, message_id = message_key.split(":")
        return memory_id, int(message_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail="message_key must be '<memory_id>:<message_id>'") from e


@router.post("/messages")
async def add_message(req: dict, pipeline=Depends(get_pipeline)):
    required = ["memory_id", "agent_id", "session_id", "user_input", "agent_response"]
    missing = [k for k in required if k not in req]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing: {missing}")

    ok, msg = await pipeline.queue_save_to_memory_task(
        req["memory_id"],
        {
            "user_id": req.get("user_id", ""),
            "agent_id": req["agent_id"],
            "session_id": req["session_id"],
            "user_input": req["user_input"],
            "agent_response": req["agent_response"],
        },
    )
    if not ok:
        raise HTTPException(status_code=500, detail=msg)
    return {"code": 0, "message": msg}


@router.get("/messages/search")
async def search_message(
    memory_id: list[str],
    query: str,
    similarity_threshold: float = 0.2,
    keywords_similarity_weight: float = 0.7,
    top_n: int = 10,
    agent_id: str = "",
    session_id: str = "",
    pipeline=Depends(get_pipeline),
):
    data = await pipeline.query_message(
        {"memory_id": memory_id, "agent_id": agent_id, "session_id": session_id},
        {
            "query": query,
            "similarity_threshold": similarity_threshold,
            "keywords_similarity_weight": keywords_similarity_weight,
            "top_n": top_n,
        },
    )
    return {"code": 0, "message": True, "data": data}


@router.get("/messages")
async def list_messages(
    memory_id: str,
    tenant_id: str = "tenant_acme_single_001",
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
    pipeline=Depends(get_pipeline),
):
    data = await pipeline.list_memory_messages(memory_id, tenant_id, page, page_size)
    return {"code": 0, "message": True, "data": data}


@router.get("/messages/recent")
async def recent_messages(
    memory_id: list[str],
    agent_id: str,
    session_id: str,
    limit: int = 10,
    pipeline=Depends(get_pipeline),
):
    data = await pipeline.get_recent_messages(memory_id, agent_id, session_id, limit)
    return {"code": 0, "message": True, "data": data}


@router.get("/messages/{message_key}")
async def get_message_content(message_key: str, pipeline=Depends(get_pipeline)):
    memory_id, message_id = parse_message_key(message_key)
    doc = await pipeline.get_message_content(memory_id, message_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"code": 0, "message": True, "data": doc}


@router.put("/messages/{message_key}")
async def update_message_status(message_key: str, req: dict, pipeline=Depends(get_pipeline)):
    memory_id, message_id = parse_message_key(message_key)
    if "status" not in req:
        raise HTTPException(status_code=400, detail="Missing status")
    ok = await pipeline.set_message_status(memory_id, message_id, bool(req["status"]))
    if not ok:
        raise HTTPException(status_code=500, detail="Update failed")
    return {"code": 0, "message": "success"}


@router.delete("/messages/{message_key}")
async def forget_message(message_key: str, pipeline=Depends(get_pipeline)):
    memory_id, message_id = parse_message_key(message_key)
    ok = await pipeline.forget_message(memory_id, message_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Forget failed")
    return {"code": 0, "message": "success"}
```

Create `app/main.py`:

```python
from fastapi import FastAPI

from app.api.routes import memories, messages
from app.dependencies import build_pipeline
from core.db import init_db, SessionLocal


app = FastAPI(title="Phase5 Memory Learning API")
app.include_router(memories.router, prefix="/api/v1", tags=["memories"])
app.include_router(messages.router, prefix="/api/v1", tags=["messages"])


@app.on_event("startup")
async def on_startup():
    await init_db()
    async with SessionLocal() as db:
        pipeline = build_pipeline(db)
        await pipeline.init_message_id_sequence()
        await pipeline.init_memory_size_cache()
```

Why this matches RAGFlow:

- Endpoint contract mirrors `/api/v1/memories` and `/api/v1/messages` family.
- Retrieval parameters are the same knobs used by RAGFlow.

---

### Step 7: Worker Loop

Create `core/worker.py`:

```python
import asyncio
import json
from redis.exceptions import ResponseError


async def run_worker(redis_client, pipeline):
    group = "rag_flow_svr_task_broker"
    consumer = "phase5-memory-worker"
    stream = "rag_flow_svr_queue"

    try:
        await redis_client.xgroup_create(name=stream, groupname=group, id="0", mkstream=True)
    except ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise

    while True:
        # First try pending entries for this consumer, then new ones.
        entries = await redis_client.xreadgroup(
            groupname=group,
            consumername=consumer,
            streams={stream: "0"},
            count=4,
            block=1000,
        )
        if not entries:
            entries = await redis_client.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream: ">"},
                count=4,
                block=5000,
            )
        if not entries:
            await asyncio.sleep(1)
            continue

        for _stream, items in entries:
            for msg_id, payload in items:
                try:
                    task = json.loads(payload["message"])
                    if task.get("task_type") == "memory":
                        await pipeline.handle_save_to_memory_task(task)
                    await redis_client.xack(stream, group, msg_id)
                except Exception:
                    # Keep message pending for retry/inspection.
                    continue
```

Create `core/worker_main.py`:

```python
import asyncio

from app.dependencies import redis_client, build_pipeline
from core.db import SessionLocal, init_db
from core.worker import run_worker


async def main():
    await init_db()
    async with SessionLocal() as db:
        pipeline = build_pipeline(db)
        await pipeline.init_message_id_sequence()
        await pipeline.init_memory_size_cache()
        await run_worker(redis_client, pipeline)


if __name__ == "__main__":
    asyncio.run(main())
```

Why this exists:

- Implements eventual extraction and keeps write-path latency low.
- Same architecture as `rag/svr/task_executor.py` memory task branch.

---

## End-to-End Runbook (Use Generated Seed Files)

0) Start API and worker

- Required env vars before start: `DOC_ENGINE=elasticsearch`, `ES_HOSTS`, `ES_USERNAME`, `ES_PASSWORD`, `POSTGRES_DSN`, `REDIS_URL`, `OPENAI_API_KEY`.
- API server: `uvicorn app.main:app --host 0.0.0.0 --port 9380 --reload`
- Worker: `python -m core.worker_main`

1) Create memory

- Raw-only fast path: `opencode/data/phase-5/memory_create_raw_only.json`
- Full extraction path: `opencode/data/phase-5/memory_create_multi_type.json`

2) Add messages

- Use `opencode/data/phase-5/messages_add_batch.json` (replace `<REPLACE_WITH_MEMORY_ID>`).

3) Start worker

- Run your phase worker (`core/worker.py`) so extraction tasks are consumed.

4) Query memory

- Use `opencode/data/phase-5/messages_search_requests.json`.

5) Validate recent/status/forget

- `messages_recent_request.json`
- `message_status_toggle_request.json`
- `forget_message_request.json`

6) Compare behavior

- Validate against `expected_search_results.json`.

---

## Validation Checklist

- RAW message appears immediately after `POST /messages`.
- Task row created for each non-RAW extraction flow.
- Extracted messages have `source_id = raw_message_id`.
- `GET /messages/search` returns semantically relevant memory hits.
- Disabled (`status=false`) and forgotten (`forget_at` set) messages stop appearing in search.
- FIFO eviction occurs when memory size limit is exceeded.

---

## Multi-Tenant Notes You Must Carry Into Your Production RFP System

1. Indexing strategy
   - Keep tenant-scoped index/table (`memory_{tenant_id}`) and memory-scoped subset (`memory_id`).

2. Model isolation
   - Resolve embedding/chat models per tenant, not globally.

3. Access control
   - Enforce tenant ownership + permission (`me|team`) on every memory endpoint.

4. Cross-memory retrieval
   - Support querying multiple memory IDs, but validate compatible embedding model families.

5. Operational safety
   - Keep async queue and progress tracking; do not make extraction fully synchronous in production.

---

## Fidelity and Simplifications

This guide stays very close to RAGFlow architecture and naming:

- Faithful: service boundaries, data flow, queue model, hybrid retrieval, memory types, prompt assembly, lifecycle fields.
- Simplified: some storage adapter details and task framework internals are trimmed for learnability.
- Recommended correction: ensure memory-size cache helper argument order is consistent in your implementation.

---

## Footnotes (Previous Phases)

1. Hybrid lexical+dense search internals build directly on Phase 1 retrieval concepts (`rag/nlp/search.py`, `rag/nlp/query.py`).
2. `agent_response` payloads in seed data intentionally resemble Phase 3 citation-style answers.
3. Some seed conversations include graph-level synthesis patterns from Phase 4, so you can test memory with graph-informed responses.
4. Existing phase docs for cross-reference:
   - `opencode/docs/PHASE3_GENERATION_WITH_CITATIONS.md`
   - `opencode/docs/PHASE4_GRAPHRAG_GUIDE.md`

---

## Post Script: Simplified Parts and Why

The guide intentionally simplifies a few implementation details so you can learn the architecture first and then harden it.

### Parity tag legend

- `high`: very close to RAGFlow behavior; mostly syntactic or ergonomic simplification.
- `medium`: core behavior is preserved, but important edge cases or guardrails are reduced.
- `low`: production-significant behavior is simplified; harden this before enterprise rollout.

| Parity Tag | Simplified Part | What It Does | How It Helps the System | How RAGFlow Does It | Why We Chose Simpler Path |
|---|---|---|---|---|---|
| medium | Lexical query builder (`MsgTextQuery`) | Builds BM25-friendly query text from user input | Improves recall for exact terms and phrases in memory search | Uses a richer pipeline in `memory/services/query.py`: term weighting, synonym expansion, Chinese fine-grained tokenization, phrase boosts | Easier to debug and reason about while learning hybrid retrieval flow |
| medium | Match expression representation | Represents text/dense/fusion queries as plain dicts in the learning code | Keeps search orchestration readable and backend-agnostic | Uses typed classes (`MatchTextExpr`, `MatchDenseExpr`, `FusionExpr`) translated by ES/Infinity adapters | Avoids abstraction depth early; focus stays on retrieval logic |
| medium | LLM extraction JSON parsing | Parses extraction output into typed memory records | Converts dialogue into semantic/episodic/procedural memory units | Uses prompt contracts + parsing utilities + task-progress integration (`memory_message_service.py`, `memory/utils/msg_util.py`) | Keeps extraction loop short for code-along implementation |
| medium | Worker runtime | Consumes memory tasks from Redis stream and runs extraction asynchronously | Keeps write latency low by offloading expensive extraction | `rag/svr/task_executor.py` additionally handles lock cleanup, richer retry controls, and broker-level metrics | Learning worker covers group creation + pending/new consumption with simpler failure policy |
| medium | Task lifecycle management | Tracks extraction task status and progress | Gives operational visibility and failure diagnostics | `TaskService` includes richer metadata and centralized update semantics used across task types | Learning repo keeps only fields needed for memory extraction while preserving progress semantics |
| medium | Message-store adapter depth | Maps logical message fields to physical index/table schema | Allows hybrid retrieval against ES/Infinity with consistent API | Separate adapters (`memory/utils/es_conn.py`, `memory/utils/infinity_conn.py`) handle field mapping, query translation, and backend-specific behavior | Reduces cognitive load while preserving conceptual boundaries |
| medium | Observability and usage accounting | Tracks model usage and traces extraction/retrieval operations | Supports cost control and production troubleshooting | `LLMBundle` updates tenant usage counters and can emit Langfuse traces | Non-essential for first end-to-end memory implementation |
| low | Authorization and permission enforcement | Ensures only allowed users/tenants can mutate/query memory | Critical for secure multi-tenant operation | API layer uses `login_required`, tenant scoping, permission checks (`me`/`team`) in SDK endpoints and services | Single-tenant learning mode lets you focus on memory mechanics first |
| medium | Memory update constraints | Prevents incompatible updates (e.g., embedding model/type changes on non-empty memory) | Protects vector consistency and retrieval quality | `api/apps/sdk/memories.py` blocks unsafe updates when memory already has content | Kept lighter so you can iterate quickly during learning |

### Suggested hardening order (after Phase 5 code-along)

1. [low] Add authz/permissions and tenant boundary tests.
2. [medium] Add richer task retry/reclaim strategy to match `task_executor` operational behavior.
3. [medium] Replace simplified lexical query path with full `memory/services/query.py` behavior.
4. [medium] Add usage tracing and operational dashboards.
5. [medium] Add memory update guards that block unsafe embedding-model changes on non-empty memories.
