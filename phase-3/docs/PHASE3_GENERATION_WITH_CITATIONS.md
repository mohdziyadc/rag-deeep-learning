# Phase 3: Generation With Citations (RAGFlow-Faithful)

## Goal

Build the **generation layer** that takes retrieved chunks from Phase 1 and the parsed content from Phase 2, then produces answers with **inline citations**. You will:

- Assemble system prompts using retrieved knowledge blocks.
- Fit multi-turn chat messages into a token budget safely.
- Stream LLM output (SSE) and handle non-stream responses.
- Insert citations post-generation when the model omits them.
- Return a response that includes answer + references so the UI can show sources.

No toy snippets, no pseudo-code. This guide gives end-to-end code and explains how each step mirrors RAGFlow.

---

## Table of Contents

1. The Big Picture
2. How RAGFlow Does It (Where To Look)
3. Phase 3 Architecture (rag-deep-learning/phase-3)
4. Step-by-Step Implementation
5. End-to-End Generation Flow
6. Testing Checklist
7. Phase 3 Outcomes (Ready for Phase 4)

---

## 1. The Big Picture

Generation is the **bridge** between retrieval and user answers. RAGFlow does this by:

- Building a system prompt that injects the knowledge chunks.
- Running the LLM with chat history.
- Adding citations either **during generation** or **afterwards**.

Pipeline view:

```
┌──────────┐   ┌────────────┐   ┌──────────────┐   ┌───────────┐   ┌───────────────┐
│  Query   │─▶│ Retrieval   │─▶│ Prompt Build │─▶│   LLM     │─▶│ Citation Insert │
│ (user)   │  │ (Phase 1)   │  │ + Context    │  │ Response  │  │ + References   │
└──────────┘   └────────────┘   └──────────────┘   └───────────┘   └───────────────┘
```

Key idea: **citations are a post-processing problem**, not just a prompt problem. RAGFlow inserts citations after generation when needed, using similarity between answer sentences and retrieved chunks.

---

## 2. How RAGFlow Does It (Where To Look)

These are the exact places where RAGFlow implements Phase 3:

- Prompt construction and citation prompt:
  - `rag/prompts/generator.py`
  - `rag/prompts/citation_prompt.md`
  - `rag/prompts/citation_plus.md`
- LLM chat wrapper + streaming:
  - `rag/llm/chat_model.py`
- Orchestration and response assembly:
  - `api/db/services/dialog_service.py`
- Citation insertion (vector + token similarity):
  - `rag/nlp/search.py` (`insert_citations`)

This guide mirrors those patterns in a minimal, learnable implementation.

---

## 3. Phase 3 Architecture (rag-deep-learning/phase-3)

Add a new generation layer to your learning project:

```
rag-deep-learning/phase-3/
├── app/
│   ├── main.py
│   ├── config.py
│   └── api/
│       └── routes/
│           └── chat.py
├── core/
│   ├── prompts/
│   │   ├── system.md
│   │   ├── citation_prompt.md
│   │   └── loader.py
│   ├── generation/
│   │   ├── prompt_builder.py
│   │   ├── context_packer.py
│   │   ├── llm_client.py
│   │   ├── citation_inserter.py
│   │   └── generator.py
│   └── retrieval/
│       └── bridge.py
├── models/
│   └── schemas.py
└── tests/
    └── test_generation.py
```

Notes:

- `core/retrieval/bridge.py` is a thin wrapper that calls your Phase 1 searcher and returns chunks.
- `citation_inserter.py` mirrors `rag/nlp/search.py`.
- `generator.py` mirrors `dialog_service.async_chat()`.

---

## 4. Step-by-Step Implementation

### Step 0: Dependencies

Add these to `rag-deep-learning/phase-3/pyproject.toml`:

```toml
[project]
dependencies = [
    # Web API
    "fastapi>=0.128.0",
    "uvicorn[standard]>=0.40.0",
    "pydantic>=2.12.5",
    "python-dotenv>=1.2.1",

    # LLM client
    "openai>=1.50.0",

    # Prompt templating
    "jinja2>=3.1.4",

    # Token counting (approximate)
    "tiktoken>=0.7.0",

    # Phase 1/2 deps
    "numpy>=1.26.0",
    "sentence-transformers>=3.0.0",
]
```

Why this matches RAGFlow:

- RAGFlow uses OpenAI-compatible chat APIs through adapters in `rag/llm/chat_model.py`.
- It uses prompt templates with Jinja and injects a citation instruction (`citation_prompt.md`).

---

### Step 1: Define Response Models

Create `models/schemas.py`:

```python
from pydantic import BaseModel, Field
from typing import Optional


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    title: str
    metadata: dict = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: str  # system | user | assistant
    content: str


class ChatRequest(BaseModel):
    question: str
    messages: list[ChatMessage] = Field(default_factory=list)
    top_k: int = 6
    stream: bool = True
    max_tokens: int = 4096
    quote: bool = True  # whether to include citations


class ReferenceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    content: str
    metadata: dict = Field(default_factory=dict)


class ReferenceBundle(BaseModel):
    chunks: list[ReferenceChunk]
    doc_aggs: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    reference: ReferenceBundle | None = None
    prompt: str | None = None
```

These mirror the structure returned by `dialog_service.decorate_answer()` in RAGFlow (answer + reference).

---

### Step 2: Prompt Templates

Create `core/prompts/system.md`:

```md
You are an intelligent assistant. Use the provided knowledge to answer the question.
If the knowledge does not contain the answer, respond with:
"The answer you are looking for is not found in the dataset!"

Here is the knowledge base:
{knowledge}
```

Create `core/prompts/citation_prompt.md` (RAGFlow style):

```md
Based on the provided document or chat history, add citations to the input text using the format specified later.

# Citation Requirements:

## Technical Rules:
- Use format: [ID:i] or [ID:i] [ID:j] for multiple sources
- Place citations at the end of sentences, before punctuation
- Maximum 4 citations per sentence
- DO NOT cite content not from <context></context>
- DO NOT modify whitespace or original text

## What MUST Be Cited:
1. Quantitative data
2. Temporal claims
3. Causal relationships
4. Comparative statements
5. Technical definitions
6. Direct attributions
7. Predictions/forecasts
8. Controversial claims
```

Why this matches RAGFlow:

- This is a reduced version of `rag/prompts/citation_prompt.md`.

---

### Step 3: Prompt Loader

Create `core/prompts/loader.py`:

```python
from pathlib import Path


PROMPT_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()
```

---

### Step 4: Prompt Builder + Context Packing

Create `core/generation/prompt_builder.py`:

```python
import re
import tiktoken
from jinja2 import Template
from core.prompts.loader import load_prompt
from models.schemas import RetrievedChunk


class PromptBuilder:
    def __init__(self, model_name: str, max_tokens: int, knowledge_budget_ratio: float = 0.7) -> None:
        self.system_template = Template(load_prompt("system"))
        self.citation_prompt = load_prompt("citation_prompt")
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.knowledge_budget = int(max_tokens * knowledge_budget_ratio)

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _draw_node(self, key: str, value: str | None) -> str | None:
        if value is None:
            return None
        text = re.sub(r"\n+", " ", str(value)).strip()
        if not text:
            return None
        return f"├── {key}: {text}"

    def format_knowledge(self, chunks: list[RetrievedChunk]) -> str:
        """
        Build a tree-style knowledge block like RAGFlow's kb_prompt:
        - Trim chunks by a token budget
        - Provide stable [ID:i] anchors for citations
        - Include title/URL/metadata when available
        """
        lines = ["<context>"]
        used_tokens = 0
        for i, c in enumerate(chunks):
            content = c.content or ""
            chunk_tokens = self._count_tokens(content)
            if used_tokens + chunk_tokens > self.knowledge_budget:
                break
            used_tokens += chunk_tokens

            lines.append(f"ID: {i}")
            node = self._draw_node("Title", c.title)
            if node:
                lines.append(node)
            url = c.metadata.get("url") if isinstance(c.metadata, dict) else None
            node = self._draw_node("URL", url)
            if node:
                lines.append(node)
            if isinstance(c.metadata, dict):
                for k, v in c.metadata.items():
                    if k in {"url", "title"}:
                        continue
                    node = self._draw_node(k, v)
                    if node:
                        lines.append(node)
            lines.append("└── Content:")
            lines.append(content)
        lines.append("</context>")
        return "\n".join(lines)

    def build_system_prompt(self, chunks: list[RetrievedChunk], quote: bool) -> str:
        """
        Assemble the final system prompt with knowledge injection and
        optional citation guidelines, matching RAGFlow's prompt layering.
        """
        knowledge = self.format_knowledge(chunks)
        base = self.system_template.render(knowledge=knowledge)
        if quote and chunks:
            return base + "\n\n" + self.citation_prompt
        return base
```

Why this mirrors RAGFlow's `kb_prompt`:

- Tree-style context formatting separates chunk metadata from content.
- Sequential `ID: i` anchors are stable for citations like `[ID:3]`.
- Knowledge is trimmed to a token budget before chat history is packed.

Tune `knowledge_budget_ratio` based on how much room you want for chat history vs. retrieval (0.6–0.8 is typical).

Create `core/generation/context_packer.py`:

```python
import tiktoken
from models.schemas import ChatMessage


class ContextPacker:
    def __init__(self, model_name: str, max_tokens: int) -> None:
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.max_tokens = max_tokens

    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def fit_messages(self, system_prompt: str, messages: list[ChatMessage]) -> list[dict]:
        """
        Trim chat history to fit a token budget while preserving the
        system prompt and the most recent turns, similar to RAGFlow's
        message_fit_in behavior.
        """
        packed = [{"role": "system", "content": system_prompt}]
        budget = self.max_tokens - self._count_tokens(system_prompt)

        # Keep latest messages first, then reverse back.
        reversed_msgs = list(reversed(messages))
        kept = []
        for msg in reversed_msgs:
            tokens = self._count_tokens(msg.content)
            if tokens > budget:
                continue
            kept.append(msg)
            budget -= tokens

        packed.extend({"role": m.role, "content": m.content} for m in reversed(kept))
        return packed
```

This matches RAGFlow's `message_fit_in` logic: keep the system prompt and as much recent history as possible.

---

### Step 5: LLM Client Wrapper

Create `core/generation/llm_client.py`:

```python
from openai import OpenAI


class LLMClient:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, messages: list[dict], temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def stream_chat(self, messages: list[dict], temperature: float = 0.2):
        """
        Stream tokens from the model and yield text deltas, matching
        RAGFlow's streaming path in chat_model.async_chat_streamly.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
```

---

### Step 6: Citation Inserter (Post-Generation)

Create `core/generation/citation_inserter.py`:

```python
import re
import numpy as np
from core.embedder import embedder  # from Phase 1


class CitationInserter:
    def __init__(self, max_citations_per_sentence: int = 4) -> None:
        self.max_citations = max_citations_per_sentence

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"([。？！.!?]\s)", text)
        if len(parts) <= 1:
            return [text]
        merged = []
        for i in range(0, len(parts) - 1, 2):
            merged.append(parts[i] + parts[i + 1])
        if len(parts) % 2 != 0:
            merged.append(parts[-1])
        return [s for s in merged if s.strip()]

    def insert(self, answer: str, chunks: list[str]) -> tuple[str, set[int]]:
        """
        Add citations by matching answer sentences to retrieved chunks
        via hybrid similarity, mirroring RAGFlow's insert_citations.
        """
        if not chunks:
            return answer, set()

        sentences = self._split_sentences(answer)
        embedder.load()
        sent_vecs, _ = embedder.encode(sentences)
        chunk_vecs, _ = embedder.encode(chunks)

        citations = {}
        for i, svec in enumerate(sent_vecs):
            sims = np.dot(chunk_vecs, svec) / (
                np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(svec) + 1e-8
            )
            top_ids = list(np.argsort(sims)[::-1])[: self.max_citations]
            citations[i] = [int(idx) for idx in top_ids if sims[idx] > 0.3]

        used = set()
        out = []
        for i, sentence in enumerate(sentences):
            out.append(sentence)
            if i in citations and citations[i]:
                for idx in citations[i]:
                    out.append(f" [ID:{idx}]")
                    used.add(idx)

        return "".join(out), used
```

Notes:

- This mirrors `rag/nlp/search.py::insert_citations`, but simplified.
- It uses cosine similarity between answer sentences and chunks.

---

### Step 7: Generator Orchestrator

Create `core/generation/generator.py`:

```python
from core.generation.prompt_builder import PromptBuilder
from core.generation.context_packer import ContextPacker
from core.generation.llm_client import LLMClient
from core.generation.citation_inserter import CitationInserter
from models.schemas import ChatMessage, ChatResponse, RetrievedChunk, ReferenceBundle, ReferenceChunk


class Generator:
    def __init__(self, llm: LLMClient, model_name: str, max_tokens: int) -> None:
        self.prompts = PromptBuilder(model_name=model_name, max_tokens=max_tokens)
        self.packer = ContextPacker(model_name=model_name, max_tokens=max_tokens)
        self.llm = llm
        self.citer = CitationInserter()

    def _build_reference(self, chunks: list[RetrievedChunk], used_ids: set[int]) -> ReferenceBundle:
        refs = []
        for i, c in enumerate(chunks):
            if used_ids and i not in used_ids:
                continue
            refs.append(
                ReferenceChunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    title=c.title,
                    content=c.content,
                    metadata=c.metadata,
                )
            )
        return ReferenceBundle(chunks=refs, doc_aggs=[])

    def generate(self, question: str, messages: list[ChatMessage], chunks: list[RetrievedChunk], quote: bool) -> ChatResponse:
        """
        End-to-end generation: build prompt, fit context, call LLM,
        then insert citations and return references.
        """
        system_prompt = self.prompts.build_system_prompt(chunks, quote=quote)
        packed = self.packer.fit_messages(system_prompt, messages + [ChatMessage(role="user", content=question)])

        answer = self.llm.chat(packed)
        used = set()
        if quote and chunks and "[ID:" not in answer:
            answer, used = self.citer.insert(answer, [c.content for c in chunks])

        reference = self._build_reference(chunks, used)
        return ChatResponse(answer=answer, reference=reference, prompt=system_prompt)

    def stream(self, question: str, messages: list[ChatMessage], chunks: list[RetrievedChunk], quote: bool):
        """
        Stream generation while preserving the same prompt and
        citation logic as non-streaming responses.
        """
        system_prompt = self.prompts.build_system_prompt(chunks, quote=quote)
        packed = self.packer.fit_messages(system_prompt, messages + [ChatMessage(role="user", content=question)])

        full = ""
        for delta in self.llm.stream_chat(packed):
            full += delta
            yield {"type": "delta", "data": delta}

        used = set()
        if quote and chunks and "[ID:" not in full:
            full, used = self.citer.insert(full, [c.content for c in chunks])

        reference = self._build_reference(chunks, used)
        yield {
            "type": "final",
            "data": ChatResponse(answer=full, reference=reference, prompt=system_prompt).model_dump(),
        }
```

---

### Step 8: Retrieval Bridge

Create `core/retrieval/bridge.py`:

```python
from core.searcher import searcher  # Phase 1 searcher
from models.schemas import RetrievedChunk


class RetrievalBridge:
    def __init__(self) -> None:
        self.searcher = searcher

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        results, _ = await self.searcher.search(query, top_k=top_k)
        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                content=r.content,
                title=r.title,
                metadata={},
            )
            for r in results
        ]
```

---

### Step 9: API Endpoint (Streaming + Non-Streaming)

Create `app/config.py`:

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_api_key: str
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    max_tokens: int = 4096

    class Config:
        env_file = ".env"


def get_settings() -> Settings:
    return Settings()
```

Create `app/api/routes/chat.py`:

```python
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.config import get_settings
from core.generation.llm_client import LLMClient
from core.generation.generator import Generator
from core.retrieval.bridge import RetrievalBridge
from models.schemas import ChatRequest


router = APIRouter()
settings = get_settings()
llm = LLMClient(settings.llm_api_key, settings.llm_base_url, settings.llm_model)
generator = Generator(llm=llm, model_name=settings.llm_model, max_tokens=settings.max_tokens)
retriever = RetrievalBridge()


@router.post("/chat")
async def chat(req: ChatRequest):
    chunks = await retriever.retrieve(req.question, top_k=req.top_k)

    if req.stream:
        async def event_stream():
            for event in generator.stream(req.question, req.messages, chunks, quote=req.quote):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    response = generator.generate(req.question, req.messages, chunks, quote=req.quote)
    return response
```

Create `app/main.py`:

```python
from fastapi import FastAPI
from app.api.routes import chat


app = FastAPI(title="Phase 3 Generation")
app.include_router(chat.router, prefix="/api", tags=["chat"])
```

---

## 5. End-to-End Generation Flow

1. Client sends `/api/chat` with question + history.
2. Retrieval bridge calls Phase 1 searcher to get top chunks.
3. Prompt builder formats knowledge as `<context> ... </context>`.
4. Context packer trims history to token budget.
5. LLM generates answer (stream or non-stream).
6. Citation inserter adds `[ID:x]` when missing.
7. Response includes answer + reference chunks.

This maps directly to `dialog_service.async_chat()` in RAGFlow.

---

## 6. Testing Checklist

- Send a query that matches known chunks; verify citations appear.
- Send a query with no relevant results; verify the empty response text.
- Stream a response and verify the final event includes `reference`.
- Force token overflow by sending long history; verify earlier turns are trimmed.
- Ensure citations use `[ID:x]` format and appear at sentence end.

---

## 7. Phase 3 Outcomes (Ready for Phase 4)

After this phase, you understand:

1. How RAGFlow builds prompts with knowledge injection.
2. How context packing prevents token overflow.
3. How citations are inserted post-generation using similarity.
4. How responses include structured references for the UI.
5. How to run streaming and non-streaming chat endpoints.

This completes Phase 3 and prepares you for Phase 4 (agentic workflows and graph reasoning).
