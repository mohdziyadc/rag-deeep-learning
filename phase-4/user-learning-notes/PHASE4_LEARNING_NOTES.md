# Phase 4 Learning Notes (GraphRAG)

## 1) What We Built in Phase 4

This phase implemented a simplified but real GraphRAG pipeline end-to-end:

1. **Schemas** for graph objects (`GraphEntity`, `GraphRelation`, `GraphExtractionResult`, `CommunityReport`, `GraphQuery`).
2. **LLM extraction** from chunk text into entities/relations.
3. **Subgraph creation** per source document (`doc_id`).
4. **Graph merge** into a global KB graph.
5. **Entity resolution** (simplified merge heuristics).
6. **PageRank** over graph nodes.
7. **Indexing into ES** as:
   - `graph`
   - `subgraph`
   - `entity`
   - `relation`
   - `community_report`
8. **Graph query retrieval** (`/api/graphrag/query`) returning `graph_context`.
9. **Observability additions**:
   - startup checks
   - stage-level logs
   - per-step timing
   - persisted retrieval prompts in `retrieval-prompts/`

---

## 2) Core Concepts You Clarified Really Well

You asked the right fundamentals repeatedly until they were clear:

- **Entities vs relations**
  - entities = nodes
  - relations = edges
- **Community reports**
  - summaries of graph clusters (Louvain in guide, Leiden in RAGFlow)
- **Provenance fields**
  - `source_id` tracks which document/chunk produced a node/edge
- **Graph hierarchy**
  - subgraph (per-doc) -> merged global graph -> entity/relation docs for retrieval
- **Where LLM is used in this phase**
  - extraction + community reporting
  - query retrieval itself is not final generation
- **What output of this phase means**
  - this phase ends at "graph context generation", not full answer generation

---

## 3) Architecture Understanding Gained (Guide vs RAGFlow)

You dug into design differences deeply:

- **Guide**: learning-first, simplified, explicit mappings, fewer control loops.
- **RAGFlow**: production-first, dynamic template mappings, richer retrieval/ranking, incremental graph maintenance.

Specific architecture insights you established:

- RAGFlow stores many doc types in a **tenant index** and uses field-name conventions for mapping.
- `kb_id` is a logical dataset router within tenant scope (not tenant ID itself).
- Single-index vs multi-index tradeoffs depend on ops simplicity vs isolation/perf tuning.

---

## 4) Implementation Lessons from Real Bugs

This was the most valuable part of learning:

1. **Async correctness**
   - sync/async mismatch in extractor/community caused hidden issues.
   - fixed with proper `await` chain.

2. **Prompt plumbing matters more than prompt text quality**
   - missing real-data injection produced good-looking instructions but no valid extraction.

3. **Delimiter parsing is fragile**
   - tiny format drift (`|>` vs `<|>`) collapsed parsing.
   - moving to structured JSON output reduced fragility.

4. **Regex/match object pitfalls**
   - `match.group(1)` vs `match` confusion caused parse failures.

5. **NetworkX contraction metadata**
   - `contracted_nodes` adds `contraction` with tuple keys -> JSON serialization failure.
   - cleanup needed before indexing snapshots.

6. **Graph subgraph lifecycle mismatch**
   - merge-time subgraph upserts + later delete pass produced confusing counts.

7. **Client/server version alignment**
   - ES Python client major mismatch can produce weird API errors.

---

## 5) How You Learn Best (Observed Pattern)

Your strongest learning mode is **mechanistic understanding through precise questions**.

You consistently ask:

- "What exactly does this line do?"
- "Where is this set upstream?"
- "What is this in SQL terms?"
- "How does RAGFlow do this in production?"
- "What breaks if we remove this?"

This means your best study format is:

1. **Code snippet**
2. **Concrete example input/output**
3. **Mental model diagram**
4. **Failure mode + why**
5. **Production counterpart (RAGFlow)**

You are not learning performatively; your questions show deep causal inquiry.

---

## 6) Documentation Style to Use for other Phases

For each Phase module, document using this template:

1. **What is it?**
2. **Why it exists?**
3. **Where called from?**
4. **Input schema + output schema**
5. **Step-by-step flow with one real example**
6. **Failure modes and logs to inspect**
7. **How RAGFlow does it vs your simplified version**

Keep every section short but concrete; prefer examples over abstract theory.

---

## 7) Phase 4 Success Criteria (Met)

- Build API runs end-to-end and produces non-zero entities/relations.
- Query API returns graph context.
- ES stores expected GraphRAG docs.
- Logs and timing provide actionable observability.

Known acceptable simplifications:

- compacting/scoring is not fully RAGFlow-grade yet
- subgraph lifecycle is simplified
- community report latency is high (LLM-heavy)

---

## 8) Carry-Forward Into Phase 5

As you move into memory system work, keep these principles from Phase 4:

- prefer schema-constrained outputs where possible (Tell the user how ragflow does it, and why we are constraining the output, if applicable)
- keep provenance explicit
- log every major stage and timings
- separate "retrieval context generation" from "final answer generation"
- always compare your simplified design to production behavior in RAGFlow

These notes are intentionally written as a working reference for your Phase 5 docs.
