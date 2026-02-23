# Phase 3 Learning Context (Conversation Digest)

This document captures the questions, intent, and learning style from building Phase 3 (generation with citations). It is meant to guide future agents to create a plan tailored to the same deep, production-minded learning approach.

## Purpose

- Provide a compact memory of how Phase 3 was understood end to end.
- Preserve the types of questions asked and the specific insights sought.
- Serve as a reusable context doc for planning future phases (e.g., agentic RAG).

## Learning Intent and Style

- Goal: understand RAG generation deeply (prompt construction, token budgets, citations, streaming).
- Preference: ask "why" at each step and compare to production RAGFlow patterns.
- Strong preference for concrete examples, line-by-line walkthroughs, and diagrams.
- Wants simplified mental models plus low-level execution details.

## Question Themes Asked (Grouped)

### Architecture and Integration

- Is Phase 3 integrating an LLM? Where is it wired in?
- Where does this generation layer fit in an agentic RAG system later?
- Is prompt packing for refinement or just for token budget?

### Data Models and Attribution

- Why do we need ReferenceChunk and ReferenceBundle if they mirror RetrievedChunk?
- What is doc_aggs and why return references at all?
- Where does server-side attribution happen? How are used_ids determined?

### Prompt Construction and Budgeting

- Why use Jinja2? What does it do?
- Why RAGFlow uses tree-style prompt formatting?
- Why not just dump chunks? Why add IDs and metadata?
- What is knowledge_budget and why cut off chunks?
- Do LLMs already have compaction? Should we prefer compaction or cutoff?

### Streaming and API Behavior

- What does yield do in streaming?
- Why yield a final event instead of returning?
- What does the client receive from SSE? JSON events or plain text?
- Why use JSON deltas instead of raw text?

### Citation Insertion Algorithm

- How does sentence splitting work and why split/merge punctuation?
- What are sent_vecs and chunk_vecs?
- Is this cosine similarity? Why add 1e-8?
- How are top_ids selected? What does citations[i] contain?
- How does this compare to RAGFlow (hybrid similarity with rag_tokenizer)?

## Core Insights Captured

- Phase 3 is where the LLM is connected: prompt build -> LLM -> citations -> references.
- Prompt building is not just formatting; it controls grounding and citation stability.
- RAGFlow uses a tree-style context block with stable IDs, metadata, and optional hashing.
- Knowledge budget protects the system prompt and user query from context overflow.
- Context packing keeps the system prompt and recent turns; older history is trimmed first.
- LLM citations are trusted if present; otherwise citations are inserted post-generation.
- Citation insertion compares each answer sentence to chunk content via embeddings.
- RAGFlow uses hybrid similarity (vector + token overlap), not pure cosine similarity.
- Streaming returns SSE events; deltas are yielded and a final structured payload is sent.

## Implementation Decisions Captured

- Added a Makefile for `make dev` and `make start` to avoid typing full uvicorn commands.
- Updated the guide to mirror RAGFlow-style prompt building with tree formatting and budgets.
- Added a high-level + low-level generation layer design document for reference.
- Chose JSON SSE events (delta + final) for streaming so the UI can append text and then render citations.

## Practical Preferences for Future Phases

- Prefer explicit budget control over opaque compaction by default.
- Prefer diagrams + high-level and low-level algorithm walkthroughs for each new phase.
- Prefer production-comparable patterns (RAGFlow) over toy examples.
- Prefer a step-by-step explanation of any non-trivial regex or similarity logic.

## Notes on Debugging and Transparency

- Returning `prompt` in the response is useful for debugging and learning; may be removed in production.
- ReferenceBundle acts as server-side attribution for citations, not the LLM itself.

## Agent Guidance for Future Phase Planning

When planning a new phase for this learner:

- Start with a high-level diagram, then a low-level data-flow breakdown.
- Explain each file and function with "what" + "why" + "where in code".
- Compare to RAGFlow’s production pattern and note simplifications.
- Provide concrete examples (inputs, intermediate outputs, final outputs).
- Explicitly call out tradeoffs (simplicity vs production robustness).

## Target Direction (Next Phases)

- Long-term goal: agentic RAG for RFP filling.
- Keep Phase 3 generation layer as the grounded, citation-enforcing output stage.
