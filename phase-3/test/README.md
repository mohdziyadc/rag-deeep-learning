# Phase 3 Manual Test Artifacts

This folder contains mock inputs, retrieved chunks, and expected prompts/answers
to manually test the Phase 3 generation flow without Elasticsearch.

## What You Get

- `mock_chunks.json`: The canonical chunk pool used in retrieval mocks.
- `requests/`: Example `ChatRequest` payloads.
- `retrieval/`: The exact chunks (ordered) to feed into the generator.
- `prompts/`: Fully rendered system prompts that should go to the LLM.
- `outputs/`: Example answers with citations for comparison.

## How To Use

1) Pick a request in `requests/` (e.g., `requests/case_01_request.json`).
2) Load the corresponding retrieved chunks in `retrieval/`.
3) Build the system prompt using:
   - `core/prompts/system.md`
   - `core/prompts/citation_prompt.md`
   - The retrieval list in the given order
4) Compare your rendered system prompt with `prompts/`.
5) Run generation and compare the final answer with `outputs/`.

## Important Notes

- Citation indices are based on the order of the retrieved chunk list.
- These prompts assume `quote=true` and at least one chunk present.
- The expected outputs are samples; your model may differ slightly, but
  citations should reference the same chunk IDs.
