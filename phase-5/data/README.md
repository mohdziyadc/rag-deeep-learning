# Phase 5 Seed Data (Conversation Memory)

This folder gives you deterministic input/output fixtures so you can implement and test Phase 5 without re-implementing prior phases first.

## File list

- `seed_manifest.json`: canonical IDs and model defaults used by fixtures.
- `memory_create_raw_only.json`: create a raw-only memory.
- `memory_create_multi_type.json`: create a memory with raw + semantic + episodic + procedural extraction.
- `memory_update_small_fifo_limit.json`: shrink capacity to force FIFO behavior.
- `prior_phase_context_outputs.json`: synthetic outputs from earlier phases (citation-heavy + graph-aware answers).
- `messages_add_batch.json`: batch of conversation turns to ingest.
- `mock_llm_extraction_output.json`: deterministic extraction output if you run extractor in mock mode.
- `mock_queue_task_payload.json`: sample Redis stream message for worker testing.
- `messages_search_requests.json`: search request presets.
- `messages_recent_request.json`: recent message request preset.
- `message_status_toggle_request.json`: disable one message.
- `forget_message_request.json`: forget one message.
- `expected_search_results.json`: expected content anchors for retrieval validation.

## Usage order

1. Create memory using `memory_create_multi_type.json`.
2. Replace `<REPLACE_WITH_MEMORY_ID>` in request fixtures.
3. Post each payload in `messages_add_batch.json` to `/api/v1/messages`.
4. Run worker and monitor task progress.
5. Run queries from `messages_search_requests.json`.
6. Compare top results with `expected_search_results.json`.

## Mock extraction mode (optional)

If you want deterministic extraction (no chat-model dependency), load `mock_llm_extraction_output.json` in your `extract_by_llm` path when an env switch like `MEMORY_EXTRACTOR_MODE=mock` is enabled.
