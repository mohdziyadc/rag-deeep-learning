# Generation Layer Design

This document summarizes the generation layer behavior at a high level and at a low level.

## High Level Flow (Conceptual)

```text
User Question + History
  |
  v
Retrieval (top_k chunks)
  |
  v
Prompt Build (system + context + citation rules)
  |
  v
Context Packing (fit system + recent history + user msg)
  |
  v
LLM Generation (stream or non-stream)
  |
  v
Citation Handling (we trust the LLM to generate answer with citations. But if it doesnt; if missing, insert via similarity)
  |
  v
Reference Bundle (used chunks + optional doc aggs)
  |
  v
Response to Client (answer + references)
```

## Low Level Flow (Function + Data Shape View)

### Inputs

- question: str
- messages: list[ChatMessage]
- top_k: int
- quote: bool
- max_tokens: int

### Steps

````text
1. RetrievalBridge.retrieve(question, top_k)
   - Output: chunks: list[RetrievedChunk]
   - RetrievedChunk = { chunk_id, doc_id, title, content, metadata }

2. PromptBuilder.build_system_prompt(chunks, quote)
   - 2.1 format_knowledge(chunks)
     - Trim by knowledge_budget (token-based)
     - Build a <context> block like:

       ```text
       <context>
       ID: i
       |-- Title: ...
       |-- URL: ...
       |-- metadata fields...
       `-- Content:
           <chunk.content>
       </context>
       ```

   - 2.2 system.md template injects {knowledge}
   - 2.3 If quote=True, append citation_prompt.md
   - Output: system_prompt: str

3. ContextPacker.fit_messages(system_prompt, messages + [user question])
   - Start with packed = [{"role": "system", "content": system_prompt}]
   - Compute remaining token budget
   - Keep most recent messages first
   - Drop older messages if overflow
   - Output: packed: list[dict{role, content}]

4. LLMClient.chat(packed) or stream_chat(packed)
   - Non-stream: returns full answer string
   - Stream: yields deltas; concatenated into full answer

5. Citation Inserter (post-gen)
   - Condition: quote=True AND chunks present AND answer lacks "[ID:"
   - 5.1 _split_sentences(answer) -> list[str]
   - 5.2 embedder.encode(sentences) -> sent_vecs (N x dim)
   - 5.3 embedder.encode(chunks.content) -> chunk_vecs (K x dim)
   - 5.4 For each sentence:
     - sims = cosine_similarity(chunk_vecs, svec) -> (K,)
     - top_ids = argsort(sims) desc, take max_citations
     - citations[i] = ids with sims[idx] > threshold
   - 5.5 Append " [ID:x]" to each sentence
   - 5.6 Track used_ids: set[int]
   - Else: used_ids may be empty (LLM handled citations)

6. Reference Builder
   - _build_reference(chunks, used_ids)
   - If used_ids present, keep only those chunks
   - Map to ReferenceChunk
   - Wrap in ReferenceBundle
   - Output: reference: ReferenceBundle

7. Response
   - ChatResponse = { answer, reference, prompt? }
````

### Streaming Special Case

- Stream deltas as they arrive
- After full answer, run citation insertion if needed
- Emit final event with reference bundle
