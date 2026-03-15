# Phase 4 Data Prerequisites

This folder provides standalone data so you can build and test GraphRAG
without depending on Phase 1 or Phase 2.

## Files

- `chunks.json`: Minimal chunk dataset to build a knowledge graph.
- `entity_types.json`: Entity types used by the graph extractor.
- `build_request.json`: Example payload for `/api/graphrag/build` (chunk objects).
- `query_request.json`: Example payload for `/api/graphrag/query` (includes kb_id).
- `query_request_security.json`: Security-focused query payload.
- `query_request_compliance.json`: Compliance-focused query payload.
- `query_request_sla.json`: SLA/uptime-focused query payload.
- `query_request_pricing.json`: Pricing-focused query payload.
- `query_request_architecture.json`: Architecture/deployment-focused query payload.

## How To Use

1) Build the graph (requires Elasticsearch, see guide). Use `kb_id` in the payload:

```
POST /api/graphrag/build
body = build_request.json
```

2) Query the graph:

```
POST /api/graphrag/query
body = query_request.json
```

## Notes

- `chunks.json` simulates parsed content from multiple documents.
- The chunks are intentionally cross-linked so you get non-trivial entities and relations.
