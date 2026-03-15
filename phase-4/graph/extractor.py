import json
import logging
from graph.prompts import GRAPH_EXTRACTION_JSON_PROMPT
from llm.client import LLMClient
from models.schemas import GraphEntity, GraphRelation, GraphExtractionResult

logger = logging.getLogger(__name__)


EXTRACTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "graph_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity_name": {"type": "string"},
                            "entity_type": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["entity_name", "entity_type", "description"],
                        "additionalProperties": False,
                    },
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "src_id": {"type": "string"},
                            "tgt_id": {"type": "string"},
                            "description": {"type": "string"},
                            "strength": {"type": "number"},
                        },
                        "required": ["src_id", "tgt_id", "description", "strength"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["entities", "relations"],
            "additionalProperties": False,
        },
    },
}


class GraphExtractor:
    def __init__(self, llm: LLMClient, entity_types: list[str]) -> None:
        self.llm = llm
        self.entity_types = entity_types

    async def extract(self, text: str) -> GraphExtractionResult:
        """
        Extract entities + relations from a chunk. Mirrors
        graphrag/general/graph_extractor.py parsing flow.
        """

        system_prompt = GRAPH_EXTRACTION_JSON_PROMPT.format(
            entity_types=", ".join(self.entity_types)
        )

        preview = text.replace("\n", " ")[:120]
        logger.info(
            "🧠 Graph extraction started | chunk_len=%d | preview=%s",
            len(text),
            preview,
        )

        try:
            raw = await self.llm.chat(
                system_prompt=system_prompt,
                user_prompt=f"Text:\n{text}",
                response_format=EXTRACTION_RESPONSE_FORMAT,
            )
        except Exception as exc:
            logger.warning(
                "json_schema response_format unsupported/failed; falling back to json_object. error=%s",
                exc,
            )
            raw = await self.llm.chat(
                system_prompt=system_prompt,
                user_prompt=f"Text:\n{text}",
                response_format={"type": "json_object"},
            )
        logger.debug("📥 LLM raw extraction response:\n%s", raw)
        payload = json.loads(raw)

        entities: list[GraphEntity] = []
        relations: list[GraphRelation] = []

        for item in payload.get("entities", []):
            entities.append(
                GraphEntity(
                    entity_name=str(item["entity_name"]).strip(),
                    entity_type=str(item["entity_type"]).strip(),
                    description=str(item["description"]).strip(),
                )
            )

        for item in payload.get("relations", []):
            relations.append(
                GraphRelation(
                    src_id=str(item["src_id"]).strip(),
                    tgt_id=str(item["tgt_id"]).strip(),
                    description=str(item["description"]).strip(),
                    strength=float(item["strength"]),
                )
            )

        logger.info(
            "✅ Graph extraction parsed | entities=%d relations=%d",
            len(entities),
            len(relations),
        )
        logger.debug(
            "🧩 Entities parsed:\n%s",
            json.dumps(
                [e.model_dump() for e in entities], ensure_ascii=False, indent=2
            ),
        )
        logger.debug(
            "🔗 Relations parsed:\n%s",
            json.dumps(
                [r.model_dump() for r in relations], ensure_ascii=False, indent=2
            ),
        )

        return GraphExtractionResult(entities=entities, relations=relations)
