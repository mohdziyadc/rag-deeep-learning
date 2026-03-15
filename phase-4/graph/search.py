import json
from graph.embedder import Embedder
from graph.store import GraphDocStore
from models.schemas import GraphQuery


class GraphSearcher:
    def __init__(self, store: GraphDocStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    @staticmethod
    def format_graph_context(
        entities: list[dict], relations: list[dict], communities: list[dict]
    ) -> str:
        entity_block = ["---- Entities ----"]
        relation_block = ["---- Relations ----"]
        community_block = ["---- Community Reports ----"]

        for entity in entities:
            desc = json.loads(entity.get("content_with_weight", "{}") or "{}").get(
                "description", ""
            )
            entity_block.append(f"Entity: {entity.get('entity_kwd', '-')}")
            entity_block.append(f"Type: {entity.get('entity_type_kwd', '-')}")
            entity_block.append(f"Description: {desc}")

        for rel in relations:
            desc = json.loads(rel.get("content_with_weight", "{}") or "{}").get(
                "description", ""
            )
            relation_block.append(f"From: {rel.get('from_entity_kwd', '-')}")
            relation_block.append(f"To: {rel.get('to_entity_kwd', '-')}")
            relation_block.append(f"Description: {desc}")

        for comm in communities:
            desc_obj = json.loads(comm.get("content_with_weight", "{}") or "{}")
            community_block.append(f"Report: {desc_obj.get('report', '')}")
            community_block.append(f"Evidences: {desc_obj.get('evidences', '')}")

        return "\n".join(entity_block + [""] + relation_block + [""] + community_block)

    async def query(self, kb_id: str, q: GraphQuery) -> str:
        """
        kb_id is used for routing/scope in retrieval. In UI-driven systems this is
        usually selected by session context; later we can replace this with a
        router that chooses KBs automatically.
        """
        await self.store.connect()

        vector = self.embedder.embed([q.question])[0].tolist()
        entities = await self.store.search_entities(kb_id, vector, q.top_entities)
        relations = await self.store.search_relations(kb_id, vector, q.top_relations)

        entity_names = [e.get("entity_kwd", "") for e in entities]
        communities = await self.store.search_community_reports(
            kb_id, entity_names, q.top_communities
        )
        """
        This phase returns the retrieved graph context block only.
        Final downstream LLM answer generation is not wired in this phase yet.
        """
        return GraphSearcher.format_graph_context(
            entities=entities, relations=relations, communities=communities
        )


"""
Router note for future iterations:
1) Rule + metadata router (no LLM)
2) Embedding similarity router (no LLM)
3) LLM router

Hybrid pattern: shortlist with embeddings, optionally finalize with LLM.
"""
