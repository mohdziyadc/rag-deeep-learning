

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
        entities: list[dict], 
        relations: list[dict], 
        communities: list[dict]
    ) -> str:
        entity_block = ["---- Entities ----"]
        relation_block = ["---- Relations ----"]
        community_block = ["---- Community Reports ----"]

        for entity in entities:
            desc = json.loads(entity.get("content_with_weight", "{}").get("description", ""))
            entity_block.append(f"Entity: {entity.get("entity_kwd", "-")}")
            entity_block.append(f"Type: {entity.get('entity_type_kwd', '-')}")
            entity_block.append(f"Description: {desc}")
        
        for rel in relations:
            desc = json.loads(entity.get("content_with_weight", "{}").get("description", ""))
            relation_block.append(f"From: {rel.get('from_entity_kwd', '-')}")
            relation_block.append(f"To: {rel.get('to_entity_kwd', '-')}")
            relation_block.append(f"Description: {desc}")
        
        for comm in communities:
            desc_obj = json.loads(comm.get("content_with_weight", "{}") or "{}")
            community_block.append(f"Report: {desc_obj.get("report", "")}")
            community_block.append(f"Evidences: {desc_obj.get("evidences", "")}")
        
        return '\n'.join(entity_block + [""] + relation_block + [""] + community_block)

    
    async def query(self, kb_id: str, q: GraphQuery) -> str:
        """
        kb_id doesnt make sense for me here. It's a user inputted field
        it seems and used for routing in APP. THink user selecting a bucket
        like "Security, HR, FINANCE" etc. Ideally a router should be in place
        to understand which KBs to use. Router implementation think peice down below
        """

        vector = self.embedder.embed([q.question])[0].tolist()
        entities = await self.store.search_entities(kb_id, vector, q.top_entities)
        relations = await self.store.search_relations(kb_id, vector, q.top_relations)

        entity_names = [e.get("entity_kwd", "") for e in entities]
        communities = await self.store.search_community_reports(kb_id, entity_names, q.top_communities)
        """
        This phase ends with this retrieved prompt generated. 
        It's not being used downstream for LLM to make decisions
        """
        return GraphSearcher.format_graph_context(entities=entities, relations=relations, communities=communities)


    




"""
Yes, a router can be an LLM call—but it doesn't have to be. There are three common routing strategies:
1) Rule + metadata router (no LLM)  
   - Use keyword lists or a lightweight classifier to map to KBs.  
   - Fast, deterministic, cheap, but brittle.
2) Embedding similarity router (no LLM)  
   - Embed the question and compare to precomputed “KB centroids” (vector per KB).  
   - Route to top-k KBs by cosine similarity.  
   - Robust and cheap after setup.
3) LLM router (LLM call)  
   - Prompt the LLM to pick the most relevant KB(s) from a list and justify.  
   - Accurate for nuanced questions, but slower/costly and can be inconsistent.
Often the best approach is a hybrid:
- Do embedding routing to shortlist (e.g., top 3 KBs),  
- Then use a small LLM to pick among them (or skip the LLM if the top score is high).
"""