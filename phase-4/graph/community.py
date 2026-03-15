import json
import networkx as nx
import community as community_louvain
from graph.prompts import COMMUNITY_REPORT_PROMPT
from llm.client import LLMClient


class CommunityReportBuilder:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def build_communities(self, graph: nx.Graph) -> dict[int, list[str]]:
        if len(graph) == 0:
            return {}

        partition = community_louvain.best_partition(graph=graph)
        communities: dict[int, list[str]] = {}
        for node, cid in partition.items():
            if cid not in communities:
                communities[cid] = []
            communities[cid].append(node)
        return communities

    async def build_report(
        self, graph: nx.Graph, nodes: list[str], weight: float
    ) -> dict:
        # weight = the weightage given to that specific community in the graph
        entities = CommunityReportBuilder.format_entities(graph, nodes)
        relations = CommunityReportBuilder.format_relations(graph, nodes)

        system_prompt = COMMUNITY_REPORT_PROMPT.format(
            entities=entities, relations=relations
        )

        response = await self.llm.chat(
            system_prompt=system_prompt,
            user_prompt="",
            response_format={"type": "json_object"},
        )
        try:
            report = json.loads(response)
        except json.JSONDecodeError:
            report = {
                "title": "Community Report",
                "summary": "Summary Unavailable due to parse error.",
                "findings": [],
            }

        report["weight"] = weight
        report["entities"] = nodes
        return report

    @staticmethod
    def format_entities(graph: nx.Graph, nodes: list[str]) -> str:
        lines = []
        for name in nodes:
            desc = graph.nodes[name].get("description", "")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def format_relations(graph: nx.Graph, nodes: list[str]) -> str:
        node_set = set(nodes)
        lines = []
        for src, tgt, data in graph.edges(data=True):
            if src in node_set and tgt in node_set:
                desc = data.get("description", "")
                lines.append(f"- {src} -> {tgt}: {desc}")
        return "\n".join(lines)
