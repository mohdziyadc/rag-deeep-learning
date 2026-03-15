import networkx as nx

GRAPH_FIELD_SEP = "<SEP>"

"""
Example to make it clear:

import networkx as nx
G = nx.Graph()
G.add_node("SOC 2", entity_type="standard", description="SOC2 compliance", source_id=["doc_1"])
G.add_node("Audit", entity_type="process", description="Annual audit", source_id=["doc_2"])
G.add_node("Security Team", entity_type="org_unit", description="Owns audits", source_id=["doc_2"])
G.add_node("Policy A", entity_type="policy", description="Defines controls", source_id=["doc_1"])
G.add_edge("SOC 2", "Audit", description="requires", weight=0.8, source_id=["doc_2"])
G.add_edge("Audit", "Security Team", description="managed by", weight=0.9, source_id=["doc_2"])
G.add_edge("Policy A", "SOC 2", description="aligns with", weight=0.6, source_id=["doc_1"])
Nodes with attributes
list(G.nodes(data=True))
# [
#   ("SOC 2", {"entity_type":"standard","description":"SOC2 compliance","source_id":["doc_1"]}),
#   ("Audit", {"entity_type":"process","description":"Annual audit","source_id":["doc_2"]}),
#   ("Security Team", {"entity_type":"org_unit","description":"Owns audits","source_id":["doc_2"]}),
#   ("Policy A", {"entity_type":"policy","description":"Defines controls","source_id":["doc_1"]})
# ]
Relations (edges) with attributes
list(G.edges(data=True))
# [
#   ("SOC 2", "Audit", {"description":"requires","weight":0.8,"source_id":["doc_2"]}),
#   ("SOC 2", "Policy A", {"description":"aligns with","weight":0.6,"source_id":["doc_1"]}),
#   ("Audit", "Security Team", {"description":"managed by","weight":0.9,"source_id":["doc_2"]})
# ]
Degree = number of edges a node has (undirected graph)
list(G.degree)
# [("SOC 2", 2), ("Audit", 2), ("Security Team", 1), ("Policy A", 1)]
Visual sketch:
[Policy A] --(aligns with)--> [SOC 2] --(requires)--> [Audit] --(managed by)--> [Security Team]
"""


def graph_merge(base: nx.Graph, subgraph: nx.Graph) -> nx.Graph:
    for node_name, attr in subgraph.nodes(data=True):
        if not base.has_node(node_name):
            base.add_node(node_name, **attr)
            continue

        node = base.nodes[node_name]
        node["description"] += GRAPH_FIELD_SEP + attr["description"]
        node["source_id"] += attr["source_id"]

    for source, target, attr in subgraph.edges(data=True):
        edge = base.get_edge_data(source, target)
        if edge is None:
            base.add_edge(source, target, **attr)
            continue
        edge["weight"] = edge.get("weight", 0) + attr.get("weight", 0)
        edge["description"] += GRAPH_FIELD_SEP + attr["description"]
        edge["source_id"] += attr["source_id"]

    for node_degree in base.degree:
        base.nodes[str(node_degree[0])]["rank"] = int(node_degree[1])

    base.graph.setdefault("source_id", [])
    """
    Example:
    subgraph.graph == {
        "source_id": ["doc_123"]
    }
    """
    base.graph["source_id"] += subgraph.graph.get("source_id", [])
    return base
