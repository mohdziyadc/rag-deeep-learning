import re
import networkx as nx


def should_merge(a: str, b: str) -> bool:
    a_norm = re.sub(r"\W+", "", a.lower())
    b_norm = re.sub(r"\W+", "", b.lower())

    if a_norm == b_norm:
        return True
    if a_norm in b_norm or b_norm in a_norm:
        return True
    return False


def resolve_entities(graph: nx.Graph) -> nx.Graph:
    nodes = list(graph.nodes())
    merged = graph.copy()

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[i] not in merged or nodes[j] not in merged:
                continue
            if should_merge(nodes[i], nodes[j]):
                nx.contracted_nodes(
                    merged, nodes[i], nodes[j], self_loops=False, copy=False
                )

    # networkx.contracted_nodes adds a "contraction" attribute containing
    # nested mappings with tuple keys, which are not JSON serializable.
    for _, data in merged.nodes(data=True):
        data.pop("contraction", None)
    for _, _, data in merged.edges(data=True):
        data.pop("contraction", None)

    return merged
