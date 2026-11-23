"""Interactive graph of the ontology (or of the sub-graph used for one answer) with pyvis."""
from __future__ import annotations

from pyvis.network import Network

from rag.ontology import Ontology

EDGE_COLORS = {
    "isa": "#9aa0a6",
    "restriction": "#e8710a",
    "definition": "#d93025",
    "property": "#1a73e8",
    "instance": "#188038",
    "assertion": "#188038",
}


def graph_json(onto: Ontology, nodes=None, highlight=()) -> dict:
    """Nodes/edges for the custom web UI (rendered with vis-network in the browser)."""
    edges = onto.edges(nodes)
    used = set(nodes) if nodes is not None else set(onto.classes | onto.individuals)
    for s, t, _, _ in edges:
        used |= {s, t}
    highlight = set(highlight)
    return {
        "nodes": [{"id": str(u), "label": onto.name(u), "kind": onto.node_kind(u), "hit": u in highlight}
                  for u in used],
        "edges": [{"from": str(s), "to": str(t), "label": label, "kind": kind} for s, t, label, kind in edges],
    }


def build_html(onto: Ontology, nodes=None, highlight=(), height: str = "650px") -> str:
    net = Network(height=height, width="100%", directed=True, bgcolor="#ffffff", font_color="#202124")
    net.barnes_hut(gravity=-6000, spring_length=140)
    edges = onto.edges(nodes)
    used = set(nodes) if nodes is not None else set(onto.classes | onto.individuals)
    for s, t, _, _ in edges:
        used |= {s, t}
    highlight = set(highlight)
    for u in used:
        kind = onto.node_kind(u)
        color = "#fbbc04" if u in highlight else ("#34a853" if kind == "individual" else "#8ab4f8")
        net.add_node(str(u), label=onto.name(u), title=f"{kind}: {onto.name(u)}", color=color,
                     shape="box" if kind == "individual" else "ellipse", size=18 if u in highlight else 12)
    for s, t, label, kind in edges:
        net.add_edge(str(s), str(t), label=label, title=label, color=EDGE_COLORS[kind],
                     dashes=kind == "isa", arrows="to", font={"size": 9, "align": "middle"})
    return net.generate_html(notebook=False)
