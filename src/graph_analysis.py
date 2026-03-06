from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

try:
    from .gnn_model import score_graph_with_gnn
except ImportError:
    score_graph_with_gnn = None


def build_transaction_graph(
    gst_csv: Path | None = None,
    related_party_csv: Path | None = None,
) -> nx.DiGraph:
    """
    Build a directed transaction graph from GST and related party ledgers.
    Nodes: entities (company/related parties/counterparties).
    Edges: aggregated transaction amounts.
    """
    G = nx.DiGraph()

    if gst_csv:
        gst_df = pd.read_csv(gst_csv)
        for _, row in gst_df.iterrows():
            src = row.get("gstin")
            dst = row.get("counterparty_gstin")
            amt = float(row.get("taxable_value", 0.0))
            if not src or not dst or amt == 0:
                continue
            if G.has_edge(src, dst):
                G[src][dst]["amount"] += amt
            else:
                G.add_edge(src, dst, amount=amt, source="gst")

    if related_party_csv:
        rp_df = pd.read_csv(related_party_csv)
        for _, row in rp_df.iterrows():
            src = "COMPANY"
            dst = row.get("counterparty_name")
            amt = float(row.get("amount", 0.0))
            if not dst or amt == 0:
                continue
            if G.has_edge(src, dst):
                G[src][dst]["amount"] += amt
            else:
                G.add_edge(src, dst, amount=amt, source="related_party")

    return G


def compute_graph_risk_scores(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute basic circular trading and concentration indicators from a transaction graph.
    This is a non-ML approximation of the GNN pillar.
    """
    if G.number_of_nodes() == 0:
        return {
            "graph_risk_score": 0.0,
            "graph_cycle_count": 0,
            "graph_max_centrality": 0.0,
            "graph_num_communities": 0,
        }

    # Simple cycle-based indicator
    simple_cycles = list(nx.simple_cycles(G))
    cycle_count = len(simple_cycles)
    example_cycles = [cycle for cycle in simple_cycles[:3]]

    # Centrality measure (in-degree)
    in_deg = nx.in_degree_centrality(G)
    max_centrality = max(in_deg.values()) if in_deg else 0.0
    top_central_entities = sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:3]

    # Community detection (weakly connected components as proxy)
    components = list(nx.weakly_connected_components(G))
    num_communities = len(components)

    # Risk score heuristic
    graph_risk_score = 0.0
    if cycle_count > 0:
        graph_risk_score += min(0.4, 0.05 * cycle_count)
    if max_centrality > 0.3:
        graph_risk_score += 0.3
    if num_communities == 1 and G.number_of_nodes() > 5:
        graph_risk_score += 0.2

    # Integrate PyTorch Geometric GNN predictions if available
    gnn_risk_score = 0.0
    high_risk_nodes = []
    if score_graph_with_gnn is not None:
        try:
            gnn_risk_score, high_risk_nodes = score_graph_with_gnn(G)
            graph_risk_score += (gnn_risk_score * 0.5) # weigh GNN output
        except Exception as e:
            print(f"Warning: GNN scoring failed: {e}")

    graph_risk_score = min(1.0, graph_risk_score)

    return {
        "graph_risk_score": float(graph_risk_score),
        "graph_cycle_count": int(cycle_count),
        "graph_max_centrality": float(max_centrality),
        "graph_num_communities": int(num_communities),
        "graph_example_cycles": example_cycles,
        "graph_top_central_entities": top_central_entities,
        "gnn_risk_score": float(gnn_risk_score),
        "gnn_high_risk_nodes": high_risk_nodes,
    }


def save_graph_image(G: nx.DiGraph, path: Path) -> None:
    """
    Save a simple visualization of the transaction graph.
    """
    if G.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6, 4))
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=50,
        arrows=False,
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()

