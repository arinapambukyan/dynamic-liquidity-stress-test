"""
Plot the indirect common-asset contagion network.

The indirect network is based on liquidity-weighted overlap in government
securities holdings. Nodes represent banks, and edges represent exposure to
the same state-bond market.

Run from project root:

    PYTHONPATH=src python scripts/50_plot_indirect_contagion_network.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from liquidity_stress.config import BANKING_INPUT_PATH
from liquidity_stress.data_loader import BankingDataLoader
from liquidity_stress.network_overlap import (
    compute_overlap_matrix,
    build_overlap_edge_table,
    build_overlap_node_table,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "charts"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


def scale_series(values: pd.Series, min_size: float, max_size: float) -> pd.Series:
    x = values.astype(float)

    if np.isclose(x.max(), x.min()):
        return pd.Series(
            np.full(len(x), (min_size + max_size) / 2.0),
            index=x.index,
        )

    return min_size + (x - x.min()) * (max_size - min_size) / (x.max() - x.min())


def compute_external_label_positions(
    pos: dict,
    expansion_factor: float = 1.25,
) -> dict:
    """
    Move labels outside the node cloud.
    """
    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])

    center_x = xs.mean()
    center_y = ys.mean()

    label_pos = {}

    for node, (x, y) in pos.items():
        dx = x - center_x
        dy = y - center_y

        norm = np.sqrt(dx**2 + dy**2)
        if norm < 1e-9:
            dx, dy = 1.0, 0.0

        label_pos[node] = (
            center_x + expansion_factor * dx,
            center_y + expansion_factor * dy,
        )

    return label_pos


def undirected_key(u: str, v: str) -> tuple[str, str]:
    return tuple(sorted((u, v)))


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    banks = BankingDataLoader(BANKING_INPUT_PATH).load_banks()

    omega = compute_overlap_matrix(banks=banks)
    edges = build_overlap_edge_table(omega=omega, min_overlap=0.0)
    nodes = build_overlap_node_table(banks=banks, omega=omega)

    omega.to_csv(TABLES_DIR / "indirect_contagion_overlap_matrix.csv")
    edges.to_csv(TABLES_DIR / "indirect_contagion_edge_table.csv", index=False)
    nodes.to_csv(TABLES_DIR / "indirect_contagion_node_table.csv", index=False)

    graph = nx.Graph()

    for _, row in nodes.iterrows():
        graph.add_node(
            row["bank_id"],
            bank_name=row["bank_name"],
            total_assets=row["total_assets"],
            systemic_importance=row["systemic_importance"],
        )

    for _, row in edges.iterrows():
        graph.add_edge(
            row["source"],
            row["target"],
            overlap=row["overlap"],
            log_overlap=row["log_overlap"],
        )

    pos = nx.spring_layout(graph, seed=42, k=1.25, iterations=400)
    label_pos = compute_external_label_positions(pos, expansion_factor=1.28)

    node_sizes = scale_series(
        nodes.set_index("bank_id")["total_assets"],
        min_size=850,
        max_size=4300,
    )

    importance_series = nodes.set_index("bank_id")["systemic_importance"]
    node_color_values = [importance_series.loc[node] for node in graph.nodes()]

    top3_nodes = (
        nodes.sort_values("systemic_importance", ascending=False)
        .head(3)["bank_id"]
        .tolist()
    )

    edge_log = edges["log_overlap"] if len(edges) > 0 else pd.Series([1.0])
    min_w, max_w = 0.15, 5.8

    if len(edge_log) > 0 and not np.isclose(edge_log.max(), edge_log.min()):
        width_map = min_w + (edge_log - edge_log.min()) * (max_w - min_w) / (
            edge_log.max() - edge_log.min()
        )
    else:
        width_map = pd.Series(np.full(len(edge_log), (min_w + max_w) / 2.0))

    edge_width_lookup = {}

    for (_, row), width in zip(edges.iterrows(), width_map.tolist()):
        edge_width_lookup[undirected_key(row["source"], row["target"])] = width

    red_edges = []
    gray_edges = []

    for u, v in graph.edges():
        if u in top3_nodes or v in top3_nodes:
            red_edges.append((u, v))
        else:
            gray_edges.append((u, v))

    gray_edge_widths = [
        edge_width_lookup[undirected_key(u, v)]
        for u, v in gray_edges
    ]

    red_edge_widths = [
        edge_width_lookup[undirected_key(u, v)]
        for u, v in red_edges
    ]

    fig, ax = plt.subplots(figsize=(16, 11.5))

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=gray_edges,
        width=gray_edge_widths,
        edge_color="lightgray",
        alpha=0.55,
        ax=ax,
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=red_edges,
        width=red_edge_widths,
        edge_color="#f94449",
        alpha=0.88,
        ax=ax,
    )

    node_collection = nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=[node_sizes.loc[node] for node in graph.nodes()],
        node_color=node_color_values,
        cmap=plt.cm.Reds,
        alpha=0.97,
        linewidths=1.8,
        edgecolors="black",
        ax=ax,
    )

    labels = {
        node: graph.nodes[node]["bank_name"]
        for node in graph.nodes()
    }

    nx.draw_networkx_labels(
        graph,
        label_pos,
        labels=labels,
        font_size=14,
        font_color="black",
        font_weight="bold",
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.82,
            pad=0.28,
        ),
        ax=ax,
    )

    for node in graph.nodes():
        x0, y0 = pos[node]
        x1, y1 = label_pos[node]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color="gray",
            linewidth=0.45,
            alpha=0.25,
        )

    cbar = plt.colorbar(node_collection, shrink=0.78, ax=ax)
    cbar.set_label("Systemic importance", fontsize=16)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        "Government Security Fire-Sale-Driven Indirect Contagion Network of Armenian Banks",
        fontsize=28,
        fontweight="bold",
        y=0.965,
    )

    fig.text(
        0.5,
        0.035,
        "Node size is proportional to total assets. Node color measures systemic importance in the state-bond overlap network. "
        "Edge width is proportional to log(1 + liquidity-weighted state-bond overlap), where overlap is computed from common government-security holdings. "
        "Red edges are all edges connected to the three most systemically important banks. "
        "Edge length is only a layout artifact and has no economic interpretation.",
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
    )

    ax.axis("off")
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.995])

    output_path = FIGURES_DIR / "indirect_contagion_network_armenia.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(output_path)
    print(TABLES_DIR / "indirect_contagion_overlap_matrix.csv")
    print(TABLES_DIR / "indirect_contagion_edge_table.csv")
    print(TABLES_DIR / "indirect_contagion_node_table.csv")


if __name__ == "__main__":
    main()