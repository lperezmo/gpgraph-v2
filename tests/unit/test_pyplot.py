"""Smoke tests for gpgraph.pyplot. Run with the Agg backend (headless)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
from gpgraph import GenotypePhenotypeGraph
from gpgraph.pyplot import (
    bins,
    construct_ax,
    despine_ax,
    draw_gpgraph,
    draw_paths,
    truncate_colormap,
)
from gpgraph.pyplot.primitives import draw_edge_labels, draw_edges, draw_node_labels, draw_nodes


def test_construct_ax_returns_pair() -> None:
    fig, ax = construct_ax(figsize=(4, 4))
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_construct_ax_rejects_bad_figsize() -> None:
    with pytest.raises(ValueError):
        construct_ax(figsize=(0, 1))


def test_despine_ax_noop_on_none() -> None:
    assert despine_ax(None) is None


def test_truncate_colormap_returns_cmap() -> None:
    cmap = truncate_colormap("plasma", 0.1, 0.9)
    assert cmap is not None


def test_bins_alias(tiny_graph: GenotypePhenotypeGraph) -> None:
    b = bins(tiny_graph)
    assert 0 in b and 3 in b


def test_draw_gpgraph_smoke(tiny_graph: GenotypePhenotypeGraph) -> None:
    fig, ax = draw_gpgraph(tiny_graph)
    assert fig is not None and ax is not None
    plt.close(fig)


def test_draw_gpgraph_with_paths(tiny_graph: GenotypePhenotypeGraph) -> None:
    from gpgraph.paths import forward_paths_prob

    tiny_graph.add_model(column="phenotypes", model="sswm")
    pp = forward_paths_prob(tiny_graph, 0, 7)
    fig, _ax = draw_gpgraph(tiny_graph, paths=pp)
    assert fig is not None
    plt.close(fig)


def test_draw_paths_smoke(tiny_graph: GenotypePhenotypeGraph) -> None:
    tiny_graph.add_model(column="phenotypes", model="sswm")
    ax = draw_paths(tiny_graph, source=0, target=7)
    assert ax is not None
    plt.close(ax.get_figure())


def test_primitives_round_trip(tiny_graph: GenotypePhenotypeGraph) -> None:
    from gpgraph.layout import flattened

    pos = flattened(tiny_graph, vertical=True)
    fig, ax = construct_ax()
    draw_edges(tiny_graph, pos, ax=ax, edgelist=list(tiny_graph.edges())[:5])
    draw_nodes(tiny_graph, pos, ax=ax, nodelist=list(tiny_graph.nodes()))
    draw_node_labels(tiny_graph, pos, ax=ax, labels={n: str(n) for n in tiny_graph.nodes()})
    draw_edge_labels(tiny_graph, pos, ax=ax, edge_labels={(0, 1): "hop"})
    plt.close(fig)
