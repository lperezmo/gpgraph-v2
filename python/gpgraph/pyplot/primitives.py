"""Thin typed wrappers around networkx drawing primitives.

The v1 ``_nx_wrapper`` introspection shim is gone (locked decision 2026-04-20);
NetworkX is version-pinned in ``pyproject.toml`` and CI catches breakage on bump.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from gpgraph.base import GenotypePhenotypeGraph


def draw_nodes(
    G: GenotypePhenotypeGraph,
    pos: dict[int, tuple[float, float]],
    ax: Axes,
    **kwargs: Any,
) -> Any:
    """Draw the graph nodes via :func:`networkx.draw_networkx_nodes`."""
    return nx.draw_networkx_nodes(G, pos=pos, ax=ax, **kwargs)


def draw_edges(
    G: GenotypePhenotypeGraph,
    pos: dict[int, tuple[float, float]],
    ax: Axes,
    **kwargs: Any,
) -> Any:
    """Draw the graph edges via :func:`networkx.draw_networkx_edges`."""
    return nx.draw_networkx_edges(G, pos=pos, ax=ax, **kwargs)


def draw_node_labels(
    G: GenotypePhenotypeGraph,
    pos: dict[int, tuple[float, float]],
    ax: Axes,
    **kwargs: Any,
) -> Any:
    return nx.draw_networkx_labels(G, pos=pos, ax=ax, **kwargs)


def draw_edge_labels(
    G: GenotypePhenotypeGraph,
    pos: dict[int, tuple[float, float]],
    ax: Axes,
    **kwargs: Any,
) -> Any:
    return nx.draw_networkx_edge_labels(G, pos=pos, ax=ax, **kwargs)
