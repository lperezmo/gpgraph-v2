"""Matplotlib-backed drawing for GenotypePhenotypeGraph.

This subpackage is optional. Install via ``pip install gpgraph-v2[plot]``
or ``uv add --extra plot gpgraph-v2``. Importing :mod:`gpgraph.pyplot`
without matplotlib available raises a clear ImportError.
"""

from __future__ import annotations

try:
    import matplotlib  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("gpgraph.pyplot requires matplotlib; install gpgraph-v2[plot]") from exc

from gpgraph.pyplot.draw import draw_gpgraph
from gpgraph.pyplot.paths import draw_paths
from gpgraph.pyplot.primitives import draw_edges, draw_nodes
from gpgraph.pyplot.utils import (
    bins,
    construct_ax,
    contrast_ink,
    despine_ax,
    resolve_node_fills,
    truncate_colormap,
)

__all__ = [
    "bins",
    "construct_ax",
    "contrast_ink",
    "despine_ax",
    "draw_edges",
    "draw_gpgraph",
    "draw_nodes",
    "draw_paths",
    "resolve_node_fills",
    "truncate_colormap",
]
