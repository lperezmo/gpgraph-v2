"""Main draw_gpgraph entry point (merged from v1's draw.py + network.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from gpgraph.layout import flattened
from gpgraph.paths import paths_prob_to_edges_flux
from gpgraph.pyplot.primitives import draw_edges, draw_nodes
from gpgraph.pyplot.utils import construct_ax, contrast_ink, resolve_node_fills

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gpgraph.base import GenotypePhenotypeGraph


def draw_gpgraph(
    G: GenotypePhenotypeGraph,
    pos: dict[int, tuple[float, float]] | None = None,
    ax: Axes | None = None,
    paths: dict[tuple[int, ...], float] | None = None,
    edge_list: list[tuple[int, int]] | None = None,
    edge_widths: float | np.ndarray = 1.0,
    edge_scalar: float = 1.0,
    edge_color: Any = "black",
    edge_style: str = "solid",
    edge_alpha: float = 1.0,
    edge_arrows: bool = False,
    edge_arrowstyle: str = "-|>",
    edge_arrowsize: int = 10,
    node_list: list[int] | None = None,
    node_size: Any = 300,
    node_color: Any = None,
    node_shape: str = "o",
    node_alpha: float = 1.0,
    node_linewidths: float = 0.0,
    node_edgecolors: Any = "black",
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    with_labels: bool = False,
    labels: dict[int, str] | None = None,
    label_font_size: float = 8.0,
    label_ink: Any = "auto",
) -> tuple[Figure, Axes]:
    """Draw a :class:`GenotypePhenotypeGraph` on a matplotlib axes.

    Node color defaults to the ``phenotypes`` node attribute (from gpmap-v2).
    Edge widths default to a constant; if ``paths`` is passed (a mapping of
    shortest-path tuples to path probabilities), edge widths are set from the
    per-edge summed flux so the visualization highlights likely trajectories.

    Set ``with_labels=True`` to label each node (defaulting to its ``genotypes``
    attribute). Labels use ``label_ink``: the default ``"auto"`` picks a dark or
    light color per node from that node's own fill luminance via
    :func:`~gpgraph.pyplot.utils.contrast_ink`, so text stays readable on both
    the bright and dark ends of the colormap and in light or dark display
    themes, with no manual color tuning. Pass any matplotlib color to
    ``label_ink`` to override. ``node_edgecolors="auto"`` applies the same
    per-node contrast logic to node outlines.
    """
    if ax is None:
        fig, ax = construct_ax()
    else:
        figure = ax.get_figure()
        # matplotlib reports this as Figure | SubFigure | None; narrow to Figure.
        from matplotlib.figure import Figure as _Fig

        assert isinstance(figure, _Fig)
        fig = figure
    assert ax is not None

    if pos is None:
        pos = flattened(G, vertical=True)

    if paths is not None:
        flux = paths_prob_to_edges_flux(paths)
        edge_list = list(flux.keys())
        edge_widths = np.asarray(list(flux.values()), dtype=np.float64)

    if edge_list is None:
        edge_list = list(G.edges())

    width_arg: Any
    if isinstance(edge_widths, np.ndarray):
        width_arg = edge_scalar * edge_widths
    else:
        width_arg = edge_scalar * float(edge_widths)

    edge_kwargs: dict[str, Any] = dict(
        edgelist=edge_list,
        width=width_arg,
        edge_color=edge_color,
        style=edge_style,
        alpha=edge_alpha,
        arrows=edge_arrows,
    )
    # NetworkX warns if arrowstyle/arrowsize are supplied alongside the default
    # LineCollection backend; only pass them when arrows are requested.
    if edge_arrows:
        edge_kwargs["arrowstyle"] = edge_arrowstyle
        edge_kwargs["arrowsize"] = edge_arrowsize
    draw_edges(G, pos, ax=ax, **edge_kwargs)

    if node_list is None:
        node_list = list(G.nodes())

    if node_color is None:
        node_color = [G.nodes[n].get("phenotypes", 0.0) for n in node_list]

    # Resolve per-node fills only when something needs to contrast against them
    # (auto outlines or auto-inked labels), so the common path stays cheap.
    need_fills = with_labels or node_edgecolors == "auto" or label_ink == "auto"
    fills = (
        resolve_node_fills(node_color, len(node_list), cmap=cmap, vmin=vmin, vmax=vmax)
        if need_fills
        else None
    )

    edgecolors_arg: Any = node_edgecolors
    if node_edgecolors == "auto" and fills is not None:
        edgecolors_arg = [contrast_ink(c) for c in fills]

    draw_nodes(
        G,
        pos,
        ax=ax,
        nodelist=node_list,
        node_size=node_size,
        node_color=node_color,
        node_shape=node_shape,
        alpha=node_alpha,
        linewidths=node_linewidths,
        edgecolors=edgecolors_arg,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if with_labels:
        if labels is None:
            labels = {n: str(G.nodes[n].get("genotypes", n)) for n in node_list}
        for i, n in enumerate(node_list):
            if n not in labels:
                continue
            x, y = pos[n]
            ink = (
                contrast_ink(fills[i]) if (label_ink == "auto" and fills is not None) else label_ink
            )
            ax.text(
                x,
                y,
                labels[n],
                ha="center",
                va="center",
                fontsize=label_font_size,
                color=ink,
                zorder=3,
            )

    return fig, ax
