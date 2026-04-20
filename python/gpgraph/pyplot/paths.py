"""Path-flux visualization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from gpgraph.layout import flattened
from gpgraph.paths import forward_paths_prob, paths_prob_to_edges_flux
from gpgraph.pyplot.primitives import draw_edges, draw_nodes
from gpgraph.pyplot.utils import despine_ax

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from gpgraph.base import GenotypePhenotypeGraph


def draw_paths(
    G: GenotypePhenotypeGraph,
    paths: dict[tuple[int, ...], float] | None = None,
    pos: dict[int, tuple[float, float]] | None = None,
    source: int | str | None = None,
    target: int | str | None = None,
    edge_scalar: float = 1.0,
    edge_color: Any = "black",
    edge_alpha: float = 1.0,
    width: float = 1.0,
    node_list: list[int] | None = None,
    node_size: Any = 300,
    node_shape: str = "o",
    cmap: str = "plasma",
    ax: Axes | None = None,
    colorbar: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Axes:
    """Visualize forward-path flux between two genotypes."""
    if ax is None:
        _, ax = plt.subplots()
        despine_ax(ax)

    if pos is None:
        pos = flattened(G, vertical=True)

    if paths is None:
        if source is None:
            source = (
                int(G.gpm.genotypes[0])
                if isinstance(G.gpm.genotypes[0], int)
                else G.gpm.genotypes[0]
            )
        if target is None:
            target = G.gpm.genotypes[-1]
        paths = forward_paths_prob(G, source=source, target=target)

    flux = paths_prob_to_edges_flux(paths)
    edge_list = list(flux.keys())
    edge_widths = np.asarray(list(flux.values()), dtype=np.float64)

    if node_list is None:
        node_list = list(G.nodes())

    draw_edges(
        G,
        pos,
        ax=ax,
        edgelist=edge_list,
        width=edge_scalar * edge_widths,
        edge_color=edge_color,
        alpha=edge_alpha,
    )
    draw_nodes(
        G,
        pos,
        ax=ax,
        nodelist=node_list,
        node_size=node_size,
        node_color=[G.nodes[n].get("phenotypes", 0.0) for n in node_list],
        node_shape=node_shape,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if colorbar:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        cm.set_array([])
        ax.figure.colorbar(cm, ax=ax)

    # Silence "width" unused-param ruff complaint; keep it in the signature
    # for callers that pass it positionally.
    _ = width
    return ax
