"""Small matplotlib helpers used by the drawing code."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from gpgraph.base import GenotypePhenotypeGraph


def despine_ax(ax: Axes | None) -> Axes | None:
    """Remove all spines and ticks from a matplotlib axis (in place)."""
    if ax is None:
        return None
    for spine in ("right", "left", "top", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def construct_ax(
    figsize: tuple[float, float] = (10, 10), despine: bool = True
) -> tuple[Figure, Axes]:
    """Create a fresh ``(fig, ax)`` pair; optionally strip spines and ticks."""
    if len(figsize) != 2 or any(v <= 0 for v in figsize):
        raise ValueError(f"figsize must be a positive 2-tuple, got {figsize}")
    fig, ax = plt.subplots(figsize=figsize)
    if despine:
        despine_ax(ax)
    return fig, ax


def truncate_colormap(
    cmap: str | Colormap,
    minval: float = 0.0,
    maxval: float = 1.0,
    n: int = 100,
) -> Colormap:
    """Return a copy of ``cmap`` restricted to ``[minval, maxval]``."""
    base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return colors.LinearSegmentedColormap.from_list(
        f"trunc({base.name},{minval:.2f},{maxval:.2f})",
        base(np.linspace(minval, maxval, n)),
    )


def bins(G: GenotypePhenotypeGraph) -> dict[int, list[int]]:
    """Deprecated alias for :func:`gpgraph.layout.bins`. Kept here for convenience."""
    from gpgraph.layout import bins as _bins

    return _bins(G)
