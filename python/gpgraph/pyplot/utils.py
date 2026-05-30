"""Small matplotlib helpers used by the drawing code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def contrast_ink(
    color: Any,
    *,
    dark: str = "#10141a",
    light: str = "#f6f8fa",
    threshold: float = 0.6,
) -> str:
    """Pick a dark or light ink that stays legible on top of ``color``.

    Returns ``dark`` when ``color`` is light and ``light`` when it is dark,
    using perceived luminance (``0.299 R + 0.587 G + 0.114 B``). Use this for
    text or outlines drawn on a filled node so they contrast with the node's
    own fill instead of the figure background. Because the fill is the same
    whatever theme the page uses, the chosen ink is legible in both light and
    dark display modes without any manual override.

    The default ``threshold`` of ``0.6`` puts the crossover in the orange band
    of perceptual colormaps (magma, viridis, plasma), so only genuinely light
    fills (yellow/orange) get dark ink.
    """
    r, g, b = colors.to_rgb(color)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return dark if luminance > threshold else light


def resolve_node_fills(
    node_color: Any,
    n: int,
    *,
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[tuple[float, ...]]:
    """Resolve ``node_color`` to one RGBA fill per node.

    Mirrors how :func:`networkx.draw_networkx_nodes` renders ``node_color``:

    - a single color (e.g. ``"red"``) is broadcast to all ``n`` nodes;
    - a 1-D sequence of scalars is mapped through ``cmap`` after a
      ``Normalize(vmin, vmax)`` (defaulting to the data min/max), matching
      matplotlib's scalar-mapping path;
    - a sequence of colors (strings or RGB/RGBA tuples) is converted as-is.

    The returned fills are what :func:`contrast_ink` should be applied to when
    choosing per-node label or outline ink.
    """
    if isinstance(node_color, str):
        return [colors.to_rgba(node_color)] * n
    arr = np.asarray(node_color)
    if arr.ndim == 1 and arr.dtype.kind in "iuf":
        cmap_obj = plt.get_cmap(cmap)
        lo = float(np.nanmin(arr)) if vmin is None else float(vmin)
        hi = float(np.nanmax(arr)) if vmax is None else float(vmax)
        norm = colors.Normalize(lo, hi)
        return [tuple(cmap_obj(norm(float(v)))) for v in arr]
    return [colors.to_rgba(c) for c in node_color]


def bins(G: GenotypePhenotypeGraph) -> dict[int, list[int]]:
    """Deprecated alias for :func:`gpgraph.layout.bins`. Kept here for convenience."""
    from gpgraph.layout import bins as _bins

    return _bins(G)
