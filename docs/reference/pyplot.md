---
title: "pyplot reference"
description: "draw_gpgraph, draw_paths, draw_nodes, draw_edges, and the layout helpers."
---

# `gpgraph.pyplot`

Optional matplotlib-backed drawing. Requires `gpgraph-v2[plot]`. See [Plotting](../guides/plotting.md) for a guide-style walkthrough.

```python
from gpgraph.pyplot import draw_gpgraph, draw_paths, draw_nodes, draw_edges
from gpgraph.pyplot import bins, construct_ax, despine_ax, truncate_colormap
from gpgraph.pyplot import contrast_ink, resolve_node_fills
```

## `draw_gpgraph`

```python
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
) -> tuple[Figure, Axes]
```

Draw the graph on a matplotlib axes.

### Key behaviors

- `pos` defaults to the Hamming-weight vertical layout (`gpgraph.layout.flattened` with `vertical=True`).
- `node_color` defaults to the `phenotypes` node attribute.
- When `paths` is provided, only the edges that appear in paths are drawn, with widths scaled by the per-edge summed flux.
- `with_labels=True` labels each node (default text = the `genotypes` attribute). With `label_ink="auto"` (the default) each label's color is chosen from its node's fill luminance via `contrast_ink`, so it is legible across the whole colormap and in both light and dark themes. `node_edgecolors="auto"` applies the same per-node contrast to outlines. Pass `labels`, `label_font_size`, or a fixed `label_ink` to override.

Returns `(fig, ax)`.

## `contrast_ink`

```python
def contrast_ink(
    color: Any,
    *,
    dark: str = "#10141a",
    light: str = "#f6f8fa",
    threshold: float = 0.6,
) -> str
```

Return `dark` when `color` is light and `light` when it is dark, using perceived luminance (`0.299 R + 0.587 G + 0.114 B`). Intended for text or outlines drawn on a filled node so they contrast with the node's own fill rather than the figure background, keeping them legible in both light and dark display modes. The default `threshold` of `0.6` puts the crossover in the orange band of perceptual colormaps (magma, viridis, plasma).

## `resolve_node_fills`

```python
def resolve_node_fills(
    node_color: Any,
    n: int,
    *,
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[tuple[float, ...]]
```

Resolve `node_color` to one RGBA fill per node, mirroring how `networkx.draw_networkx_nodes` renders it: a single color is broadcast, a 1-D scalar sequence is mapped through `cmap` after `Normalize(vmin, vmax)`, and a sequence of colors is converted as-is. The returned fills are what `contrast_ink` should be applied to when picking per-node ink.

## `draw_paths`

```python
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
) -> Axes
```

Visualize forward-path flux between two genotypes. If `paths` is `None`, computes it via `forward_paths_prob(G, source=source or first_genotype, target=target or last_genotype)`.

Returns the `Axes`.

## `draw_nodes` and `draw_edges`

Thin wrappers over `nx.draw_networkx_nodes` and `nx.draw_networkx_edges` with the same keyword arguments. Use these when you want full control:

```python
from gpgraph.pyplot import draw_nodes, draw_edges, construct_ax
from gpgraph.layout import flattened

fig, ax = construct_ax()
pos = flattened(G, vertical=True)
draw_edges(G, pos, ax=ax, edge_color="lightgray", width=0.5)
draw_nodes(G, pos, ax=ax, node_size=200, cmap="viridis")
```

## Layout helpers

### `flattened`

```python
from gpgraph.layout import flattened

pos = flattened(
    G,
    node_list=None,
    scale=1.0,
    vertical=False,
)
```

Hamming-weight rows. Each node's row index is its `n_mutations` (Hamming distance to WT). Within a row, nodes are spaced evenly and centered on the origin. With `vertical=False`, rows grow left-to-right; with `vertical=True`, top-to-bottom.

### `bins`

```python
from gpgraph.layout import bins

bins_dict = bins(G)   # dict[hamming_weight, list[node_index]]
```

Group node indices by Hamming weight. Useful for color-coding nodes per layer.

## Utility helpers

```python
from gpgraph.pyplot import construct_ax, despine_ax, truncate_colormap
```

| Helper | Purpose |
|---|---|
| `construct_ax()` | Open a new `(fig, ax)` pair sized for genotype graphs |
| `despine_ax(ax)` | Strip the four spines and ticks from a matplotlib axes |
| `truncate_colormap(cmap, minval, maxval)` | Return a truncated colormap, e.g. cut off the low-saturation tail |
