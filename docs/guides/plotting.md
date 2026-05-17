---
title: "Plotting"
description: "Draw a GenotypePhenotypeGraph with the Hamming-layer layout, overlay forward-path flux, and customize colors and sizes."
---

# Plotting

`gpgraph.pyplot` is an optional matplotlib-backed subpackage for drawing genotype-phenotype graphs. Install via:

=== "uv"

    ```bash
    uv add "gpgraph-v2[plot]"
    ```

=== "pip"

    ```bash
    pip install "gpgraph-v2[plot]"
    ```

Importing `gpgraph.pyplot` without matplotlib installed raises a clear `ImportError`.

## Draw the graph

```python
from gpgraph.pyplot import draw_gpgraph

fig, ax = draw_gpgraph(G)
```

By default:

- Layout is the Hamming-weight vertical layout (`gpgraph.layout.flattened` with `vertical=True`): row index is the Hamming distance to WT, nodes within a row are spaced evenly.
- Node color is the `phenotypes` node attribute, mapped via the `"plasma"` colormap.
- Edges are drawn black, uniform width, no arrowheads.

Customize:

```python
draw_gpgraph(
    G,
    node_size=500,
    node_shape="s",
    edge_color="#888",
    edge_alpha=0.6,
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
)
```

| Group | Parameters |
|---|---|
| Node | `node_list`, `node_size`, `node_color`, `node_shape`, `node_alpha`, `node_linewidths`, `node_edgecolors` |
| Edge | `edge_list`, `edge_widths`, `edge_scalar`, `edge_color`, `edge_style`, `edge_alpha`, `edge_arrows`, `edge_arrowstyle`, `edge_arrowsize` |
| Colormap | `cmap`, `vmin`, `vmax` |

## Overlay path flux

If you pass `paths=` (a mapping from path tuples to probabilities, as returned by `forward_paths_prob`), `draw_gpgraph` draws only those edges and scales widths by the per-edge summed flux:

```python
from gpgraph.paths import forward_paths_prob
from gpgraph.pyplot import draw_gpgraph

paths_prob = forward_paths_prob(G, source="AAA", target="TTT")

fig, ax = draw_gpgraph(G, paths=paths_prob, edge_scalar=50)
```

`edge_scalar` is a multiplier so you can dial widths up or down independent of the underlying probabilities.

## `draw_paths`

The more direct flux overlay is `draw_paths`, which computes the flux for you:

```python
from gpgraph.pyplot import draw_paths

ax = draw_paths(G, source="AAA", target="TTT")
```

Or supply pre-computed `paths`:

```python
ax = draw_paths(G, paths=paths_prob, edge_scalar=50, colorbar=True)
```

Key parameters:

| Parameter | Default | Notes |
|---|---|---|
| `source`, `target` | first and last genotype | Either node id, genotype string, or binary string |
| `paths` | computed if not given | A `dict[tuple, float]` from `forward_paths_prob` |
| `edge_scalar` | `1.0` | Multiplier on edge widths |
| `cmap` | `"plasma"` | Colormap for node phenotypes |
| `colorbar` | `False` | If `True`, add a phenotype colorbar |

## Layouts

The default layout function is `gpgraph.layout.flattened`, which lays nodes in Hamming-weight rows:

```python
from gpgraph.layout import flattened

pos = flattened(G, vertical=True)  # dict[node_index, (x, y)]
fig, ax = draw_gpgraph(G, pos=pos)
```

Set `vertical=False` for left-to-right Hamming layers, `scale=N` to widen rows.

Or use any NetworkX layout:

```python
import networkx as nx

pos = nx.spring_layout(G, seed=0)
fig, ax = draw_gpgraph(G, pos=pos)
```

## Low-level primitives

For full control, drop down to `draw_nodes` and `draw_edges`:

```python
from gpgraph.pyplot import draw_nodes, draw_edges, construct_ax, despine_ax
from gpgraph.layout import flattened

fig, ax = construct_ax()
despine_ax(ax)
pos = flattened(G, vertical=True)

draw_edges(G, pos, ax=ax, edge_color="lightgray", width=0.5)
draw_nodes(G, pos, ax=ax, node_color=[G.nodes[n]["phenotypes"] for n in G.nodes], cmap="plasma")
```

Both `draw_nodes` and `draw_edges` are thin wrappers over `nx.draw_networkx_nodes` and `nx.draw_networkx_edges`, with the kwargs forwarded directly. See the [pyplot reference](../reference/pyplot.md) for the full signatures.

## Saving figures

`fig` is a plain matplotlib `Figure`, so any save method works:

```python
fig.savefig("graph.png", dpi=200, bbox_inches="tight")
fig.savefig("graph.svg")
```
