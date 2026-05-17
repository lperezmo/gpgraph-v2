---
title: "Paths reference"
description: "Reference for forward_paths, forward_paths_prob, paths_to_edges, paths_prob_to_edges_flux, and edges_flux_to_node_flux."
---

# `gpgraph.paths`

Forward-path enumeration and flux helpers. See [Forward paths and flux](../guides/forward-paths.md) for a guide-style walkthrough.

## `forward_paths`

```python
def forward_paths(
    G: GenotypePhenotypeGraph,
    source: int | str,
    target: int | str,
    max_paths: int | None = None,
) -> list[list[int]]
```

Return all shortest forward paths from `source` to `target` as a list of node-index lists.

### Parameters

`G` (`GenotypePhenotypeGraph`, required)
:   The graph.

`source`, `target` (`int | str`, required)
:   Endpoints. Each may be a node index, a genotype string, or a binary string (matching one of `gpm.binary[i]`).

`max_paths` (`int | None`, default `None`)
:   Optional cap on the number of returned paths. The number of shortest paths in an orthotope grows combinatorially, so this is a safety rail.

### Raises

`ValueError`
:   If an endpoint cannot be resolved to a node index.

## `forward_paths_prob`

```python
def forward_paths_prob(
    G: GenotypePhenotypeGraph,
    source: int | str,
    target: int | str,
    max_paths: int | None = None,
) -> dict[tuple[int, ...], float]
```

Return a `dict` mapping each shortest path (as a node-index tuple) to the product of edge `prob` attributes along it.

Calling `forward_paths_prob` before `G.add_model(column=..., model=...)` raises `ValueError("edge ... has no 'prob' attribute; call add_model first")`.

## `paths_to_edges`

```python
def paths_to_edges(
    paths: list[list[int]] | list[tuple[int, ...]],
    repeat: bool = False,
) -> list[tuple[int, int]]
```

Flatten paths to edges. With `repeat=False` (default), duplicate edges are dropped. With `repeat=True`, each edge is listed once per path that uses it.

## `paths_to_edges_count`

```python
def paths_to_edges_count(paths: list[list[int]]) -> Counter[tuple[int, int]]
```

Returns a `collections.Counter` keyed by edge, counting how many paths traverse each edge.

## `paths_prob_to_edges_flux`

```python
def paths_prob_to_edges_flux(
    paths_prob: dict[tuple[int, ...], float],
) -> dict[tuple[int, int], float]
```

Sum path probabilities onto their constituent edges. Returns a `dict` keyed by `(src, dst)` with the per-edge flux as the value. This is what `gpgraph.pyplot.draw_paths` uses to size edge widths.

## `edges_flux_to_node_flux`

```python
def edges_flux_to_node_flux(G: GenotypePhenotypeGraph) -> dict[int, float]
```

Sum the `capacity` edge attribute (note: capacity, not prob) of incoming edges for each node. Returns `dict[node_index, total_in_flux]`.

Use this when you have a precomputed edge capacity field (e.g., from an external flow analysis) and want per-node ingress.
