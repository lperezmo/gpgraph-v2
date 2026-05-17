---
title: "GenotypePhenotypeGraph"
description: "Reference for the core GenotypePhenotypeGraph class: from_gpm, add_model, model, and the file-based constructors."
---

# `gpgraph.base`

The core graph class, plus the I/O class-method constructors.

## `GenotypePhenotypeGraph`

```python
class GenotypePhenotypeGraph(nx.DiGraph):
    def __init__(self) -> None: ...
```

A NetworkX directed graph wrapping a gpmap-v2 `GenotypePhenotypeMap`. Construction is only supported through `from_gpm`; the v1 `add_gpm` post-construction injection is removed.

### `from_gpm`

```python
@classmethod
def from_gpm(
    cls,
    gpm: GenotypePhenotypeMap,
    neighbor_function: str | Callable[..., bool] = "hamming",
    cutoff: int = 1,
) -> GenotypePhenotypeGraph
```

Build a graph from a gpmap-v2 `GenotypePhenotypeMap`.

#### Parameters

`gpm` (`GenotypePhenotypeMap`, required)
:   A gpmap-v2 instance. Will be stashed as `G.gpm`.

`neighbor_function` (`str | Callable`, default `"hamming"`)
:   `"hamming"`, `"codon"`, or a `f(g1, g2, cutoff=...) -> bool` callable. Determines how neighbors are detected.

`cutoff` (`int`, default `1`)
:   Maximum neighbor distance. Must be `>= 0`. `cutoff == 0` yields no edges.

#### Returns

A `GenotypePhenotypeGraph` with:

- One node per row in `gpm.data`, carrying all of its columns as node attributes.
- One directed edge per neighbor pair, each undirected pair appearing as both `(i, j)` and `(j, i)`.

### `add_model`

```python
def add_model(
    self,
    column: str | None = None,
    model: str | Callable[..., Any] | None = None,
    **model_params: Any,
) -> None
```

Populate the edge attribute `prob` with a fixation model's output.

#### Parameters

`column` (`str | None`, default `None`)
:   Column in `gpm.data` to use as the node fitness. If `None`, every node is given fitness 1.0.

`model` (`str | Callable | None`, default `None`)
:   One of `"sswm"`, `"ratio"`, `"moran"`, `"mcclandish"`, or a callable `f(fi, fj, **kwargs)`. `None` assigns 1.0 to every edge.

`**model_params`
:   Forwarded to the model on every call. The common one is `population_size` for `moran` / `mcclandish`.

#### Notes

The vectorized fast path uses the model's array entry point. If the supplied callable raises `TypeError` on arrays, `add_model` falls back to a Python `for` loop.

### `model`

```python
def model(self, v1: float, v2: float) -> float
```

Evaluate the last-added fixation model at a single `(v1, v2)` pair. Raises `GpgraphError` if `add_model` has not been called yet.

### `gpm` (property)

```python
@property
def gpm(self) -> GenotypePhenotypeMap
```

Returns the attached map. Raises `GpgraphError` if the graph was constructed without one.

### `read_json` and `read_csv`

```python
@classmethod
def read_json(cls, fname: str) -> GenotypePhenotypeGraph

@classmethod
def read_csv(cls, fname: str) -> GenotypePhenotypeGraph
```

Load a gpmap-v2 JSON or CSV save and lift it into a graph with default neighbor settings (`hamming`, cutoff 1). For graph-level serialization (with `prob` attributes), use NetworkX's own writers.

### Inherited NetworkX surface

Because `GenotypePhenotypeGraph` subclasses `nx.DiGraph`, the full NetworkX API is available: `G.nodes`, `G.edges`, `G.add_node`, `G.degree`, `nx.has_path(G, ...)`, etc. The class adds gpmap-specific construction and the `prob` edge attribute convention; it does not hide or rename any of the parent API.
