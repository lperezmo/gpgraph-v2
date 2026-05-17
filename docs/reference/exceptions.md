---
title: "Exceptions"
description: "GpgraphError and NeighborError."
---

# `gpgraph.exceptions`

A two-class hierarchy.

## Hierarchy

```
Exception
└── GpgraphError
    └── NeighborError
```

## `GpgraphError`

```python
class GpgraphError(Exception):
    """Base class for all gpgraph-specific errors."""
```

Raised by:

- `GenotypePhenotypeGraph.gpm` when accessed on a graph that has no map attached.
- `GenotypePhenotypeGraph.add_model` for unknown model strings, unknown fitness columns.
- `GenotypePhenotypeGraph.model` when called before `add_model`.
- `from_gpm` when the input is not a `GenotypePhenotypeMap`.

## `NeighborError`

```python
class NeighborError(GpgraphError):
    """Raised when neighbor detection cannot proceed."""
```

Raised by `gpgraph.neighbors.get_neighbors` for invalid cutoffs (negative, non-integer) or unknown `neighbor_function` arguments.

## Catching pattern

```python
from gpgraph.exceptions import GpgraphError, NeighborError

try:
    G = GenotypePhenotypeGraph.from_gpm(gpm, neighbor_function="unknown")
except NeighborError as e:
    log.error("Neighbor kernel unavailable: %s", e)
except GpgraphError as e:
    log.error("Graph construction failed: %s", e)
```

Library code that wants to swallow only gpgraph failures should catch `GpgraphError` (which covers `NeighborError` by subclass) and re-raise anything else.
