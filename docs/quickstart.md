---
title: "Quickstart"
description: "Lift a gpmap-v2 map into a graph, attach a fixation model, enumerate forward paths."
---

# Quickstart

Build a `GenotypePhenotypeGraph` from a `gpmap-v2` map, attach a fixation model to populate edge probabilities, and pull out the most likely paths from WT to a target.

## Build the graph

```python
from gpmap import GenotypePhenotypeMap
from gpgraph import GenotypePhenotypeGraph

gpm = GenotypePhenotypeMap(
    wildtype="AAA",
    genotypes=["AAA", "AAT", "ATA", "TAA", "ATT", "TAT", "TTA", "TTT"],
    phenotypes=[0.1, 0.2, 0.2, 0.6, 0.4, 0.6, 1.0, 1.1],
)

G = GenotypePhenotypeGraph.from_gpm(gpm)
G.number_of_nodes()  # 8
G.number_of_edges()  # 24 directed (12 undirected pairs)
```

`from_gpm` populates one node per genotype (carrying the columns from `gpm.data` as node attributes) and one directed edge per neighbor pair under Hamming distance with cutoff 1.

## Attach a fixation model

```python
G.add_model(column="phenotypes", model="sswm")

G.edges[(0, 1)]["prob"]  # SSWM fixation probability from AAA -> AAT
```

Available models out of the box: `"sswm"`, `"ratio"`, `"moran"`, `"mcclandish"`. The last two need `population_size=N` as a keyword argument:

```python
G.add_model(column="phenotypes", model="moran", population_size=100)
```

You can also pass any `f(fi, fj, **kwargs) -> float` (or vectorized `f(fi_array, fj_array, **kwargs) -> array`) callable as `model`.

## Enumerate forward paths

```python
from gpgraph.paths import forward_paths, forward_paths_prob

paths = forward_paths(G, source="AAA", target="TTT")
# [[0, 1, 4, 7], [0, 1, 5, 7], ...]   nested list of node indices

paths_prob = forward_paths_prob(G, source="AAA", target="TTT")
# {(0, 1, 4, 7): 0.0017, (0, 1, 5, 7): 0.0021, ...}
```

`forward_paths` returns all shortest paths from source to target. `forward_paths_prob` multiplies the `prob` edge attributes along each path. Endpoints can be node indices, genotype strings, or binary-encoded strings.

## Reduce paths to edge flux

```python
from gpgraph.paths import paths_prob_to_edges_flux

flux = paths_prob_to_edges_flux(paths_prob)
# {(0, 1): 0.0038, (1, 4): 0.0017, (1, 5): 0.0021, ...}
```

Edge flux is the sum of probabilities across paths that use a given edge. It is what you want to size edge widths by when visualizing.

## Plot the graph

```python
from gpgraph.pyplot import draw_gpgraph, draw_paths

fig, ax = draw_gpgraph(G)                      # default Hamming-weight vertical layout
ax = draw_paths(G, source="AAA", target="TTT") # flux overlay
```

Both functions require `gpgraph-v2[plot]` (matplotlib). See [Plotting](guides/plotting.md) for the full options.

## Codon neighbors

```python
G = GenotypePhenotypeGraph.from_gpm(gpm, neighbor_function="codon", cutoff=1)
```

The codon kernel uses an amino-acid minimum-base-pair distance table to decide whether two codons are one substitution apart at the codon level. See [Neighbors](concepts/neighbors.md) for when to use it.

## Where next

- [Graph model](concepts/graph-model.md) for the conceptual model and the relationship to `gpmap-v2`.
- [Neighbors](concepts/neighbors.md) for the dispatch policy and the codon kernel.
- [Fixation models](concepts/fixation-models.md) for the math behind sswm, ratio, moran, and mcclandish.
- [Forward paths](guides/forward-paths.md) for path enumeration and flux helpers.
- [Plotting](guides/plotting.md) for `gpgraph.pyplot`.
