---
title: "Neighbors reference"
description: "Reference for get_neighbors, hamming, codon_neighbors, and edges_array_to_tuples."
---

# `gpgraph.neighbors`

Neighbor detection with automatic dispatch between Rust kernels. See [Concepts: Neighbors](../concepts/neighbors.md) for the dispatch policy at a glance.

## `get_neighbors`

```python
def get_neighbors(
    genotypes: Sequence[str],
    neighbor_function: str | Callable[..., bool] = "hamming",
    cutoff: int = 1,
    binary_packed: NDArray[np.uint8] | None = None,
) -> NDArray[np.int64]
```

Return the directed edge list of genotype neighbors.

### Parameters

`genotypes` (`Sequence[str]`, required)
:   Iterable of genotype strings, all the same length.

`neighbor_function` (`str | Callable`, default `"hamming"`)
:   `"hamming"`, `"codon"`, or a `f(g1, g2, cutoff=...) -> bool` callable.

`cutoff` (`int`, default `1`)
:   Maximum distance between two genotypes to be considered neighbors. Must be `>= 0`. `cutoff == 0` yields no edges.

`binary_packed` (`NDArray[np.uint8] | None`, default `None`)
:   Optional uint8 `(N, L)` matrix from `gpm.binary_packed`. When supplied and the alphabet is biallelic (max value 0 or 1), the bit-flip fast path is used for cutoffs 1 and 2.

### Returns

`np.ndarray[int64]` of shape `(E, 2)`, sorted lexicographically. Each undirected pair appears twice: `(i, j)` and `(j, i)`.

### Raises

`NeighborError`
:   If `cutoff < 0`, or `neighbor_function` is not a known string and not callable.

## `hamming`

```python
def hamming(g1: str, g2: str, cutoff: int = 1) -> bool
```

Scalar Hamming-distance neighbor test. Kept for API continuity with v1 and for use as a user-supplied `neighbor_function`. The array path goes through Rust.

## `codon_neighbors`

```python
def codon_neighbors(g1: str, g2: str, cutoff: int = 1) -> bool
```

Scalar codon-distance neighbor test. Uses the same 256x256 amino-acid minimum-base-pair distance table as the Rust kernel; calls into Rust with a two-genotype list.

## `edges_array_to_tuples`

```python
def edges_array_to_tuples(edges: NDArray[np.int64]) -> list[tuple[int, int]]
```

Convenience: convert the `(E, 2)` int64 edge array into a list of `(src, dst)` tuples. Used internally by `GenotypePhenotypeGraph.from_gpm`; exposed for downstream callers that want to feed edges into a raw `nx.add_edges_from`.

## Dispatch table

```
+------------------------------------------------+--------------------------------+
| Problem shape                                  | Kernel                         |
+------------------------------------------------+--------------------------------+
| user-supplied callable                         | pure Python O(N^2) fallback    |
| hamming, biallelic packed, cutoff <= 2         | Rust bit-flip (O(N*C(L,c)))    |
| hamming, biallelic packed, larger cutoff       | Rust rayon-parallel packed     |
| hamming, non-binary alphabet                   | Rust rayon-parallel string     |
| codon                                          | Rust rayon-parallel codon      |
+------------------------------------------------+--------------------------------+
```
