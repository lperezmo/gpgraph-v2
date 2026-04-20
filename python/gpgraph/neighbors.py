"""Neighbor detection for genotype-phenotype graphs.

Dispatches between several implementations based on the problem shape:

1. user-supplied callable                 -> pure-Python O(N^2) fallback
2. biallelic packed + cutoff in {1, 2}    -> Rust bit-flip (O(N * C(L, k)))
3. biallelic packed + larger cutoff       -> Rust rayon-parallel pairwise
4. arbitrary alphabet + hamming           -> Rust string pairwise
5. codon neighbors                        -> Rust codon pairwise

The Rust entry points are in :mod:`gpgraph._rust`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from gpgraph import _rust
from gpgraph.exceptions import NeighborError


def hamming(g1: str, g2: str, cutoff: int = 1) -> bool:
    """Scalar Hamming-distance neighbor test.

    Kept for API continuity with v1 (and for use as a user-supplied
    ``neighbor_function`` in :meth:`GenotypePhenotypeGraph.add_gpm`; the
    array path goes through Rust).
    """
    if len(g1) != len(g2):
        raise ValueError("g1 and g2 must have the same length")
    return sum(a != b for a, b in zip(g1, g2, strict=True)) <= cutoff


def codon_neighbors(g1: str, g2: str, cutoff: int = 1) -> bool:
    """Scalar codon-distance neighbor test.

    Uses the same 256 x 256 amino-acid min-bp distance table that the Rust
    kernel uses, fetched via a single-pair call.
    """
    if len(g1) != len(g2):
        raise ValueError("g1 and g2 must have the same length")
    # One-shot: ask the Rust kernel for neighbors of the two-genotype list.
    arr = _rust.codon_neighbors([g1, g2], cutoff)
    return bool(arr.size)


def _edges_from_user_callable(
    genotypes: Sequence[str],
    func: Callable[..., bool],
    cutoff: int,
) -> NDArray[np.int64]:
    n = len(genotypes)
    out: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if func(genotypes[i], genotypes[j], cutoff=cutoff):
                out.append((i, j))
                out.append((j, i))
    out.sort()
    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(out, dtype=np.int64)


def get_neighbors(
    genotypes: Sequence[str],
    neighbor_function: str | Callable[..., bool] = "hamming",
    cutoff: int = 1,
    binary_packed: NDArray[np.uint8] | None = None,
) -> NDArray[np.int64]:
    """Return the directed edge list of genotype neighbors.

    Parameters
    ----------
    genotypes:
        Iterable of genotype strings (all same length).
    neighbor_function:
        Either ``"hamming"``, ``"codon"``, or a user-supplied callable
        ``f(g1, g2, cutoff=...) -> bool``.
    cutoff:
        Maximum distance at which two genotypes are considered neighbors.
        Must be a non-negative integer. ``cutoff == 0`` yields no edges.
    binary_packed:
        Optional uint8 ``(N, L)`` matrix from ``gpm.binary_packed``. When
        supplied and the alphabet is biallelic, the Rust kernel dispatches
        to the packed hamming path (faster than comparing strings byte-by-byte).

    Returns
    -------
    np.ndarray[int64] shape ``(E, 2)``
        Each undirected pair ``(i, j)`` appears twice: ``(i, j)`` and ``(j, i)``.
        Rows are sorted lexicographically so output is deterministic.
    """
    try:
        cutoff = int(cutoff)
    except (TypeError, ValueError) as exc:
        raise NeighborError(f"cutoff must be a non-negative integer, got {cutoff!r}") from exc
    if cutoff < 0:
        raise NeighborError(f"cutoff must be >= 0, got {cutoff}")
    if cutoff == 0:
        return np.empty((0, 2), dtype=np.int64)

    genotypes = list(genotypes)
    if not genotypes:
        return np.empty((0, 2), dtype=np.int64)

    # User-supplied callable: pure-Python slow path.
    if callable(neighbor_function):
        return _edges_from_user_callable(genotypes, neighbor_function, cutoff)

    if neighbor_function == "hamming":
        if binary_packed is not None:
            bp = np.ascontiguousarray(binary_packed, dtype=np.uint8)
            # Biallelic bit-flip fast path for cutoff <= 2.
            if cutoff <= 2 and np.max(bp) <= 1:
                return _rust.hamming_bitflip_neighbors(bp, cutoff)
            return _rust.hamming_neighbors_packed(bp, cutoff)
        return _rust.hamming_neighbors_strings(genotypes, cutoff)

    if neighbor_function == "codon":
        return _rust.codon_neighbors(genotypes, cutoff)

    raise NeighborError(
        f"neighbor_function must be 'hamming', 'codon', or a callable; got {neighbor_function!r}"
    )


def edges_array_to_tuples(edges: NDArray[np.int64]) -> list[tuple[int, int]]:
    """Convenience: convert the ``(E, 2)`` int64 edge array into a list of tuples."""
    return [(int(a), int(b)) for a, b in edges]


__all__ = [
    "codon_neighbors",
    "edges_array_to_tuples",
    "get_neighbors",
    "hamming",
]
