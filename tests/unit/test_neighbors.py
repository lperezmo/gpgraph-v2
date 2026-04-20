"""Tests for gpgraph.neighbors."""

from __future__ import annotations

import numpy as np
import pytest
from gpgraph import _rust
from gpgraph.exceptions import NeighborError
from gpgraph.neighbors import codon_neighbors, get_neighbors, hamming


def _edges_set(arr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in arr}


def test_hamming_scalar_basic() -> None:
    assert hamming("A", "A", 0)
    assert hamming("A", "A", 1000)
    assert hamming("AAAA", "AAAA", 0)
    assert hamming("S", "P", 1)
    assert not hamming("S", "P", 0)


def test_hamming_scalar_length_mismatch() -> None:
    with pytest.raises(ValueError):
        hamming("AAA", "AA", 0)


def test_get_neighbors_hamming_strings_cutoff1() -> None:
    genotypes = ["AA", "AT", "TA", "TT"]
    expected = {(0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2)}
    arr = get_neighbors(genotypes, "hamming", cutoff=1)
    assert _edges_set(arr) == expected
    assert arr.dtype == np.int64
    assert arr.shape == (len(expected), 2)


def test_get_neighbors_hamming_cutoff2() -> None:
    # At cutoff=2 the 4 biallelic length-2 genotypes are all neighbors of each other.
    genotypes = ["AA", "AT", "TA", "TT"]
    arr = get_neighbors(genotypes, "hamming", cutoff=2)
    assert arr.shape == (12, 2)  # 4 * 3 directed edges


def test_get_neighbors_hamming_cutoff0_empty() -> None:
    arr = get_neighbors(["AA", "AT"], "hamming", cutoff=0)
    assert arr.shape == (0, 2)


def test_bitflip_matches_pairwise_cutoff1() -> None:
    # For biallelic data with cutoff=1, the bit-flip kernel and the pairwise
    # kernel must produce the same edge set.
    rng = np.random.default_rng(0)
    n, l = 32, 6
    bp = rng.integers(0, 2, size=(n, l), dtype=np.uint8)
    # Deduplicate rows so the neighbor relationship is well-defined.
    _, unique_idx = np.unique(bp, axis=0, return_index=True)
    bp = bp[np.sort(unique_idx)]
    bitflip = _rust.hamming_bitflip_neighbors(bp, 1)
    pairwise = _rust.hamming_neighbors_packed(bp, 1)
    assert _edges_set(bitflip) == _edges_set(pairwise)


def test_bitflip_matches_pairwise_cutoff2() -> None:
    rng = np.random.default_rng(1)
    n, l = 24, 5
    bp = rng.integers(0, 2, size=(n, l), dtype=np.uint8)
    _, unique_idx = np.unique(bp, axis=0, return_index=True)
    bp = bp[np.sort(unique_idx)]
    bitflip = _rust.hamming_bitflip_neighbors(bp, 2)
    pairwise = _rust.hamming_neighbors_packed(bp, 2)
    assert _edges_set(bitflip) == _edges_set(pairwise)


def test_bitflip_rejects_cutoff_gt_2() -> None:
    bp = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        _rust.hamming_bitflip_neighbors(bp, 3)


def test_bitflip_rejects_non_binary() -> None:
    bp = np.array([[0, 0], [0, 2]], dtype=np.uint8)
    with pytest.raises(ValueError):
        _rust.hamming_bitflip_neighbors(bp, 1)


def test_codon_neighbors_scalar() -> None:
    assert codon_neighbors("S", "P", 1)
    assert not codon_neighbors("S", "P", 0)
    assert not codon_neighbors("F", "G", 1)
    assert codon_neighbors("F", "G", 2)


def test_get_neighbors_codon_sequences() -> None:
    genotypes = ["SG", "PF", "SF", "PG"]
    # From v1: aa distance of 1 pairs are (S,G)-(P,G) and (P,F)-(S,F)
    arr = get_neighbors(genotypes, "codon", cutoff=1)
    edges = _edges_set(arr)
    assert (0, 3) in edges and (3, 0) in edges
    assert (1, 2) in edges and (2, 1) in edges


def test_get_neighbors_user_callable() -> None:
    def always_neighbor(a: str, b: str, cutoff: int) -> bool:
        return True

    genotypes = ["A", "B", "C"]
    arr = get_neighbors(genotypes, always_neighbor, cutoff=1)
    # 3 nodes, all pairs: 6 directed edges.
    assert arr.shape == (6, 2)


def test_get_neighbors_invalid_function() -> None:
    with pytest.raises(NeighborError):
        get_neighbors(["AA", "AT"], "bogus", cutoff=1)  # type: ignore[arg-type]


def test_get_neighbors_negative_cutoff() -> None:
    with pytest.raises(NeighborError):
        get_neighbors(["AA"], "hamming", cutoff=-1)


def test_hamming_neighbors_packed_matches_strings() -> None:
    rng = np.random.default_rng(2)
    n, l = 16, 4
    bp = rng.integers(0, 2, size=(n, l), dtype=np.uint8)
    genotypes = ["".join(str(x) for x in row) for row in bp]
    packed = _rust.hamming_neighbors_packed(bp, 2)
    strings = _rust.hamming_neighbors_strings(genotypes, 2)
    assert _edges_set(packed) == _edges_set(strings)
