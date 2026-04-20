"""Property-based tests for the neighbor kernels.

Hamming neighborhood is a symmetric relation. The three Rust entry points
(packed, bit-flip, strings) must agree on biallelic inputs. All three must
agree with a Python reference on arbitrary alphabets.
"""

from __future__ import annotations

import numpy as np
from gpgraph import _rust
from hypothesis import given, settings
from hypothesis import strategies as st


def _python_hamming_edges(genotypes: list[str], cutoff: int) -> set[tuple[int, int]]:
    n = len(genotypes)
    out: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = sum(a != b for a, b in zip(genotypes[i], genotypes[j], strict=True))
            if d <= cutoff:
                out.add((i, j))
                out.add((j, i))
    return out


def _edges(arr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in arr}


@settings(max_examples=50, deadline=None)
@given(
    length=st.integers(min_value=1, max_value=6),
    n=st.integers(min_value=1, max_value=20),
    cutoff=st.integers(min_value=0, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_packed_hamming_matches_python(length: int, n: int, cutoff: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    bp = rng.integers(0, 2, size=(n, length), dtype=np.uint8)
    genotypes = ["".join(str(int(x)) for x in row) for row in bp]
    rust_out = _edges(_rust.hamming_neighbors_packed(bp, cutoff))
    py_out = _python_hamming_edges(genotypes, cutoff)
    assert rust_out == py_out


@settings(max_examples=50, deadline=None)
@given(
    length=st.integers(min_value=1, max_value=6),
    n=st.integers(min_value=1, max_value=16),
    cutoff=st.integers(min_value=0, max_value=2),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_bitflip_matches_python(length: int, n: int, cutoff: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    bp = rng.integers(0, 2, size=(n, length), dtype=np.uint8)
    _, unique_idx = np.unique(bp, axis=0, return_index=True)
    bp = bp[np.sort(unique_idx)]
    genotypes = ["".join(str(int(x)) for x in row) for row in bp]
    rust_out = _edges(_rust.hamming_bitflip_neighbors(bp, cutoff))
    py_out = _python_hamming_edges(genotypes, cutoff)
    assert rust_out == py_out


@settings(max_examples=40, deadline=None)
@given(
    length=st.integers(min_value=1, max_value=5),
    n=st.integers(min_value=1, max_value=12),
    alphabet=st.text(alphabet="ABCD", min_size=2, max_size=4).map(
        lambda s: "".join(sorted(set(s)))
    ),
    cutoff=st.integers(min_value=0, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_hamming_strings_matches_python(
    length: int, n: int, alphabet: str, cutoff: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    genotypes = ["".join(rng.choice(list(alphabet), size=length)) for _ in range(n)]
    rust_out = _edges(_rust.hamming_neighbors_strings(genotypes, cutoff))
    py_out = _python_hamming_edges(genotypes, cutoff)
    assert rust_out == py_out


def test_hamming_symmetry_axiom() -> None:
    """For every (i, j) in the edge list there must be a (j, i)."""
    rng = np.random.default_rng(7)
    bp = rng.integers(0, 2, size=(30, 5), dtype=np.uint8)
    for cutoff in (1, 2, 3):
        arr = _rust.hamming_neighbors_packed(bp, cutoff)
        edges = _edges(arr)
        for a, b in edges:
            assert (b, a) in edges
