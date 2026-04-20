"""pytest-benchmark suite for the hamming kernels.

Run with ``uv run pytest tests/benchmarks/ --benchmark-only``. These are
excluded from the regular test run (default `pytest.ini_options` / CI do
not set ``--benchmark-only``, so pytest-benchmark records stats without
failing on them).
"""

from __future__ import annotations

import numpy as np
import pytest
from gpgraph import _rust


def _python_pairwise(genotypes: list[str], cutoff: int) -> int:
    """Pure-python O(N^2) reference for the same problem the Rust kernel solves."""
    n = len(genotypes)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = sum(a != b for a, b in zip(genotypes[i], genotypes[j], strict=True))
            if d <= cutoff:
                count += 2
    return count


@pytest.mark.parametrize("n,l", [(256, 6), (1024, 8)])
def test_bench_packed_pairwise(benchmark, n: int, l: int) -> None:
    rng = np.random.default_rng(0)
    bp = rng.integers(0, 2, size=(n, l), dtype=np.uint8)
    benchmark(_rust.hamming_neighbors_packed, bp, 1)


@pytest.mark.parametrize("n,l", [(256, 6), (1024, 8)])
def test_bench_bitflip(benchmark, n: int, l: int) -> None:
    rng = np.random.default_rng(0)
    bp = rng.integers(0, 2, size=(n, l), dtype=np.uint8)
    _, idx = np.unique(bp, axis=0, return_index=True)
    bp = bp[np.sort(idx)]
    benchmark(_rust.hamming_bitflip_neighbors, bp, 1)


@pytest.mark.parametrize("n,l", [(64, 6), (256, 6)])
def test_bench_python_baseline(benchmark, n: int, l: int) -> None:
    rng = np.random.default_rng(0)
    bp = rng.integers(0, 2, size=(n, l), dtype=np.uint8)
    genotypes = ["".join(str(int(x)) for x in row) for row in bp]
    benchmark(_python_pairwise, genotypes, 1)
