"""Shared pytest fixtures for gpgraph-v2 tests."""

from __future__ import annotations

import numpy as np
import pytest
from gpgraph import GenotypePhenotypeGraph
from gpmap import GenotypePhenotypeMap


@pytest.fixture
def tiny_gpm() -> GenotypePhenotypeMap:
    """A complete 3-site biallelic map with 8 genotypes."""
    return GenotypePhenotypeMap(
        wildtype="AAA",
        genotypes=["AAA", "AAT", "ATA", "TAA", "ATT", "TAT", "TTA", "TTT"],
        phenotypes=[0.1, 0.2, 0.2, 0.6, 0.4, 0.6, 1.0, 1.1],
        stdeviations=[0.05] * 8,
    )


@pytest.fixture
def tiny_graph(tiny_gpm: GenotypePhenotypeMap) -> GenotypePhenotypeGraph:
    """The tiny_gpm wrapped in a graph with cutoff=1 Hamming edges."""
    return GenotypePhenotypeGraph.from_gpm(tiny_gpm)


@pytest.fixture
def complete_binary_gpm_l4() -> GenotypePhenotypeMap:
    """A complete 4-site biallelic map (16 genotypes) with deterministic phenotypes."""
    genotypes: list[str] = []
    phenotypes: list[float] = []
    for i in range(16):
        g = "".join("T" if (i >> k) & 1 else "A" for k in range(4))
        genotypes.append(g)
        phenotypes.append(0.1 + 0.1 * bin(i).count("1"))
    return GenotypePhenotypeMap(
        wildtype="AAAA",
        genotypes=genotypes,
        phenotypes=phenotypes,
        stdeviations=[0.01] * 16,
    )


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
