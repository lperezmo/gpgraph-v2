"""Tests for the flattened layer layout and bin helper."""

from __future__ import annotations

import pytest
from gpgraph import GenotypePhenotypeGraph
from gpgraph.layout import bins, flattened


def test_flattened_assigns_per_hamming_layer(tiny_graph: GenotypePhenotypeGraph) -> None:
    pos = flattened(tiny_graph, vertical=True)
    # node 0 is the wildtype (0 mutations) -> y == 0
    assert pos[0][1] == 0
    # node 7 is the triple mutant (3 mutations) -> y == -3
    assert pos[7][1] == -3


def test_flattened_horizontal(tiny_graph: GenotypePhenotypeGraph) -> None:
    pos = flattened(tiny_graph, vertical=False)
    assert pos[0][0] == 0
    assert pos[7][0] == 3


def test_flattened_bad_scale(tiny_graph: GenotypePhenotypeGraph) -> None:
    with pytest.raises(ValueError):
        flattened(tiny_graph, scale=0.0)


def test_flattened_bad_node_list(tiny_graph: GenotypePhenotypeGraph) -> None:
    with pytest.raises(ValueError):
        flattened(tiny_graph, node_list=[999])


def test_bins_counts_by_layer(tiny_graph: GenotypePhenotypeGraph) -> None:
    b = bins(tiny_graph)
    # 3-site biallelic: layers are 0: {0}, 1: {1, 2, 3}, 2: {4, 5, 6}, 3: {7}
    assert len(b[0]) == 1
    assert len(b[1]) == 3
    assert len(b[2]) == 3
    assert len(b[3]) == 1
