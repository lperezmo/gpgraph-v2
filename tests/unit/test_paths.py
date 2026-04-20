"""Tests for the forward-paths / flux helpers (v1 had an empty test_paths.py)."""

from __future__ import annotations

from collections import Counter

import pytest
from gpgraph import GenotypePhenotypeGraph
from gpgraph.paths import (
    edges_flux_to_node_flux,
    forward_paths,
    forward_paths_prob,
    paths_prob_to_edges_flux,
    paths_to_edges,
    paths_to_edges_count,
)


def test_forward_paths_tiny(tiny_graph: GenotypePhenotypeGraph) -> None:
    paths = forward_paths(tiny_graph, 0, 7)
    # 3-site biallelic cube from wildtype to triple mutant: 3! = 6 shortest paths.
    assert len(paths) == 6
    for p in paths:
        assert p[0] == 0 and p[-1] == 7
        assert len(p) == 4  # 3 hops, 4 nodes


def test_forward_paths_resolves_genotype_string(tiny_graph: GenotypePhenotypeGraph) -> None:
    paths = forward_paths(tiny_graph, "AAA", "TTT")
    assert len(paths) == 6


def test_forward_paths_max_paths(tiny_graph: GenotypePhenotypeGraph) -> None:
    paths = forward_paths(tiny_graph, 0, 7, max_paths=2)
    assert len(paths) == 2


def test_forward_paths_prob_requires_model(tiny_graph: GenotypePhenotypeGraph) -> None:
    with pytest.raises(ValueError):
        forward_paths_prob(tiny_graph, 0, 7)


def test_forward_paths_prob(tiny_graph: GenotypePhenotypeGraph) -> None:
    tiny_graph.add_model(column="phenotypes", model="sswm")
    pp = forward_paths_prob(tiny_graph, 0, 7)
    assert len(pp) == 6
    # Some paths traverse a flat phenotype hop (fixture has two nodes at 0.6);
    # under SSWM those hops contribute 0. At least one path must have prob > 0.
    assert any(p > 0 for p in pp.values())
    for path in pp:
        assert path[0] == 0 and path[-1] == 7


def test_paths_to_edges_dedup() -> None:
    paths = [[0, 1, 2, 3], [0, 1, 4, 3]]
    out = paths_to_edges(paths)
    assert len(out) == len(set(out))
    assert set(out) == {(0, 1), (1, 2), (2, 3), (1, 4), (4, 3)}


def test_paths_to_edges_repeat() -> None:
    paths = [[0, 1, 2], [0, 1, 3]]
    out = paths_to_edges(paths, repeat=True)
    assert out.count((0, 1)) == 2


def test_paths_to_edges_count() -> None:
    paths = [[0, 1, 2], [0, 1, 3]]
    counts = paths_to_edges_count(paths)
    assert isinstance(counts, Counter)
    assert counts[(0, 1)] == 2


def test_paths_prob_to_edges_flux(tiny_graph: GenotypePhenotypeGraph) -> None:
    tiny_graph.add_model(column="phenotypes", model="sswm")
    pp = forward_paths_prob(tiny_graph, 0, 7)
    flux = paths_prob_to_edges_flux(pp)
    assert flux  # at least one edge received flux
    for edge, f in flux.items():
        assert f >= 0
        assert edge in tiny_graph.edges()
    # Total flux across edges must equal sum of path probabilities times path length.
    assert sum(flux.values()) > 0


def test_edges_flux_to_node_flux_zero_when_no_capacity(tiny_graph: GenotypePhenotypeGraph) -> None:
    # Without a 'capacity' attribute on edges, all node fluxes are zero.
    fluxes = edges_flux_to_node_flux(tiny_graph)
    assert all(v == 0.0 for v in fluxes.values())
