"""Tests for the GenotypePhenotypeGraph class."""

from __future__ import annotations

import pytest
from gpgraph import GenotypePhenotypeGraph
from gpgraph.exceptions import GpgraphError


def test_from_gpm_builds_graph(tiny_graph: GenotypePhenotypeGraph) -> None:
    assert tiny_graph.number_of_nodes() == 8
    # 3-site biallelic, cutoff=1: 12 undirected edges = 24 directed.
    assert tiny_graph.number_of_edges() == 24


def test_repr_has_no_side_effect(tiny_graph: GenotypePhenotypeGraph) -> None:
    # v1 __repr__ called draw_gpgraph (opened a matplotlib figure). Regression
    # guard: just call repr and confirm it returns a string without raising.
    s = repr(tiny_graph)
    assert "GenotypePhenotypeGraph" in s


def test_repr_empty_graph() -> None:
    G = GenotypePhenotypeGraph()
    assert "empty" in repr(G)


def test_gpm_access_requires_from_gpm() -> None:
    G = GenotypePhenotypeGraph()
    with pytest.raises(GpgraphError):
        _ = G.gpm


def test_from_gpm_rejects_non_gpm() -> None:
    with pytest.raises(GpgraphError):
        GenotypePhenotypeGraph.from_gpm(object())  # type: ignore[arg-type]


def test_add_model_sswm_populates_prob(tiny_graph: GenotypePhenotypeGraph) -> None:
    tiny_graph.add_model(column="phenotypes", model="sswm")
    # Forward edges (lower phenotype -> higher) should have positive prob,
    # backward edges should have prob 0 under SSWM.
    for u, v, data in tiny_graph.edges(data=True):
        assert "prob" in data
        fu = tiny_graph.nodes[u]["phenotypes"]
        fv = tiny_graph.nodes[v]["phenotypes"]
        if fv > fu:
            assert data["prob"] > 0
        elif fv < fu:
            assert data["prob"] == 0


def test_add_model_none_yields_unit(tiny_graph: GenotypePhenotypeGraph) -> None:
    tiny_graph.add_model()
    for _, _, data in tiny_graph.edges(data=True):
        assert data["prob"] == 1.0


def test_add_model_bad_column_raises(tiny_graph: GenotypePhenotypeGraph) -> None:
    with pytest.raises(GpgraphError):
        tiny_graph.add_model(column="does_not_exist", model="sswm")


def test_add_model_bad_model_raises(tiny_graph: GenotypePhenotypeGraph) -> None:
    with pytest.raises(GpgraphError):
        tiny_graph.add_model(column="phenotypes", model="bogus")  # type: ignore[arg-type]


def test_model_method_requires_add_model(tiny_graph: GenotypePhenotypeGraph) -> None:
    with pytest.raises(GpgraphError):
        tiny_graph.model(1.0, 2.0)


def test_model_method_evaluates(tiny_graph: GenotypePhenotypeGraph) -> None:
    tiny_graph.add_model(column="phenotypes", model="sswm")
    assert tiny_graph.model(0.5, 1.0) > 0.0
