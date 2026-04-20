"""Forward-path enumeration and flux helpers for GenotypePhenotypeGraph."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from gpgraph.base import GenotypePhenotypeGraph


def _coerce_endpoint(G: GenotypePhenotypeGraph, endpoint: int | str) -> int:
    """Resolve a node id, a genotype string, or a binary-encoded string to the node index."""
    if isinstance(endpoint, (int, np.integer)):
        return int(endpoint)
    if isinstance(endpoint, str):
        genotypes = list(G.gpm.genotypes)
        if endpoint in genotypes:
            return genotypes.index(endpoint)
        binary_arr = G.gpm.binary
        binary_list = [str(b) for b in binary_arr]
        if endpoint in binary_list:
            return binary_list.index(endpoint)
    raise ValueError(f"could not resolve endpoint {endpoint!r} to a node index")


def forward_paths(
    G: GenotypePhenotypeGraph,
    source: int | str,
    target: int | str,
    max_paths: int | None = None,
) -> list[list[int]]:
    """Return all shortest forward paths from ``source`` to ``target``.

    Parameters
    ----------
    G:
        A :class:`GenotypePhenotypeGraph`.
    source, target:
        Node index, genotype string, or binary string identifying each endpoint.
    max_paths:
        Optional cap on the number of returned paths. The number of shortest
        paths in an orthotope grows combinatorially, so this is a safety rail.
    """
    src = _coerce_endpoint(G, source)
    tgt = _coerce_endpoint(G, target)
    paths_iter = nx.all_shortest_paths(G, source=src, target=tgt)
    if max_paths is None:
        return [list(p) for p in paths_iter]

    out: list[list[int]] = []
    for i, p in enumerate(paths_iter):
        if i >= max_paths:
            break
        out.append(list(p))
    return out


def forward_paths_prob(
    G: GenotypePhenotypeGraph,
    source: int | str,
    target: int | str,
    max_paths: int | None = None,
) -> dict[tuple[int, ...], float]:
    """Return a dict of shortest-path tuples -> product of edge ``prob`` attributes."""
    paths = forward_paths(G, source, target, max_paths=max_paths)
    out: dict[tuple[int, ...], float] = {}
    for path in paths:
        p = 1.0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            try:
                p *= float(G.edges[edge]["prob"])
            except KeyError as exc:
                raise ValueError(
                    f"edge {edge} has no 'prob' attribute; call add_model first"
                ) from exc
        out[tuple(path)] = p
    return out


def paths_to_edges(
    paths: list[list[int]] | list[tuple[int, ...]],
    repeat: bool = False,
) -> list[tuple[int, int]]:
    """Flatten a list of paths to a list of edges. Drop duplicates unless ``repeat``."""
    edges: list[tuple[int, int]] = []
    for path in paths:
        edges.extend((int(path[i]), int(path[i + 1])) for i in range(len(path) - 1))
    if repeat:
        return edges
    seen: set[tuple[int, int]] = set()
    dedup: list[tuple[int, int]] = []
    for e in edges:
        if e not in seen:
            seen.add(e)
            dedup.append(e)
    return dedup


def paths_to_edges_count(paths: list[list[int]]) -> Counter[tuple[int, int]]:
    """Count edge visits across a list of paths."""
    return Counter(paths_to_edges(paths, repeat=True))


def paths_prob_to_edges_flux(
    paths_prob: dict[tuple[int, ...], float],
) -> dict[tuple[int, int], float]:
    """Sum path probabilities onto their constituent edges."""
    flux: dict[tuple[int, int], float] = {}
    for path, prob in paths_prob.items():
        for i in range(len(path) - 1):
            edge = (int(path[i]), int(path[i + 1]))
            flux[edge] = flux.get(edge, 0.0) + float(prob)
    return flux


def edges_flux_to_node_flux(G: GenotypePhenotypeGraph) -> dict[int, float]:
    """Sum the ``capacity`` attribute of incoming edges for each node."""
    out: dict[int, float] = {}
    for node in G.nodes:
        acc = 0.0
        for _u, _v, cap in G.in_edges(node, data="capacity"):
            if cap:
                acc += float(cap)
        out[int(node)] = acc
    return out
