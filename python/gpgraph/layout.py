"""Node layout helpers for GenotypePhenotypeGraph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gpgraph.base import GenotypePhenotypeGraph


def flattened(
    G: GenotypePhenotypeGraph,
    node_list: list[int] | None = None,
    scale: float = 1.0,
    vertical: bool = False,
) -> dict[int, tuple[float, float]]:
    """Compute flat Hamming-layer positions for a :class:`GenotypePhenotypeGraph`.

    Nodes are laid out in rows indexed by ``n_mutations`` (the Hamming weight
    from wildtype as reported by gpmap-v2). Within a row nodes are spaced
    evenly and centered on the origin.

    Parameters
    ----------
    G:
        A :class:`GenotypePhenotypeGraph` with a gpmap attached.
    node_list:
        Optional subset of node indices to position. Defaults to all nodes.
    scale:
        Spacing multiplier within a row. Must be > 0.
    vertical:
        If True, layers grow top-to-bottom; otherwise left-to-right.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    gpm_nodes = list(G.nodes) if node_list is None else list(node_list)

    if not set(gpm_nodes).issubset(set(G.nodes)):
        raise ValueError("node_list contains nodes not present in G")

    # Use the gpm-provided per-node mutation count instead of re-counting '1's.
    mutation_count = np.asarray(G.gpm.n_mutations, dtype=np.int64)

    level: dict[int, int] = {}
    for n in gpm_nodes:
        level[int(n)] = int(mutation_count[int(n)])

    # Count nodes per level, then distribute them evenly within the row.
    max_level = max(level.values())
    offsets: dict[int, np.ndarray] = {}
    for lvl in range(max_level + 1):
        count = sum(1 for v in level.values() if v == lvl)
        if count > 0:
            offsets[lvl] = np.arange(count, dtype=np.float64) - (count - 1) / 2.0

    # Stable assignment per row.
    cursor: dict[int, int] = {lvl: 0 for lvl in offsets}
    positions: dict[int, tuple[float, float]] = {}
    for n in gpm_nodes:
        lvl = level[int(n)]
        slot = offsets[lvl][cursor[lvl]] * scale
        cursor[lvl] += 1
        if vertical:
            positions[int(n)] = (float(slot), float(-lvl))
        else:
            positions[int(n)] = (float(lvl), float(slot))

    return positions


def bins(G: GenotypePhenotypeGraph) -> dict[int, list[int]]:
    """Group node indices by Hamming weight from wildtype."""
    mutation_count = np.asarray(G.gpm.n_mutations, dtype=np.int64)
    out: dict[int, list[int]] = {}
    for n in G.nodes:
        lvl = int(mutation_count[int(n)])
        out.setdefault(lvl, []).append(int(n))
    return out
