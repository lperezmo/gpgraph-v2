"""GenotypePhenotypeGraph: NetworkX DiGraph view of a gpmap-v2 GenotypePhenotypeMap."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from gpgraph import fixation, neighbors
from gpgraph.exceptions import GpgraphError

if TYPE_CHECKING:
    from gpmap import GenotypePhenotypeMap


class GenotypePhenotypeGraph(nx.DiGraph):  # type: ignore[misc]
    """A NetworkX directed graph wrapping a gpmap-v2 :class:`GenotypePhenotypeMap`.

    Construction is only supported through :meth:`from_gpm`. The v1
    ``add_gpm`` post-construction injection is removed. Each node carries the
    attributes of its row in ``gpm.data`` (``genotypes``, ``phenotypes``,
    ``binary``, ``n_mutations``, plus any user-added columns).

    Use :meth:`add_model` to populate the ``prob`` edge attribute with a
    fixation model over a chosen fitness column.
    """

    def __init__(self) -> None:
        super().__init__()
        self._gpm: GenotypePhenotypeMap | None = None
        self._fixation_model: Callable[..., Any] | None = None
        self._fixation_model_params: dict[str, Any] = {}

    # Node and edge contract: see SCHEMA.md.

    @property
    def gpm(self) -> GenotypePhenotypeMap:
        if self._gpm is None:
            raise GpgraphError(
                "GenotypePhenotypeGraph has no gpmap attached; use .from_gpm(gpm) to construct."
            )
        return self._gpm

    def __repr__(self) -> str:  # no draw side effect, unlike v1
        if self._gpm is None:
            return "<GenotypePhenotypeGraph (empty)>"
        n = self.number_of_nodes()
        m = self.number_of_edges()
        return f"<GenotypePhenotypeGraph n_nodes={n} n_edges={m}>"

    @classmethod
    def from_gpm(
        cls,
        gpm: GenotypePhenotypeMap,
        neighbor_function: str | Callable[..., bool] = "hamming",
        cutoff: int = 1,
    ) -> GenotypePhenotypeGraph:
        """Build a :class:`GenotypePhenotypeGraph` from a gpmap-v2 map.

        Parameters
        ----------
        gpm:
            A ``gpmap.GenotypePhenotypeMap`` instance.
        neighbor_function:
            ``"hamming"``, ``"codon"``, or a ``f(g1, g2, cutoff=...) -> bool`` callable.
        cutoff:
            Maximum neighbor distance (integer >= 0).
        """
        cls._check_gpm(gpm)

        g = cls()
        g._gpm = gpm

        # Populate nodes from gpm.data rows.
        data = gpm.data
        for i, row in data.iterrows():
            g.add_node(int(i), **{col: row[col] for col in data.columns})

        # Pick the fastest available neighbor kernel.
        binary_packed: NDArray[np.uint8] | None = None
        if neighbor_function == "hamming":
            # gpmap-v2 exposes binary_packed (uint8 2D). Use it when alphabet is binary.
            try:
                binary_packed = np.ascontiguousarray(gpm.binary_packed, dtype=np.uint8)
            except AttributeError:  # pragma: no cover - defensive
                binary_packed = None

        genotypes = list(gpm.genotypes)
        edges = neighbors.get_neighbors(
            genotypes,
            neighbor_function=neighbor_function,
            cutoff=cutoff,
            binary_packed=binary_packed,
        )
        # NetworkX will happily accept a numpy (E, 2) array of int64 edge pairs;
        # add_edges_from wants an iterable of tuples, so iterate rows.
        g.add_edges_from((int(a), int(b)) for a, b in edges)

        return g

    def add_model(
        self,
        column: str | None = None,
        model: str | Callable[..., Any] | None = None,
        **model_params: Any,
    ) -> None:
        """Populate the edge attribute ``prob`` with a fixation model's output.

        Parameters
        ----------
        column:
            Column in ``gpm.data`` to use as the node fitness. If ``None``,
            every node is given fitness 1.0 (so every edge gets the model's
            value at ``f=(1.0, 1.0)``).
        model:
            One of ``"sswm"``, ``"ratio"``, ``"moran"``, ``"mcclandish"``,
            or a callable ``f(fi, fj, **kwargs)``. ``None`` assigns 1.0 to
            every edge.
        **model_params:
            Extra keyword arguments forwarded to the model on every call
            (e.g. ``population_size`` for ``moran`` / ``mcclandish``).
        """
        if self._gpm is None:
            raise GpgraphError("call from_gpm before add_model")
        gpm = self._gpm

        if column is None:
            values = np.ones(len(gpm.genotypes), dtype=np.float64)
        else:
            try:
                values = np.asarray(gpm.data[column].to_numpy(), dtype=np.float64)
            except KeyError as exc:
                raise GpgraphError(f"column {column!r} not in gpm.data") from exc

        resolved = self._resolve_model(model)

        self._fixation_model = resolved
        self._fixation_model_params = dict(model_params)

        if self.number_of_edges() == 0:
            return

        # Vectorized edge-wise evaluation.
        edge_pairs = np.asarray(list(self.edges()), dtype=np.int64)
        src = edge_pairs[:, 0]
        dst = edge_pairs[:, 1]
        fi = values[src]
        fj = values[dst]

        if resolved is _const_one:
            probs = np.ones(fi.shape, dtype=np.float64)
        else:
            try:
                probs = np.asarray(resolved(fi, fj, **model_params), dtype=np.float64)
            except TypeError:
                # User-supplied callable that only accepts scalars; fall back to Python loop.
                probs = np.fromiter(
                    (
                        float(resolved(float(a), float(b), **model_params))
                        for a, b in zip(fi, fj, strict=True)
                    ),
                    dtype=np.float64,
                    count=fi.size,
                )
            if probs.shape != fi.shape:
                probs = np.broadcast_to(probs, fi.shape)

        nx.set_edge_attributes(
            self,
            {
                (int(a), int(b)): {"prob": float(p)}
                for (a, b), p in zip(edge_pairs, probs, strict=True)
            },
        )

    def model(self, v1: float, v2: float) -> float:
        """Evaluate the stored fixation model at a single (v1, v2) pair."""
        if self._fixation_model is None:
            raise GpgraphError("add_model has not been called yet")
        return float(self._fixation_model(v1, v2, **self._fixation_model_params))

    @classmethod
    def read_json(cls, fname: str) -> GenotypePhenotypeGraph:
        """Read a graph from a gpmap-v2 JSON file."""
        from gpmap import read_json  # local import keeps cold-import lean

        gpm = read_json(fname)
        return cls.from_gpm(gpm)

    @classmethod
    def read_csv(cls, fname: str) -> GenotypePhenotypeGraph:
        """Read a graph from a gpmap-v2 CSV sidecar bundle."""
        from gpmap import read_csv

        gpm = read_csv(fname)
        return cls.from_gpm(gpm)

    # ------------- helpers

    @staticmethod
    def _check_gpm(gpm: object) -> None:
        from gpmap import GenotypePhenotypeMap

        if not isinstance(gpm, GenotypePhenotypeMap):
            raise GpgraphError(
                f"gpm must be a gpmap.GenotypePhenotypeMap, got {type(gpm).__name__}"
            )

    @staticmethod
    def _resolve_model(
        model: str | Callable[..., Any] | None,
    ) -> Callable[..., Any]:
        if model is None:
            return _const_one
        if callable(model):
            return model
        try:
            return fixation.MODEL_REGISTRY[model]  # type: ignore[return-value]
        except KeyError as exc:
            choices = ", ".join(sorted(fixation.MODEL_REGISTRY))
            raise GpgraphError(
                f"unknown model {model!r}; expected a callable or one of: {choices}"
            ) from exc


def _const_one(*_args: Any, **_kwargs: Any) -> float:
    """Sentinel default model: every edge gets probability 1.0."""
    return 1.0
