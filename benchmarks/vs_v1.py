"""Benchmark gpgraph-v2 graph construction and SSWM model assignment.

To reproduce the v1 vs v2 comparison from the README, run in separate environments:

    # v1 environment
    uv venv .venv-v1 && uv pip install --python .venv-v1/Scripts/python.exe gpgraph==0.2.0 gpmap==0.7.0 networkx matplotlib
    .venv-v1/Scripts/python benchmarks/vs_v1.py

    # v2 environment
    uv run python benchmarks/vs_v1.py

v1 note: graph construction in v1 uses an O(N^2 * L) pure-Python neighbor search.
L=14+ is impractical with v1; sizes above L=12 are v2-only in the comparison.
"""
from __future__ import annotations

import itertools
import json
import timeit
from pathlib import Path

import numpy as np

SIZES = [8, 10, 12, 14, 16]
N_REPEATS = 3


def make_gpm(L: int):
    from gpmap import GenotypePhenotypeMap

    genotypes = ["".join(g) for g in itertools.product("AT", repeat=L)]
    # SSWM requires positive fitness values
    phenotypes = np.exp(np.random.default_rng(42).normal(size=len(genotypes)) * 0.3)
    return GenotypePhenotypeMap(wildtype="A" * L, genotypes=genotypes, phenotypes=phenotypes)


def best_ms(fn, n: int) -> float:
    return min(timeit.repeat(fn, number=1, repeat=n)) * 1000


try:
    from gpgraph import __version__ as _ver
except Exception:
    _ver = "unknown"

results: dict = {"version": _ver, "results": {}}
print(f"gpgraph {_ver}")

print("build graph")
build_ms: dict[str, float] = {}
for L in SIZES:
    gpm = make_gpm(L)
    try:
        from gpgraph import GenotypePhenotypeGraph
        t = best_ms(lambda: GenotypePhenotypeGraph.from_gpm(gpm), N_REPEATS)
    except AttributeError:
        t = best_ms(lambda: GenotypePhenotypeGraph(gpm), N_REPEATS)
    build_ms[f"L{L}"] = round(t, 4)
    print(f"  L={L:2d} ({2**L:6d} genotypes): {t:.3f} ms")
results["results"]["build_graph_ms"] = build_ms

print("add_model sswm")
sswm_ms: dict[str, float] = {}
for L in SIZES:
    gpm = make_gpm(L)
    from gpgraph import GenotypePhenotypeGraph

    def run_sswm() -> None:
        try:
            G = GenotypePhenotypeGraph.from_gpm(gpm)
            G.add_model(column="phenotypes", model="sswm")
        except AttributeError:
            from gpgraph.models import strong_selection_weak_mutation
            G = GenotypePhenotypeGraph(gpm)
            G.add_model(model=strong_selection_weak_mutation)

    t = best_ms(run_sswm, N_REPEATS)
    sswm_ms[f"L{L}"] = round(t, 4)
    print(f"  L={L:2d} ({2**L:6d} genotypes): {t:.3f} ms")
results["results"]["add_model_sswm_ms"] = sswm_ms

out = Path(__file__).parent / f"results_{_ver.replace('.', '_')}.json"
out.write_text(json.dumps(results, indent=2))
print(f"Saved {out}")
