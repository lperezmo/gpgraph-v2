# gpgraph-v2

[![CI](https://github.com/lperezmo/gpgraph-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/lperezmo/gpgraph-v2/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gpgraph-v2.svg)](https://pypi.org/project/gpgraph-v2/)
[![Python](https://img.shields.io/pypi/pyversions/gpgraph-v2.svg)](https://pypi.org/project/gpgraph-v2/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gpgraph-v2.streamlit.app)

NetworkX-backed, Rust-accelerated graphs over genotype-phenotype maps.

gpgraph-v2 lifts a [gpmap-v2](https://github.com/lperezmo/gpmap-v2) `GenotypePhenotypeMap` into a
NetworkX `DiGraph`. Nodes carry the row attributes from the map; edges connect neighbors under
a chosen distance metric (Hamming or codon). Fixation models (SSWM, ratio, Moran, McCandlish)
populate an edge `prob` attribute for evolutionary trajectory analysis.

This is a clean-break rewrite of [harmslab/gpgraph](https://github.com/harmslab/gpgraph).
Hot paths live in Rust via PyO3 + rayon; fixation models stay in vectorized numpy.

## Why v2

- **Fast.** Neighbor detection runs in Rust with rayon parallelism. Biallelic cutoff-1 and
  cutoff-2 hit a bit-flip fast path: `O(N * L^cutoff)` instead of `O(N^2 * L)`.
- **Typed.** Full type hints, `mypy --strict` in CI.
- **Modern tooling.** `uv` + `maturin` + `pyproject.toml`. Releases via
  `python-semantic-release`. OIDC-based PyPI publishing.
- **Consumes gpmap-v2.** Speaks the locked `SCHEMA.md` contract; reads `binary_packed`,
  `n_mutations`, and `phenotypes` (not the removed v1 `fitnesses`).
- **No Cython, no `setup.py`, no `.c` blobs.**

## Install

```bash
pip install gpgraph-v2
```

Or with uv:

```bash
uv add gpgraph-v2
```

Plotting support is optional. For matplotlib:

```bash
pip install "gpgraph-v2[plot]"
```

Python 3.11+. Prebuilt wheels ship for Linux (x86_64, aarch64), macOS (x86_64, aarch64),
and Windows (x64).

## Quick start

```python
from gpmap import GenotypePhenotypeMap
from gpgraph import GenotypePhenotypeGraph

gpm = GenotypePhenotypeMap(
    wildtype="AAA",
    genotypes=["AAA", "AAT", "ATA", "TAA", "ATT", "TAT", "TTA", "TTT"],
    phenotypes=[0.1, 0.2, 0.2, 0.6, 0.4, 0.6, 1.0, 1.1],
    stdeviations=[0.05] * 8,
)

G = GenotypePhenotypeGraph.from_gpm(gpm)
G.number_of_nodes()  # 8
G.number_of_edges()  # 24 directed (12 undirected pairs)

G.add_model(column="phenotypes", model="sswm")
G.edges[(0, 1)]["prob"]  # SSWM fixation probability from AAA -> AAT
```

Available fixation models: `"sswm"`, `"ratio"`, `"moran"`, `"mcclandish"`. Pass
`population_size=N` as a keyword for the last two. Pass any `f(fi, fj, **kw) -> float`
for a custom model.

## Forward paths and flux

```python
from gpgraph.paths import forward_paths_prob, paths_prob_to_edges_flux

G.add_model(column="phenotypes", model="sswm")
paths = forward_paths_prob(G, source=0, target=7)
flux = paths_prob_to_edges_flux(paths)
```

## Plotting

```python
from gpgraph.pyplot import draw_gpgraph, draw_paths

fig, ax = draw_gpgraph(G)             # Hamming-weight vertical layout
draw_paths(G, source=0, target=7)     # paths flux overlay
```

## Benchmarks

Rust kernels vs a pure-Python pairwise reference. Windows 11, Ryzen, release build.
Run via `uv run pytest tests/benchmarks/ --benchmark-only`.

| problem              | bit-flip (Rust) | pairwise (Rust) | pairwise (Python) |
|---|---|---|---|
| N=256, L=6, cutoff=1 | 77 us           | 284 us          | 29.3 ms           |
| N=1024, L=8, cutoff=1| 167 us          | 1.2 ms          | (skipped, minutes)|

At L=6 the Rust bit-flip path is ~380x faster than a naive Python loop. Past
N = 1000 on biallelic data the bit-flip specialization pulls away from the
rayon-parallel pairwise version by another ~7x.

## Dispatch policy (neighbor kernels)

`gpgraph.neighbors.get_neighbors` chooses the fastest available implementation:

| problem shape                                     | kernel |
|---|---|
| user-supplied `f(g1, g2, cutoff=...) -> bool`     | pure Python `O(N^2)` |
| Hamming, biallelic `binary_packed`, cutoff <= 2   | Rust bit-flip (`O(N * C(L, cutoff))`) |
| Hamming, biallelic `binary_packed`, larger cutoff | Rust rayon-parallel packed pairwise |
| Hamming, non-binary alphabet                      | Rust rayon-parallel string pairwise |
| codon                                             | Rust rayon-parallel codon pairwise |

See [SCHEMA.md](SCHEMA.md) for the full node/edge contract.

## Development

```bash
git clone https://github.com/lperezmo/gpgraph-v2
cd gpgraph-v2
uv sync
uv run maturin develop --release
uv run pytest
uv run ruff check python/gpgraph tests
```

After editing Rust:

```bash
uv run maturin develop --release && uv run pytest
```

## Consuming from another local project

During co-development with gpmap-v2, point at the local source:

```toml
[tool.uv.sources]
gpgraph-v2 = { path = "/absolute/path/to/gpgraph-v2", editable = true }

[project]
dependencies = ["gpgraph-v2"]
```

Imports remain `from gpgraph import GenotypePhenotypeGraph`.

## Migration from v1 (`harmslab/gpgraph`)

gpgraph-v2 is not wire-compatible with v1. Key differences:

- Distribution name is `gpgraph-v2` on PyPI; import path is still `gpgraph`.
- Construction: use `GenotypePhenotypeGraph.from_gpm(gpm, ...)`. The v1 two-step
  `G = GenotypePhenotypeGraph(); G.add_gpm(gpm)` is removed.
- Reads gpmap-v2 columns: node attribute is `phenotypes`, not `fitnesses`.
- `__repr__` no longer renders a matplotlib figure as a side effect.
- Matplotlib is an optional extra (`gpgraph-v2[plot]`); the core install is headless.
- The `_nx_wrapper` introspection shim in `pyplot/primitives.py` is gone. NetworkX is
  version-pinned to `>=3.6`.
- `gpgraph.neighbor_functions` is renamed `gpgraph.neighbors` and the pairwise
  `get_neighbors` returns a NumPy `(E, 2) int64` array instead of a list of tuples.
- Python 3.11+.

## Releases

Releases are driven by [`python-semantic-release`](https://python-semantic-release.readthedocs.io/)
on merge to `main`. Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

- `fix: ...` -> patch
- `feat: ...` -> minor
- `feat!: ...` or a `BREAKING CHANGE:` footer -> major

`CHANGELOG.md`, version bumps, Git tags, GitHub Releases, wheel builds, and PyPI uploads
all happen automatically.

## License

MIT. See [LICENSE](LICENSE).
