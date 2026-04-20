# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this
project adheres to [Semantic Versioning](https://semver.org/). Releases are cut
automatically by `python-semantic-release` from Conventional Commits.

## [0.1.0] - 2026-04-20

Initial clean-break release. Not yet published to PyPI.

### Added

- `GenotypePhenotypeGraph` subclassing `networkx.DiGraph`, constructed via
  `GenotypePhenotypeGraph.from_gpm(gpm, neighbor_function=..., cutoff=...)`.
- Rust hot-path crate `gpgraph-core`, exposed as `gpgraph._rust`:
  - `hamming_neighbors_packed` (rayon pairwise on `binary_packed`)
  - `hamming_bitflip_neighbors` (O(N * C(L, cutoff)) fast path for biallelic
    cutoff 1 or 2)
  - `hamming_neighbors_strings` (rayon pairwise over arbitrary alphabets)
  - `codon_neighbors` (rayon pairwise with precomputed AA min-bp distance table)
- Vectorized numpy fixation models in `gpgraph.fixation`: `strong_selection_weak_mutation`
  (`sswm`), `ratio`, `moran`, `mcclandish`, with overflow-protection branches
  that preserve the scalar v1 behavior.
- `add_model` applies the chosen fixation model over all edges in a single
  numpy pass.
- `gpgraph.paths`: `forward_paths`, `forward_paths_prob` (with optional
  `max_paths` cap), `paths_to_edges`, `paths_to_edges_count`,
  `paths_prob_to_edges_flux`, `edges_flux_to_node_flux`.
- `gpgraph.layout.flattened` and `gpgraph.layout.bins` driven by
  `gpmap.n_mutations` instead of counting "1" characters in binary strings.
- Optional matplotlib drawing layer under `gpgraph.pyplot` (`gpgraph-v2[plot]`):
  merged `draw_gpgraph` (v1 had a broken duplicate in `draw.py` that was
  deleted), `draw_paths`, and primitives wrappers around `networkx.draw_networkx_*`.
- Locked node/edge contract in `SCHEMA.md`.
- CI matrix: Python 3.11 / 3.12 / 3.13 on Ubuntu / macOS / Windows.
- Release pipeline: `python-semantic-release` + maturin wheel matrix +
  OIDC-trusted PyPI publish.

### Changed

- Build system moved from `setup.py` to `pyproject.toml` + maturin.
- Package distribution name is `gpgraph-v2` (import path stays `gpgraph`).
- `gpgraph.neighbor_functions` renamed to `gpgraph.neighbors`.
- `get_neighbors` returns a NumPy `(E, 2) int64` array, not a list of tuples.
- `__repr__` on `GenotypePhenotypeGraph` no longer opens a matplotlib figure.

### Removed

- `add_gpm` method (clean break; use `from_gpm` classmethod).
- `_nx_wrapper` introspection shim in `pyplot/primitives.py`.
- v1 notebook examples (`examples/*.ipynb`); the Streamlit showcase under
  `examples/streamlit/` replaces them.
- Support for Python 3.10 and older (networkx 3.6 requires 3.11+).
