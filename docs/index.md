---
title: "gpgraph-v2"
description: "NetworkX-backed, Rust-accelerated graphs over genotype-phenotype maps."
---

# gpgraph-v2

NetworkX-backed, Rust-accelerated graphs over genotype-phenotype maps. Lifts a [`gpmap-v2`](https://github.com/lperezmo/gpmap-v2) `GenotypePhenotypeMap` into a NetworkX `DiGraph`, populates edges under Hamming or codon distance, and overlays fixation-probability edge weights for evolutionary trajectory analysis.

<div class="grid cards" markdown>

-   :material-rocket-launch: **Quickstart**

    Build a graph from a `GenotypePhenotypeMap`, add a fixation model, and pull forward-path probabilities.

    [:octicons-arrow-right-24: Read the quickstart](quickstart.md)

-   :material-package-down: **Installation**

    Install from PyPI. Matplotlib is an optional extra.

    [:octicons-arrow-right-24: Install gpgraph-v2](installation.md)

-   :material-lightbulb-on: **Concepts**

    Graph model, neighbor kernels, and the fixation-model registry.

    [:octicons-arrow-right-24: Learn the model](concepts/graph-model.md)

-   :material-book-open-page-variant: **Reference**

    Per-module API for the graph, neighbors, fixation models, paths, and pyplot.

    [:octicons-arrow-right-24: Browse the reference](reference/graph.md)

</div>

## What it does

- **Lift a map into a graph.** `GenotypePhenotypeGraph.from_gpm(gpm)` returns a NetworkX `DiGraph` with one node per genotype and directed edges between neighbors.
- **Plug in a fixation model.** Pick from `sswm`, `ratio`, `moran`, `mcclandish`, or bring your own callable. `add_model(column="phenotypes", model="sswm")` populates the `prob` edge attribute in one vectorized numpy pass.
- **Enumerate trajectories.** `forward_paths`, `forward_paths_prob`, and the flux helpers give you the shortest paths from WT to target and their probability mass.
- **Plot when needed.** The optional `gpgraph.pyplot` subpackage draws the graph with a Hamming-layer layout and overlays path flux on the edges.

## Why v2

- **Fast.** Neighbor detection runs in Rust with rayon parallelism. Biallelic cutoff-1 and cutoff-2 hit a bit-flip fast path: `O(N * L^cutoff)` instead of `O(N^2 * L)`.
- **Typed.** Full type hints, `mypy --strict` in CI.
- **Modern tooling.** `uv` plus `maturin` plus `pyproject.toml`. Releases via `python-semantic-release`. OIDC-based PyPI publishing.
- **Consumes gpmap-v2.** Speaks the locked schema contract; reads `binary_packed`, `n_mutations`, and `phenotypes`.
- **No Cython, no `setup.py`, no `.c` blobs.**

## Live demo

A multi-page Streamlit tour is published at [gpgraph-v2.streamlit.app](https://gpgraph-v2.streamlit.app) (source under [`examples/streamlit/`](https://github.com/lperezmo/gpgraph-v2/tree/main/examples/streamlit) in the repo).

## Next steps

- Just want to wire it up? [Quickstart](quickstart.md).
- Already familiar with the legacy `harmslab/gpgraph`? [Migration notes](concepts/graph-model.md#migration-from-v1).
- Building a downstream consumer that needs the neighbor matrix directly? [Neighbors concept](concepts/neighbors.md).
- Need forward-path flux for plotting? [Forward paths guide](guides/forward-paths.md).
