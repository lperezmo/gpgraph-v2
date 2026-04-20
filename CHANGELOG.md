# CHANGELOG


## v1.0.0 (2026-04-20)

### Features

- Scaffold gpgraph-v2 with uv, maturin, and PyO3
  ([`2f9d780`](https://github.com/lperezmo/gpgraph-v2/commit/2f9d78021d2c344eccce550c8843aeb38f9c725e))

Clean-break rewrite of harmslab/gpgraph. NetworkX-backed, Rust-accelerated directed graphs over
  gpmap-v2 genotype-phenotype maps.

Stack: uv + maturin + PyO3 + rayon. Python 3.11+. Reads the locked gpmap-v2 SCHEMA.md contract
  (binary_packed, n_mutations, phenotypes). Fixation models (sswm, ratio, moran, mcclandish) stay in
  vectorized numpy; neighbor detection and codon distance live in Rust.

Release automation via python-semantic-release on Conventional Commits. PyPI via OIDC trusted
  publisher. build_command is empty and allow_zero_version=false so the first release jumps straight
  to 1.0.0 rather than climbing from 0.0.1, and semantic-release does not try to invoke maturin
  inside a Docker image without Rust installed.

See CHANGELOG.md for the full list of ported and fixed behaviors.
