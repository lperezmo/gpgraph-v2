"""About page."""

from __future__ import annotations

import streamlit as st

st.title("About")

st.markdown(
    """
**gpgraph-v2** is a clean-break rewrite of
[harmslab/gpgraph](https://github.com/harmslab/gpgraph). It shares a
modernization pattern with its sister projects:

- [gpmap-v2](https://github.com/lperezmo/gpmap-v2) - the genotype-phenotype
  map container and simulator toolkit that gpgraph consumes.
- [epistasis-v2](https://github.com/lperezmo/epistasis-v2) - the epistatic
  coefficient fitting library.

All three: `uv` + `maturin` + PyO3 + rayon where it earns its keep,
vectorized numpy everywhere else, type-hinted public API, `mypy --strict`
in CI, `python-semantic-release` with OIDC-trusted PyPI publishing.
"""
)
