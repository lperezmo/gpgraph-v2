"""Landing page."""

from __future__ import annotations

import streamlit as st

st.title("gpgraph-v2")

st.markdown(
    """
A NetworkX-backed, Rust-accelerated toolkit for turning
[gpmap-v2](https://github.com/lperezmo/gpmap-v2) genotype-phenotype maps into
directed graphs you can reason about.

### Three-line quickstart

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
G.add_model(column="phenotypes", model="sswm")
```

### What the pages show

- **Graph builder** - choose a simulator, pick an alphabet and neighbor
  distance, and see the resulting graph with timing.
- **Fixation explorer** - run a fixation model and visualize the edge-probability
  distribution.
- **Path flux** - source / target picker over the Hamming hypercube with
  flux overlay.
- **Benchmarks** - how the Rust bit-flip kernel compares against the
  rayon-parallel pairwise kernel as N grows.
"""
)
