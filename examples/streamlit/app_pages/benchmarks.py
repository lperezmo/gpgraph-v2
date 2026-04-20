"""Live benchmark: Rust bit-flip vs pairwise packed hamming."""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from gpgraph import _rust

st.title("Benchmarks")

st.markdown(
    """
Rust bit-flip vs rayon-parallel pairwise hamming on random biallelic
packed data. The bit-flip kernel specializes on cutoff 1 and 2 and scales
as `O(N * C(L, cutoff))`; the pairwise kernel is `O(N^2 * L)`. At moderate
N the bit-flip path wins decisively.
"""
)

length = st.slider("L", 4, 12, 8)
cutoff = st.slider("cutoff", 1, 2, 1)
max_n_pow = st.slider("Max log2(N)", 6, 14, 12)

ns = [2**k for k in range(6, max_n_pow + 1)]

rows: list[dict[str, float]] = []
progress = st.progress(0.0)
for i, n in enumerate(ns):
    rng = np.random.default_rng(0)
    bp = rng.integers(0, 2, size=(n, length), dtype=np.uint8)
    # Deduplicate for the bit-flip kernel (it requires unique rows indirectly;
    # pairwise handles duplicates fine).
    _, idx = np.unique(bp, axis=0, return_index=True)
    bp_u = bp[np.sort(idx)]

    t0 = time.perf_counter()
    _rust.hamming_bitflip_neighbors(bp_u, cutoff)
    t_flip = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    _rust.hamming_neighbors_packed(bp, cutoff)
    t_pair = (time.perf_counter() - t0) * 1e3

    rows.append({"N": n, "bit-flip (ms)": t_flip, "pairwise (ms)": t_pair})
    progress.progress((i + 1) / len(ns))

df = pd.DataFrame(rows)
st.dataframe(df, width="stretch")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["N"], df["bit-flip (ms)"], marker="o", label="bit-flip")
ax.plot(df["N"], df["pairwise (ms)"], marker="s", label="pairwise")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("N")
ax.set_ylabel("time (ms)")
ax.set_title(f"hamming neighbors, L={length}, cutoff={cutoff}")
ax.legend()
st.pyplot(fig)
