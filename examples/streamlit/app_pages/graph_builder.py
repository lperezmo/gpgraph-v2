"""Interactive graph builder."""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import streamlit as st
from gpgraph import GenotypePhenotypeGraph
from gpgraph.pyplot import draw_gpgraph
from utils import make_fuji_gpm

st.title("Graph builder")

col = st.columns(3)
length = col[0].slider("Sequence length L", min_value=2, max_value=8, value=4)
alphabet = col[1].selectbox("Alphabet", ["AT (binary)", "ATCG (quaternary)"])
neighbor_function = col[2].selectbox("Neighbor function", ["hamming", "codon"])
cutoff = st.slider("Cutoff (max distance between neighbors)", min_value=1, max_value=3, value=1)

alph = ("A", "T") if alphabet.startswith("AT ") else ("A", "T", "C", "G")

gpm = make_fuji_gpm(length=length, alphabet=alph, seed=0)

t0 = time.perf_counter()
G = GenotypePhenotypeGraph.from_gpm(gpm, neighbor_function=neighbor_function, cutoff=cutoff)
elapsed_ms = (time.perf_counter() - t0) * 1e3

col = st.columns(3)
col[0].metric("Nodes", G.number_of_nodes())
col[1].metric("Directed edges", G.number_of_edges())
col[2].metric("Build time", f"{elapsed_ms:.1f} ms")

fig, ax = plt.subplots(figsize=(7, 7))
draw_gpgraph(G, ax=ax, node_size=180)
ax.set_axis_off()
st.pyplot(fig)
