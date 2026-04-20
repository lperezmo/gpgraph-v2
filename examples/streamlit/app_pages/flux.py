"""Forward-path flux visualization."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import streamlit as st
from gpgraph import GenotypePhenotypeGraph
from gpgraph.paths import forward_paths_prob, paths_prob_to_edges_flux
from gpgraph.pyplot import draw_gpgraph
from utils import make_fuji_gpm

st.title("Forward-path flux")

length = st.slider("L", 2, 6, 4)

gpm = make_fuji_gpm(length=length, seed=1)
G = GenotypePhenotypeGraph.from_gpm(gpm)
G.add_model(column="phenotypes", model="sswm")

genotypes = list(gpm.genotypes)
col = st.columns(2)
source = col[0].selectbox("Source", genotypes, index=0)
target = col[1].selectbox("Target", genotypes, index=len(genotypes) - 1)

try:
    paths = forward_paths_prob(G, source, target)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

flux = paths_prob_to_edges_flux(paths)

col = st.columns(3)
col[0].metric("shortest paths", len(paths))
col[1].metric("edges with flux", len(flux))
col[2].metric("sum of path probs", f"{sum(paths.values()):.3f}")

fig, ax = plt.subplots(figsize=(8, 8))
draw_gpgraph(G, ax=ax, paths=paths, node_size=180)
ax.set_axis_off()
st.pyplot(fig)
