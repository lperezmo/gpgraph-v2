"""Fixation model explorer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from gpgraph import GenotypePhenotypeGraph
from gpgraph.pyplot import draw_gpgraph
from utils import make_fuji_gpm

st.title("Fixation model explorer")

col = st.columns(3)
length = col[0].slider("L", 2, 6, 4)
model_name = col[1].selectbox("Model", ["sswm", "ratio", "moran", "mcclandish"])
population_size = col[2].slider("Population size N", 1, 1000, 100, 1)

gpm = make_fuji_gpm(length=length, seed=0)
G = GenotypePhenotypeGraph.from_gpm(gpm)

model_kwargs: dict[str, float] = {}
if model_name in {"moran", "mcclandish"}:
    model_kwargs["population_size"] = float(population_size)

G.add_model(column="phenotypes", model=model_name, **model_kwargs)
probs = np.array([G.edges[e]["prob"] for e in G.edges])

col = st.columns(3)
col[0].metric("mean prob", f"{probs.mean():.3f}")
col[1].metric("median prob", f"{float(np.median(probs)):.3f}")
col[2].metric("edges with prob > 0", int((probs > 0).sum()))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].hist(probs, bins=30, color="#4C72B0")
axes[0].set_xlabel("edge 'prob'")
axes[0].set_ylabel("count")
axes[0].set_title(f"{model_name} edge distribution")

draw_gpgraph(G, ax=axes[1], node_size=180)
axes[1].set_axis_off()

st.pyplot(fig)
