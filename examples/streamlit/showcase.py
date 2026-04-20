"""Entry point for the gpgraph-v2 Streamlit showcase."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="gpgraph-v2",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

pages = [
    st.Page("app_pages/intro.py", title="Intro", icon=":material/home:"),
    st.Page("app_pages/graph_builder.py", title="Graph builder", icon=":material/hub:"),
    st.Page(
        "app_pages/fixation_explorer.py",
        title="Fixation explorer",
        icon=":material/biotech:",
    ),
    st.Page("app_pages/flux.py", title="Path flux", icon=":material/route:"),
    st.Page("app_pages/benchmarks.py", title="Benchmarks", icon=":material/speed:"),
    st.Page("app_pages/about.py", title="About", icon=":material/info:"),
]

nav = st.navigation(pages)
nav.run()
