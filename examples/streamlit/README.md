# gpgraph-v2 Streamlit showcase

A multi-page Streamlit app that tours the gpgraph-v2 API end-to-end. Deployed
at [gpgraph-v2.streamlit.app](https://gpgraph-v2.streamlit.app).

## Pages

1. **Intro** - what gpgraph-v2 is and a three-line quickstart.
2. **Graph builder** - pick a gpmap simulator, L, alphabet, neighbor function
   and cutoff. Shows the resulting graph and the time it took.
3. **Fixation explorer** - choose a fitness column and a model (sswm, ratio,
   moran, mcclandish). Sliders for population size. Histograms the edge
   probability distribution and redraws the graph with edge widths from flux.
4. **Flux** - source/target picker, draws forward-path flux.
5. **Benchmarks** - live bitflip-vs-pairwise timing at several N.

## Run locally

```bash
uv sync
uv run maturin develop --release
uv run streamlit run examples/streamlit/showcase.py
```

## Streamlit Cloud

`requirements.txt` pins everything Streamlit Cloud needs. The Rust crate is
built from sdist via maturin when Cloud installs gpgraph-v2.
