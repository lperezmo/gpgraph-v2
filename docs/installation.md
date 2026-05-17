---
title: "Install gpgraph-v2"
description: "Install from PyPI with uv or pip. Matplotlib is an optional extra. Build from source with maturin if you need an unreleased commit."
---

# Install gpgraph-v2

gpgraph-v2 ships prebuilt wheels for Linux (x86_64, aarch64), macOS (x86_64, aarch64), and Windows (x64). Python 3.11 or newer is required.

## Stable release

=== "uv"

    ```bash
    uv add gpgraph-v2
    ```

=== "pip"

    ```bash
    pip install gpgraph-v2
    ```

The package installs the import path `gpgraph`, regardless of the distribution name being `gpgraph-v2`.

```python
import gpgraph
print(gpgraph.__version__)
```

## Optional extras

Matplotlib is split out so the core install stays headless.

=== "uv"

    ```bash
    uv add "gpgraph-v2[plot]"
    ```

=== "pip"

    ```bash
    pip install "gpgraph-v2[plot]"
    ```

| Extra | What it adds |
|---|---|
| `gpgraph-v2[plot]` | matplotlib, used by `gpgraph.pyplot` |
| `gpgraph-v2[dev]` | `pytest`, `pytest-benchmark`, `mypy`, `ruff`, `maturin` |

Importing `gpgraph.pyplot` without matplotlib installed raises a clear `ImportError` directing you to install the extra.

## Build from source

You need a Rust toolchain (stable, `>= 1.70`) and `maturin`. The recommended workflow uses `uv`:

```bash
git clone https://github.com/lperezmo/gpgraph-v2
cd gpgraph-v2
uv sync
uv run maturin develop --release
uv run pytest
```

After editing Rust under `crates/`:

```bash
uv run maturin develop --release && uv run pytest
```

## Consuming from another local project

For co-development with sister packages like [`gpmap-v2`](https://github.com/lperezmo/gpmap-v2) and [`epistasis-v2`](https://github.com/lperezmo/epistasis-v2), point at a local editable copy in your consumer's `pyproject.toml`:

```toml
[tool.uv.sources]
gpgraph-v2 = { path = "/absolute/path/to/gpgraph-v2", editable = true }
gpmap-v2 = { path = "/absolute/path/to/gpmap-v2", editable = true }

[project]
dependencies = ["gpgraph-v2", "gpmap-v2"]
```

Then `uv sync` in the consumer. Imports remain `from gpgraph import GenotypePhenotypeGraph`.

## Verify the install

```python
from gpmap import GenotypePhenotypeMap
from gpgraph import GenotypePhenotypeGraph

gpm = GenotypePhenotypeMap(
    wildtype="AA",
    genotypes=["AA", "AT", "TA", "TT"],
    phenotypes=[0.1, 0.2, 0.3, 1.0],
)
G = GenotypePhenotypeGraph.from_gpm(gpm)
print(G.number_of_nodes(), G.number_of_edges())  # 4 8
```

If `G` reports 4 nodes and 8 directed edges, the Rust extension is wired in correctly.
