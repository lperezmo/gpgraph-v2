---
title: "Fixation models"
description: "The four built-in fixation models and the math behind them. When to pick each, and how to plug in a custom callable."
---

# Fixation models

A fixation model takes two fitness values, `f_i` (source) and `f_j` (target), and returns the probability that an organism with fitness `f_j` displaces one with fitness `f_i` in some population model. `gpgraph-v2` ships four canonical models and accepts user callables.

All four built-in models accept either scalar or matching array inputs. The array path is what `add_model` uses to populate the entire edge set in a single numpy pass.

## `strong_selection_weak_mutation`

Gillespie (1984). Probability of fixation in the strong-selection weak-mutation limit:

$$
\pi_{i \to j} \;=\; 1 - \exp(-s_{ij}), \qquad
s_{ij} \;=\; \frac{f_j - f_i}{f_i}
$$

For $f_j \le f_i$, returns 0 (no advantageous fixation).

```python
from gpgraph.fixation import strong_selection_weak_mutation

strong_selection_weak_mutation(1.0, 1.1)  # 0.0952
```

The string alias is `"sswm"`:

```python
G.add_model(column="phenotypes", model="sswm")
```

This is the default choice for most analyses. It is parameter-free and has a clean interpretation: each edge prob is the fixation probability of the target mutation given the source.

## `ratio`

$$
\operatorname{ratio}(f_i, f_j) \;=\; \frac{f_j}{f_i}
$$

Not a probability, just the relative fitness. Useful when you want edge weights that grow without bound rather than saturate.

```python
from gpgraph.fixation import ratio

ratio(1.0, 2.0)  # 2.0
```

The string alias is `"ratio"`.

!!! warning "Not a probability"

    `ratio` is not in `[0, 1]` for `f_j > f_i`. Treat the resulting `prob` edge attribute as a relative-flow weight, not a probability mass.

## `moran`

Sella and Hirsch (2005). Moran-process fixation probability for population size $N$:

$$
\pi_{i \to j} \;=\; \frac{1 - f_i / f_j}{1 - (f_i / f_j)^{N}}
$$

with overflow-protected branches preserving the scalar v1 behavior. For $f_i = f_j$, returns $1/N$ via a two-point average of slightly perturbed evaluations (matches v1).

```python
from gpgraph.fixation import moran

moran(1.0, 1.01, population_size=100)  # ~0.0179
```

The string alias is `"moran"`. Requires `population_size=N` as a keyword:

```python
G.add_model(column="phenotypes", model="moran", population_size=100)
```

For `N == 1`, returns 1.0 by convention (matches v1).

## `mcclandish`

McCandlish (2011). An exponentiated form of the Moran probability:

$$
\pi_{i \to j} \;=\; \frac{1 - \exp\!\bigl(-2(f_j - f_i)\bigr)}{1 - \exp\!\bigl(-2 N (f_j - f_i)\bigr)}
$$

with the same overflow-protected branches and $f_i = f_j$ handling as `moran`.

```python
from gpgraph.fixation import mcclandish

mcclandish(1.0, 1.01, population_size=100)  # ~0.0179
```

The string alias is `"mcclandish"`.

!!! note "Spelling"

    The function and registry key are spelled `mcclandish` (one `d`), matching the v1 spelling.

## Custom callables

`add_model` accepts any callable as `model`:

```python
def my_model(fi, fj, **_):
    return (fj > fi).astype(float)   # 1.0 for advantageous, 0.0 otherwise

G.add_model(column="phenotypes", model=my_model)
```

The function should either:

- **Accept arrays.** Receives two equal-length NumPy arrays of float64 source and target fitnesses; returns an array of the same shape. The array path is one numpy call per `add_model`. This is the fast path.
- **Accept scalars.** Receives two floats; returns a float. `add_model` falls back to a Python `for` loop. Slower but correct.

If your callable raises `TypeError` on arrays, `add_model` automatically detects this and switches to the scalar loop.

## Choosing a model

| When you want... | Use |
|---|---|
| The default, parameter-free trajectory analysis | `sswm` |
| Edge weights proportional to relative fitness (not normalized) | `ratio` |
| Population-size-dependent fixation, classic Moran formulation | `moran` |
| Population-size-dependent fixation, exponentiated form | `mcclandish` |
| Custom logic (e.g., enforce monotonicity) | a callable |

`sswm` is what `gpgraph-v2`'s test suite and the streamlit demo use unless they explicitly call out a different model.

## Numeric stability

All four built-in models use log-space decomposition to avoid overflow for large `population_size` or large fitness ratios. The boundary cases (`f_j == f_i`, `population_size == 1`, very large negative `s_ij`) are handled with the exact same branching as v1 so numerics agree at the seams.

If you see `NaN` or `inf` in `G.edges[(i, j)]["prob"]`, check that:

1. Your fitness column has no zeros for `moran` or `mcclandish`. Both require strictly positive fitnesses.
2. Your fitnesses are not negative. SSWM, moran, and mcclandish all assume strictly positive fitness.
3. `population_size` is `>= 1`.
