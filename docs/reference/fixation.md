---
title: "Fixation models reference"
description: "Reference for strong_selection_weak_mutation, ratio, moran, mcclandish, and the MODEL_REGISTRY."
---

# `gpgraph.fixation`

Vectorized fixation models. Each accepts scalars (returns a float) or matching arrays (returns an array of the same shape). The math behind each model is covered in [Concepts: Fixation models](../concepts/fixation-models.md).

## `strong_selection_weak_mutation`

```python
def strong_selection_weak_mutation(
    fitness_i: float | NDArray[np.float64],
    fitness_j: float | NDArray[np.float64],
) -> float | NDArray[np.float64]
```

Gillespie (1984) strong-selection weak-mutation fixation probability:

```
pi = 1 - exp(-s_ij)
s_ij = (f_j - f_i) / f_i
```

For `f_j <= f_i`, returns 0.

### Raises

`ValueError`
:   If any input fitness is `<= 0`.

## `ratio`

```python
def ratio(
    fitness_i: float | NDArray[np.float64],
    fitness_j: float | NDArray[np.float64],
) -> float | NDArray[np.float64]
```

Simple ratio `f_j / f_i` with a log2-domain overflow guard. Not a probability, not guaranteed to be in `[0, 1]`.

### Raises

`ValueError`
:   If any `fitness_i == 0`.

## `moran`

```python
def moran(
    fitness_i: float | NDArray[np.float64],
    fitness_j: float | NDArray[np.float64],
    population_size: float | NDArray[np.float64],
) -> float | NDArray[np.float64]
```

Sella and Hirsch (2005) Moran-process fixation probability.

### Parameters

`fitness_i`, `fitness_j` (`float | NDArray[float64]`)
:   Source and target fitnesses. Must be > 0.

`population_size` (`float | NDArray[float64]`)
:   Effective population size, must be >= 1.

### Special cases

- `population_size == 1`: returns 1.0 by convention.
- `fitness_i == fitness_j`: returns the limit value computed by averaging two slightly perturbed evaluations.

### Raises

`ValueError`
:   If any input fitness is `<= 0` or any `population_size < 1`.

## `mcclandish`

```python
def mcclandish(
    fitness_i: float | NDArray[np.float64],
    fitness_j: float | NDArray[np.float64],
    population_size: float | NDArray[np.float64],
) -> float | NDArray[np.float64]
```

McCandlish (2011) fixation probability:

```
pi = (1 - exp(-2*(fj - fi))) / (1 - exp(-2*N*(fj - fi)))
```

Same input requirements and special cases as `moran`.

## `MODEL_REGISTRY`

```python
MODEL_REGISTRY: dict[str, Callable] = {
    "sswm": strong_selection_weak_mutation,
    "ratio": ratio,
    "moran": moran,
    "mcclandish": mcclandish,
}
```

Lookup table used by `GenotypePhenotypeGraph.add_model` when `model` is a string. Pass any of these keys, or pass a callable directly to bypass the registry.
