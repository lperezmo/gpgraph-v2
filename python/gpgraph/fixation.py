"""Vectorized fixation models for genotype-phenotype graphs.

Each model accepts either two scalar fitness values (v1-style) or two
equal-length numpy arrays, returns the corresponding fixation probability
(or ratio) per entry. The array path is used by :func:`add_model` to
populate all edge probabilities in one numpy pass.

Overflow-protection branches from the v1 implementation are preserved via
``np.where`` so numeric agreement holds at the boundaries.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_MAXEXP = float(np.finfo(float).maxexp)


def _as_float_array(x: object) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def strong_selection_weak_mutation(
    fitness_i: float | NDArray[np.float64],
    fitness_j: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Gillespie (1984) strong-selection weak-mutation fixation probability.

    ``pi_{i -> j} = 1 - exp(-s_ij)`` where ``s_ij = (f_j - f_i) / f_i``.

    Parameters
    ----------
    fitness_i, fitness_j:
        Source and target fitnesses. Must be > 0. Scalars or matching arrays.

    Returns
    -------
    Fixation probability (scalar or array of same shape).
    """
    fi = _as_float_array(fitness_i)
    fj = _as_float_array(fitness_j)
    if np.any(fi <= 0) or np.any(fj <= 0):
        raise ValueError("fitness values must be > 0")

    a = fj - fi
    # a <= 0 -> no fixation probability advantage; use 0.0.
    # a > 0  -> compute via exp with overflow protection using log2 decomposition.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_log = np.log2(np.where(a > 0, a, 1.0)) - np.log2(fi)
    sij = np.where(
        a <= 0,
        0.0,
        np.where(ratio_log > _MAXEXP, np.inf, np.power(2.0, ratio_log)),
    )
    out = np.where(a <= 0, 0.0, 1.0 - np.exp(-sij))
    if fi.ndim == 0 and fj.ndim == 0:
        return float(out)
    return out


def ratio(
    fitness_i: float | NDArray[np.float64],
    fitness_j: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Simple ratio ``f_j / f_i`` with log2-domain overflow guard.

    Not a true fixation probability. Not guaranteed to be in [0, 1].
    """
    fi = _as_float_array(fitness_i)
    fj = _as_float_array(fitness_j)
    if np.any(fi == 0):
        raise ValueError("fitness_i must not be zero")

    sign = np.sign(fj) * np.sign(fi)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log2(np.abs(fj)) - np.log2(np.abs(fi))
    value = np.where(a > _MAXEXP, np.inf, np.power(2.0, a))
    out: NDArray[np.float64] = value * sign
    if fi.ndim == 0 and fj.ndim == 0:
        return float(out)
    return out


def _moran_safe(
    fi: NDArray[np.float64], fj: NDArray[np.float64], n: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Moran fixation probability evaluated element-wise with overflow guards.

    Implements the branches in v1's scalar ``moran`` in vector form. Expects
    fi, fj > 0, n >= 1, fi != fj. The fi == fj case is handled by averaging
    two slightly-perturbed evaluations at the call site.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.log2(fi) - np.log2(fj)
        log2_abs_a = np.log2(np.abs(a))
        # Handle the overflow guard |a|*N > maxexp (the "huge" branch in v1).
        huge = log2_abs_a + np.log2(n) > _MAXEXP
        # For the huge branch we fall back to the asymptotic forms from v1.
        # a > 0: 1 / 2^N (or 0 if N overflows).
        # a < 0: 1 - 2^a (or 1 if -a overflows).
        asymptotic_pos = np.where(n > _MAXEXP, 0.0, 1.0 / np.power(2.0, np.minimum(n, _MAXEXP)))
        asymptotic_neg = np.where(
            -a > _MAXEXP, 1.0, 1.0 - np.power(2.0, np.clip(a, -_MAXEXP, _MAXEXP))
        )
        huge_out = np.where(a > 0, asymptotic_pos, asymptotic_neg)

        # Regular branch: check whether b = a*N overflows. If it does, return
        # 2^(a-b) (per v1). Otherwise compute (1 - 2^a) / (1 - 2^b) directly.
        b = a * n
        b_overflow = b > _MAXEXP
        power_a = np.power(2.0, np.clip(a, -_MAXEXP, _MAXEXP))
        power_b = np.power(2.0, np.clip(b, -_MAXEXP, _MAXEXP))
        # 1 - power_b can be 0 when a == 0 (fi == fj); those points are masked
        # out at the call site, so a NaN here is fine.
        safe_denom = np.where(power_b == 1.0, 1.0, 1.0 - power_b)
        regular = np.where(b_overflow, np.power(2.0, a - b), (1.0 - power_a) / safe_denom)

    return np.where(huge, huge_out, regular)


def moran(
    fitness_i: float | NDArray[np.float64],
    fitness_j: float | NDArray[np.float64],
    population_size: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Sella and Hirsch (2005) Moran-process fixation probability.

    Parameters
    ----------
    fitness_i, fitness_j:
        Source and target fitnesses (> 0).
    population_size:
        Effective population size (>= 1).

    Notes
    -----
    For population_size == 1 this returns 1.0 by convention, matching v1.
    When ``fitness_i == fitness_j``, v1 averaged evaluations at two slightly
    perturbed operating points to avoid the removable 0/0 singularity. We
    reproduce that behavior element-wise.
    """
    fi = _as_float_array(fitness_i)
    fj = _as_float_array(fitness_j)
    n = _as_float_array(population_size)
    if np.any(fi <= 0) or np.any(fj <= 0):
        raise ValueError("fitness values must be > 0")
    if np.any(n < 1):
        raise ValueError("population_size must be >= 1")

    fi_b = np.broadcast_to(fi, np.broadcast_shapes(fi.shape, fj.shape, n.shape)).astype(
        np.float64, copy=True
    )
    fj_b = np.broadcast_to(fj, fi_b.shape).astype(np.float64, copy=True)
    n_b = np.broadcast_to(n, fi_b.shape).astype(np.float64, copy=True)

    # N == 1: return 1.0 per v1 convention.
    n_is_one = n_b == 1

    # fi == fj: average two perturbed evaluations (matches v1 branching).
    eq_mask = fi_b == fj_b
    if np.any(eq_mask):
        eval_a = _moran_safe(fi_b * 0.99999, fj_b, n_b)
        eval_b = _moran_safe(fi_b, fj_b * 0.99999, n_b)
        averaged = 0.5 * (eval_a + eval_b)
    else:
        averaged = np.zeros_like(fi_b)

    direct = _moran_safe(fi_b, fj_b, n_b)
    out = np.where(eq_mask, averaged, direct)
    out = np.where(n_is_one, 1.0, out)

    if fi.ndim == 0 and fj.ndim == 0 and n.ndim == 0:
        return float(out)
    return out


def _mcclandish_safe(
    fi: NDArray[np.float64], fj: NDArray[np.float64], n: NDArray[np.float64]
) -> NDArray[np.float64]:
    """McCandlish (2011) fixation probability, element-wise with overflow guard."""
    a = fj - fi
    power_coeff = -2.0 * np.log2(np.e)
    l2_power_coeff = np.log2(2.0 * np.log2(np.e))

    with np.errstate(divide="ignore", invalid="ignore"):
        log2_abs_a = np.log2(np.abs(a))

        can_do_2a = log2_abs_a + l2_power_coeff <= _MAXEXP
        can_do_exp_2a = can_do_2a & ((a * power_coeff) <= _MAXEXP)
        can_do_2aN = log2_abs_a + np.log2(n) + l2_power_coeff <= _MAXEXP
        can_do_exp_2aN = can_do_2aN & ((a * n * power_coeff) <= _MAXEXP)

        regular_mask = can_do_exp_2a & can_do_exp_2aN

        # Regular path.
        neg2a = np.where(can_do_2a, a * power_coeff, 0.0)
        exp_neg2a = np.where(can_do_exp_2a, np.power(2.0, np.clip(neg2a, -_MAXEXP, _MAXEXP)), 0.0)
        neg2aN = np.where(can_do_2aN, a * n * power_coeff, 0.0)
        exp_neg2aN = np.where(
            can_do_exp_2aN, np.power(2.0, np.clip(neg2aN, -_MAXEXP, _MAXEXP)), 0.0
        )

        regular = np.where(
            regular_mask,
            (1.0 - exp_neg2a) / np.where(exp_neg2aN == 1.0, 1.0, 1.0 - exp_neg2aN),
            0.0,
        )

        # Non-regular asymptotic branch from v1.
        # a > 0 and exp(-2a) overflowed -> 1.0
        # a > 0 and exp(-2a) finite     -> 1 - exp(-2a)
        # a < 0 -> 0.0 in all overflow cases
        pos_branch = np.where(can_do_exp_2a, 1.0 - exp_neg2a, 1.0)
        asymptotic = np.where(a > 0, pos_branch, 0.0)

    return np.where(regular_mask, regular, asymptotic)


def mcclandish(
    fitness_i: float | NDArray[np.float64],
    fitness_j: float | NDArray[np.float64],
    population_size: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """McCandlish (2011) fixation probability.

    ``pi = (1 - exp(-2*(fj - fi))) / (1 - exp(-2*N*(fj - fi)))``
    with overflow-protected branches that preserve the scalar v1 behavior.
    """
    fi = _as_float_array(fitness_i)
    fj = _as_float_array(fitness_j)
    n = _as_float_array(population_size)
    if np.any(fi <= 0) or np.any(fj <= 0):
        raise ValueError("fitness values must be > 0")
    if np.any(n < 1):
        raise ValueError("population_size must be >= 1")

    shape = np.broadcast_shapes(fi.shape, fj.shape, n.shape)
    fi_b = np.broadcast_to(fi, shape).astype(np.float64, copy=True)
    fj_b = np.broadcast_to(fj, shape).astype(np.float64, copy=True)
    n_b = np.broadcast_to(n, shape).astype(np.float64, copy=True)

    n_is_one = n_b == 1
    eq_mask = fi_b == fj_b
    if np.any(eq_mask):
        eval_a = _mcclandish_safe(fi_b * 0.99999, fj_b, n_b)
        eval_b = _mcclandish_safe(fi_b, fj_b * 0.99999, n_b)
        averaged = 0.5 * (eval_a + eval_b)
    else:
        averaged = np.zeros_like(fi_b)

    direct = _mcclandish_safe(fi_b, fj_b, n_b)
    out = np.where(eq_mask, averaged, direct)
    out = np.where(n_is_one, 1.0, out)

    if fi.ndim == 0 and fj.ndim == 0 and n.ndim == 0:
        return float(out)
    return out


# Lookup registry used by :meth:`GenotypePhenotypeGraph.add_model`.
MODEL_REGISTRY = {
    "sswm": strong_selection_weak_mutation,
    "ratio": ratio,
    "moran": moran,
    "mcclandish": mcclandish,
}
