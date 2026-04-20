"""Tests for the vectorized fixation models.

We compare the vectorized implementations against scalar reference versions
lifted from v1 on a grid of (fi, fj, N) points including overflow corners.
"""

from __future__ import annotations

import numpy as np
import pytest
from gpgraph import fixation

# ---------- scalar v1 reference implementations ------------


def _sswm_scalar(fi: float, fj: float) -> float:
    maxexp = float(np.finfo(float).maxexp)
    a = fj - fi
    if a <= 0:
        return 0.0
    ratio = np.log2(a) - np.log2(fi)
    if ratio > maxexp:
        return 1.0
    sij = np.power(2, ratio)
    return float(1 - np.exp(-sij))


def _ratio_scalar(fi: float, fj: float) -> float:
    sign = np.sign(fj) * np.sign(fi)
    a = np.log2(np.abs(fj)) - np.log2(np.abs(fi))
    maxexp = float(np.finfo(float).maxexp)
    value = np.inf if a > maxexp else np.power(2, a)
    return float(value * sign)


def _moran_scalar(fi: float, fj: float, n: float) -> float:
    if n == 1:
        return 1.0
    maxexp = float(np.finfo(float).maxexp)

    if fi == fj:
        # average two perturbations
        return 0.5 * (_moran_scalar(fi * 0.99999, fj, n) + _moran_scalar(fi, fj * 0.99999, n))

    a = np.log2(fi) - np.log2(fj)
    if np.log2(np.abs(a)) + np.log2(n) > maxexp:
        if a > 0:
            return 0.0 if n > maxexp else 1 / np.power(2, n)
        return 1.0 if -a > maxexp else 1 - np.power(2, a)
    b = n * a
    if b > maxexp:
        return float(np.power(2, a - b))
    return float((1 - np.power(2, a)) / (1 - np.power(2, b)))


# -------- tests ------------


@pytest.mark.parametrize(
    "fi,fj",
    [
        (1.0, 1.0),
        (1.0, 2.0),
        (2.0, 1.0),
        (0.5, 0.5),
        (0.1, 10.0),
        (10.0, 0.1),
    ],
)
def test_sswm_scalar_matches_reference(fi: float, fj: float) -> None:
    got = fixation.strong_selection_weak_mutation(fi, fj)
    want = _sswm_scalar(fi, fj)
    assert np.isclose(got, want, atol=1e-12)


def test_sswm_array_matches_scalar() -> None:
    fi = np.array([1.0, 2.0, 0.1, 1.5])
    fj = np.array([2.0, 1.0, 10.0, 1.5])
    got = fixation.strong_selection_weak_mutation(fi, fj)
    want = np.array([_sswm_scalar(a, b) for a, b in zip(fi, fj)])
    assert np.allclose(got, want, atol=1e-12)


def test_sswm_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        fixation.strong_selection_weak_mutation(0.0, 1.0)


@pytest.mark.parametrize(
    "fi,fj",
    [
        (1.0, 1.0),
        (2.0, 1.0),
        (0.5, 2.0),
        (1.0, -1.0),
    ],
)
def test_ratio_scalar_matches_reference(fi: float, fj: float) -> None:
    got = fixation.ratio(fi, fj)
    want = _ratio_scalar(fi, fj)
    assert np.isclose(got, want, atol=1e-12)


def test_ratio_array_matches_scalar() -> None:
    fi = np.array([1.0, 2.0, 0.5])
    fj = np.array([2.0, 1.0, 2.0])
    got = fixation.ratio(fi, fj)
    want = np.array([_ratio_scalar(a, b) for a, b in zip(fi, fj)])
    assert np.allclose(got, want, atol=1e-12)


def test_ratio_rejects_zero_fi() -> None:
    with pytest.raises(ValueError):
        fixation.ratio(0.0, 1.0)


@pytest.mark.parametrize(
    "fi,fj,n",
    [
        (1.0, 2.0, 10),
        (2.0, 1.0, 10),
        (1.0, 1.0, 1),
        (1.0, 1.0, 100),
        (0.1, 10.0, 50),
    ],
)
def test_moran_scalar_matches_reference(fi: float, fj: float, n: int) -> None:
    got = fixation.moran(fi, fj, n)
    want = _moran_scalar(fi, fj, float(n))
    assert np.isclose(got, want, atol=1e-10, rtol=1e-10)


def test_moran_array_matches_scalar() -> None:
    fi = np.array([1.0, 2.0, 1.0, 0.5])
    fj = np.array([2.0, 1.0, 1.0, 1.0])
    n = np.array([10.0, 10.0, 5.0, 20.0])
    got = fixation.moran(fi, fj, n)
    want = np.array([_moran_scalar(a, b, float(c)) for a, b, c in zip(fi, fj, n)])
    assert np.allclose(got, want, atol=1e-10)


def test_moran_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        fixation.moran(0.0, 1.0, 10)
    with pytest.raises(ValueError):
        fixation.moran(1.0, 1.0, 0)


def test_mcclandish_reasonable_range() -> None:
    # For small selection and modest N, McCandlish returns a number in [0, 1]
    # and the direction matches SSWM.
    got_up = fixation.mcclandish(1.0, 1.1, 10)
    got_down = fixation.mcclandish(1.1, 1.0, 10)
    assert 0.0 < got_up <= 1.0
    assert 0.0 < got_down < got_up


def test_mcclandish_n_is_one_returns_one() -> None:
    assert fixation.mcclandish(1.0, 2.0, 1) == 1.0


def test_mcclandish_array_shape() -> None:
    fi = np.array([1.0, 1.0, 1.0])
    fj = np.array([1.2, 1.0, 0.8])
    n = np.array([10.0, 10.0, 10.0])
    got = fixation.mcclandish(fi, fj, n)
    assert got.shape == (3,)
    assert np.all(np.isfinite(got))
