import numpy as np
import pytest

from genomorph.measures import (
    centroid,
    jordan_decomposition,
    normalize_measure,
    signed_mass_delta,
    support_width,
    total_variation,
)


def test_jordan_reconstructs_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    pos, neg = jordan_decomposition(x)
    assert np.all(pos >= 0) and np.all(neg >= 0)
    assert np.allclose(pos - neg, x)
    assert np.allclose(pos * neg, 0)  # disjoint support


def test_normalize_sums_to_one():
    w = np.array([1.0, 2.0, 1.0])
    n, total = normalize_measure(w)
    assert n.sum() == pytest.approx(1.0)
    assert total == pytest.approx(4.0)


def test_normalize_zero_mass_is_uniform():
    n, total = normalize_measure(np.zeros(5))
    assert np.allclose(n, 0.2)
    assert total == 0.0


def test_normalize_rejects_negative():
    with pytest.raises(ValueError):
        normalize_measure(np.array([-1.0, 2.0]))


def test_signed_mass_delta_sign():
    ref = np.array([1.0, 1.0, 1.0])
    assert signed_mass_delta(ref, ref * 2) == pytest.approx(3.0)
    assert signed_mass_delta(ref, ref * 0.5) == pytest.approx(-1.5)


def test_total_variation_nonneg():
    ref = np.array([1.0, 2.0, 3.0])
    alt = np.array([3.0, 2.0, 1.0])
    assert total_variation(ref, alt) == pytest.approx(4.0)
    assert total_variation(ref, ref) == 0.0


def test_centroid_of_symmetric_peak():
    coords = np.arange(11, dtype=float)
    w = np.exp(-0.5 * ((coords - 5) / 1.5) ** 2)
    assert centroid(coords, w) == pytest.approx(5.0, abs=1e-6)


def test_support_width_increases_with_spread():
    coords = np.arange(101, dtype=float)
    narrow = np.exp(-0.5 * ((coords - 50) / 2) ** 2)
    wide = np.exp(-0.5 * ((coords - 50) / 8) ** 2)
    assert support_width(coords, wide) > support_width(coords, narrow)
