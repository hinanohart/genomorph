import numpy as np
import pytest

from genomorph.ot import wasserstein1d, wasserstein1d_pot


def _dirac(n, i):
    w = np.zeros(n)
    w[i] = 1.0
    return w


def test_w1_between_diracs_equals_distance():
    # On a unit-spaced grid, W1 between point masses is their coordinate gap.
    coords = np.arange(20, dtype=float)
    for a, b in [(2, 5), (0, 19), (7, 8), (10, 3)]:
        d = wasserstein1d(coords, _dirac(20, a), _dirac(20, b))
        assert d == pytest.approx(abs(a - b))


def test_w1_scales_with_bin_size():
    coords = np.arange(10, dtype=float) * 128.0
    d = wasserstein1d(coords, _dirac(10, 1), _dirac(10, 4))
    assert d == pytest.approx(3 * 128.0)


def test_w1_symmetry_and_identity():
    rng = np.random.default_rng(0)
    coords = np.arange(50, dtype=float)
    u = rng.random(50)
    v = rng.random(50)
    assert wasserstein1d(coords, u, u) == pytest.approx(0.0, abs=1e-12)
    assert wasserstein1d(coords, u, v) == pytest.approx(wasserstein1d(coords, v, u))


def test_w1_triangle_inequality():
    rng = np.random.default_rng(1)
    coords = np.arange(40, dtype=float)
    u, v, w = (rng.random(40) for _ in range(3))
    uv = wasserstein1d(coords, u, v)
    vw = wasserstein1d(coords, v, w)
    uw = wasserstein1d(coords, u, w)
    assert uw <= uv + vw + 1e-9


def test_w1_matches_pot():
    rng = np.random.default_rng(2)
    coords = np.arange(64, dtype=float)
    u = rng.random(64)
    v = rng.random(64)
    mine = wasserstein1d(coords, u, v)
    pot = wasserstein1d_pot(coords, coords, u, v)
    assert mine == pytest.approx(pot, rel=1e-6, abs=1e-6)


def test_w1_zero_mass_is_uniform_and_finite():
    coords = np.arange(10, dtype=float)
    d = wasserstein1d(coords, np.zeros(10), _dirac(10, 5))
    assert np.isfinite(d) and d > 0


def test_w1_rejects_unsorted_coords():
    with pytest.raises(ValueError):
        wasserstein1d(np.array([0.0, 2.0, 1.0]), np.ones(3), np.ones(3))


def test_w1_deterministic():
    coords = np.arange(30, dtype=float)
    rng = np.random.default_rng(3)
    u, v = rng.random(30), rng.random(30)
    vals = {wasserstein1d(coords, u, v) for _ in range(50)}
    assert len(vals) == 1
