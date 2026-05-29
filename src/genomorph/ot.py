"""1-D Wasserstein-1 distance between measures on genomic coordinates.

For two measures supported on a shared, sorted coordinate grid the
1-Wasserstein distance has the closed form

    W1(u, v) = integral |F_u(t) - F_v(t)| dt
             = sum_i |C_u[i] - C_v[i]| * (x[i+1] - x[i])

where ``C`` are the cumulative masses and ``F`` the (right-continuous) CDFs,
which are piecewise constant between grid points. This is exact and runs in
O(n) once the grid is sorted -- no optimisation, no sampling. We use this for
the common case where REF/ALT live on the same bin grid, and fall back to POT's
sample-based estimator for arbitrary supports (and to cross-check in tests).
"""

from __future__ import annotations

import numpy as np

from .measures import normalize_measure

__all__ = ["wasserstein1d", "wasserstein1d_pot"]


def wasserstein1d(
    coords: np.ndarray,
    u_weights: np.ndarray,
    v_weights: np.ndarray,
    *,
    normalize: bool = True,
) -> float:
    """Exact 1-D W1 between two measures on a shared coordinate grid.

    Parameters
    ----------
    coords:
        Strictly increasing bin coordinates (bp). Shared by both measures.
    u_weights, v_weights:
        Non-negative masses at each coordinate.
    normalize:
        If true (default) both measures are normalised to probability measures
        first, so the distance reflects *shape* only and is comparable across
        modalities of different total mass. If false the raw masses are used
        (the measures must then already carry equal total mass for W1 to be a
        true transport distance).
    """
    x = np.asarray(coords, dtype=np.float64)
    if x.ndim != 1 or x.shape[0] < 1:
        raise ValueError("coords must be a non-empty 1-D array")
    if np.any(np.diff(x) <= 0):
        raise ValueError("coords must be strictly increasing")

    if normalize:
        u = normalize_measure(u_weights)[0]
        v = normalize_measure(v_weights)[0]
    else:
        u = np.asarray(u_weights, dtype=np.float64)
        v = np.asarray(v_weights, dtype=np.float64)

    if x.shape[0] == 1:
        return 0.0

    cu = np.cumsum(u)[:-1]
    cv = np.cumsum(v)[:-1]
    widths = np.diff(x)
    return float(np.sum(np.abs(cu - cv) * widths))


def wasserstein1d_pot(
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    u_weights: np.ndarray,
    v_weights: np.ndarray,
) -> float:
    """POT-backed 1-D W1 for arbitrary (possibly distinct) supports.

    Wraps :func:`ot.wasserstein_1d` (p=1) and normalises both inputs to
    probability measures. Used for cross-validation and for backends that emit
    measures on non-shared supports.
    """
    import ot  # POT (MIT); declared core dependency

    ua = normalize_measure(u_weights)[0]
    va = normalize_measure(v_weights)[0]
    val = ot.wasserstein_1d(
        np.asarray(u_coords, dtype=np.float64),
        np.asarray(v_coords, dtype=np.float64),
        ua,
        va,
        p=1,
    )
    return float(np.asarray(val).item())
