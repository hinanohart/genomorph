"""Turn (possibly signed) coverage profiles into 1-D measures.

The fingerprint separates two things a scalar effect size conflates:

* **shape** -- *where* along the genome the predicted signal sits, captured by
  normalising a profile to a probability measure on bin coordinates;
* **mass** -- *how much* total signal changed, captured by the signed sum of
  the ALT-minus-REF difference.

Signed tracks (log fold-change, contact-difference) are split with a Jordan
decomposition into a non-negative positive part and non-negative negative part,
each of which is a measure in its own right.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_measure",
    "jordan_decomposition",
    "signed_mass_delta",
    "total_variation",
    "centroid",
    "support_width",
]

_EPS = 1e-12


def normalize_measure(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalise non-negative ``weights`` to a probability measure.

    Returns ``(normalised, total_mass)``. If the total mass is below
    machine-epsilon the measure is treated as uniform (so downstream distances
    are well defined and deterministic for an all-zero profile).
    """
    w = np.asarray(weights, dtype=np.float64)
    if np.any(w < -_EPS):
        raise ValueError("normalize_measure expects non-negative weights")
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= _EPS:
        n = w.shape[0]
        return np.full(n, 1.0 / n), 0.0
    return w / total, total


def jordan_decomposition(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a signed profile into ``(positive_part, negative_part)``.

    Both parts are non-negative; ``profile == positive_part - negative_part``.
    """
    p = np.asarray(profile, dtype=np.float64)
    return np.clip(p, 0.0, None), np.clip(-p, 0.0, None)


def signed_mass_delta(ref: np.ndarray, alt: np.ndarray) -> float:
    """Net change in total signal, ``sum(alt) - sum(ref)``."""
    return float(
        np.asarray(alt, dtype=np.float64).sum()
        - np.asarray(ref, dtype=np.float64).sum()
    )


def total_variation(ref: np.ndarray, alt: np.ndarray) -> float:
    """L1 mass of the difference, ``sum(|alt - ref|)`` (magnitude of change)."""
    return float(
        np.abs(
            np.asarray(alt, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
        ).sum()
    )


def centroid(coords: np.ndarray, weights: np.ndarray) -> float:
    """Mass-weighted centre of a (not necessarily normalised) measure, in bp."""
    w = normalize_measure(weights)[0]
    return float(np.dot(np.asarray(coords, dtype=np.float64), w))


def support_width(coords: np.ndarray, weights: np.ndarray) -> float:
    """Spread of a measure as its mass-weighted standard deviation, in bp."""
    c = np.asarray(coords, dtype=np.float64)
    w, _ = normalize_measure(weights)
    mu = float(np.dot(c, w))
    var = float(np.dot((c - mu) ** 2, w))
    return float(np.sqrt(max(var, 0.0)))
