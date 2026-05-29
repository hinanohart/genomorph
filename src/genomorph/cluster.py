"""Mechanism clustering and the bootstrap statistics behind the CLAIM.

The headline statistic is a *paired* bootstrap of the difference in Adjusted
Rand Index (ARI) between the genomorph fingerprint and a baseline
representation, both clustered against the same ground-truth mechanism labels.
The CLAIM is upheld only if the 95% CI of ``ARI(fingerprint) - ARI(baseline)``
lies entirely above zero; otherwise it is withdrawn (downgraded to an
exploration/visualisation tool).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score

__all__ = ["cluster_labels", "ari_bootstrap_ci", "paired_ari_diff_ci", "BootstrapCI"]


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    n_boot: int

    @property
    def excludes_zero_above(self) -> bool:
        """True iff the whole 95% CI is strictly above zero."""
        return self.lo > 0.0

    def __str__(self) -> str:
        return f"{self.point:.4f} [95% CI {self.lo:.4f}, {self.hi:.4f}]"


def cluster_labels(
    x: np.ndarray, n_clusters: int, *, seed: int = 0, method: str = "kmeans"
) -> np.ndarray:
    """Deterministic clustering of fingerprint rows.

    ``kmeans`` (default) is fully determined by ``seed``; ``ward`` is
    agglomerative Ward linkage (no randomness).
    """
    x = np.asarray(x, dtype=np.float64)
    if method == "kmeans":
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)  # type: ignore[arg-type]
        return km.fit_predict(x)
    if method == "ward":
        ac = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        return ac.fit_predict(x)
    raise ValueError(f"unknown clustering method {method!r}")


def ari_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the ARI of a fixed partition.

    The point estimate is ``ARI(y_true, y_pred)``. The CI is obtained by
    resampling (true, pred) pairs with replacement -- isolating the sampling
    variability of the index for the given clustering. ARI is invariant to
    label permutation, so no alignment is needed.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]
    point = float(adjusted_rand_score(y_true, y_pred))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = adjusted_rand_score(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return BootstrapCI(point=point, lo=float(lo), hi=float(hi), n_boot=n_boot)


def paired_ari_diff_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> BootstrapCI:
    """Paired bootstrap CI of ``ARI(a) - ARI(b)`` against shared ``y_true``.

    ``a`` is the genomorph fingerprint clustering, ``b`` the baseline. The same
    resampled indices are applied to both, so the difference is paired and
    cancels shared sampling noise. ``excludes_zero_above`` being true is the
    machine-checkable condition for upholding the CLAIM.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    n = y_true.shape[0]
    point = float(
        adjusted_rand_score(y_true, y_pred_a) - adjusted_rand_score(y_true, y_pred_b)
    )
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = adjusted_rand_score(y_true[idx], y_pred_a[idx]) - adjusted_rand_score(
            y_true[idx], y_pred_b[idx]
        )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return BootstrapCI(point=point, lo=float(lo), hi=float(hi), n_boot=n_boot)
