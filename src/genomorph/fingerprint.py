"""Fixed-dimension mechanism fingerprint of a variant effect.

For each modality the fingerprint records five numbers that, together, separate
the regulatory mechanisms a single effect-size scalar cannot:

* ``w1_shape``     -- W1 between the mass-normalised REF and ALT profiles: how
                      far the signal *distribution* moved (a peak shift is large
                      here even when total signal is unchanged).
* ``sign_shift``   -- signed centroid shift (bp), ALT minus REF: the *direction*
                      of movement.
* ``mass_delta``   -- signed change in total signal (a gain/loss of a peak).
* ``jordan_w1``    -- W1 between the normalised positive and negative parts of
                      the ALT-minus-REF difference: the spatial separation of
                      *where signal was gained* from *where it was lost*. Large
                      for a shift, ~0 for a pure scaling.
* ``width_ratio``  -- ALT support width / REF support width: spreading vs
                      sharpening of the signal.

Feature scales differ wildly across modalities and feature types (bp vs
coverage vs dimensionless), so :class:`MADScaler` (median / median-absolute-
deviation) is applied across the cohort before any distance or clustering --
this is the single most load-bearing step for cross-modality comparability.
"""

from __future__ import annotations

import numpy as np

from .measures import (
    centroid,
    jordan_decomposition,
    signed_mass_delta,
    support_width,
)
from .ot import wasserstein1d
from .types import TrackProfile, VariantEffect

__all__ = [
    "FEATURES_PER_MODALITY",
    "modality_features",
    "FingerprintExtractor",
    "MADScaler",
]

FEATURES_PER_MODALITY = (
    "w1_shape",
    "sign_shift",
    "mass_delta",
    "jordan_w1",
    "width_ratio",
)

_EPS = 1e-12


def modality_features(track: TrackProfile) -> np.ndarray:
    """The five mechanism features for one modality, in raw units."""
    coords = track.coords_bp
    ref = track.ref
    alt = track.alt

    if track.signed:
        # For signed assays the meaningful "distribution" is over magnitudes;
        # use |.| for the shape comparison and keep the sign in mass_delta.
        ref_mag = np.abs(ref)
        alt_mag = np.abs(alt)
    else:
        ref_mag = ref
        alt_mag = alt

    w1_shape = wasserstein1d(coords, ref_mag, alt_mag, normalize=True)
    sign_shift = centroid(coords, alt_mag) - centroid(coords, ref_mag)
    mass_delta = signed_mass_delta(ref, alt)

    delta = alt - ref
    pos, neg = jordan_decomposition(delta)
    pos_mass = float(pos.sum())
    neg_mass = float(neg.sum())
    # Spatial separation of gain vs loss is only defined when both exist;
    # a pure gain or pure loss has no "shift" (that is carried by mass_delta).
    if pos_mass > _EPS and neg_mass > _EPS:
        jordan_w1 = wasserstein1d(coords, pos, neg, normalize=True)
    else:
        jordan_w1 = 0.0

    ref_w = support_width(coords, ref_mag)
    alt_w = support_width(coords, alt_mag)
    width_ratio = (alt_w + track.bin_size) / (ref_w + track.bin_size)

    return np.array(
        [w1_shape, sign_shift, mass_delta, jordan_w1, width_ratio],
        dtype=np.float64,
    )


class FingerprintExtractor:
    """Builds a fixed-length fingerprint over a canonical modality order.

    Modalities absent from a given variant contribute a zero block, so every
    fingerprint has length ``5 * len(modalities)`` regardless of backend.
    """

    def __init__(self, modalities: list[str]):
        if not modalities:
            raise ValueError("at least one modality is required")
        self.modalities = list(modalities)

    @property
    def dim(self) -> int:
        return len(self.modalities) * len(FEATURES_PER_MODALITY)

    @property
    def feature_names(self) -> list[str]:
        return [f"{m}:{f}" for m in self.modalities for f in FEATURES_PER_MODALITY]

    def transform_one(self, effect: VariantEffect) -> np.ndarray:
        present = {t.modality: t for t in effect.tracks}
        blocks = []
        for m in self.modalities:
            if m in present:
                blocks.append(modality_features(present[m]))
            else:
                blocks.append(np.zeros(len(FEATURES_PER_MODALITY)))
        return np.concatenate(blocks)

    def transform(self, effects: list[VariantEffect]) -> np.ndarray:
        return np.vstack([self.transform_one(e) for e in effects])


class MADScaler:
    """Robust per-feature scaler: ``(x - median) / (1.4826 * MAD)``.

    The 1.4826 factor makes the MAD a consistent estimator of the standard
    deviation under normality. Features with zero MAD (constant across the
    cohort) are left centred but unscaled to avoid division blow-ups.
    """

    SCALE = 1.4826

    def __init__(self) -> None:
        self.median_: np.ndarray | None = None
        self.mad_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "MADScaler":
        x = np.asarray(x, dtype=np.float64)
        self.median_ = np.median(x, axis=0)
        mad = np.median(np.abs(x - self.median_), axis=0) * self.SCALE
        mad[mad <= _EPS] = 1.0
        self.mad_ = mad
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.mad_ is None:
            raise RuntimeError("MADScaler must be fit before transform")
        return (np.asarray(x, dtype=np.float64) - self.median_) / self.mad_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)
