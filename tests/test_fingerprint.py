import numpy as np
import pytest

from genomorph.fingerprint import (
    FEATURES_PER_MODALITY,
    FingerprintExtractor,
    MADScaler,
    modality_features,
)
from genomorph.types import TrackProfile, VariantEffect


def _peak(centre, width=4.0, amp=2.0, n=128):
    c = np.arange(n, dtype=float)
    return amp * np.exp(-0.5 * ((c - centre) / width) ** 2) + 0.05


def _effect(ref, alt, modality="RNA", bin_size=1):
    return VariantEffect("v", [TrackProfile(modality, bin_size, ref, alt)])


def test_feature_vector_shape_and_names():
    ex = FingerprintExtractor(["RNA", "DNASE"])
    assert ex.dim == 2 * len(FEATURES_PER_MODALITY)
    assert ex.feature_names[0] == "RNA:w1_shape"
    assert len(ex.feature_names) == ex.dim


def test_shift_mechanism_has_large_jordan_w1():
    # A pure spatial shift separates gain and loss -> large jordan_w1.
    ref = _peak(60)
    alt = _peak(72)
    feats = dict(
        zip(FEATURES_PER_MODALITY, modality_features(TrackProfile("RNA", 1, ref, alt)))
    )
    assert feats["jordan_w1"] > 5.0
    assert feats["sign_shift"] > 5.0  # centroid moved right


def test_loss_mechanism_negative_mass_delta_small_jordan():
    ref = _peak(60, amp=3.0)
    alt = _peak(60, amp=1.0)  # attenuated, same place
    feats = dict(
        zip(FEATURES_PER_MODALITY, modality_features(TrackProfile("RNA", 1, ref, alt)))
    )
    assert feats["mass_delta"] < 0
    assert feats["jordan_w1"] == pytest.approx(0.0)  # pure loss, no spatial shift


def test_gain_mechanism_positive_mass_delta():
    ref = _peak(60, amp=1.0)
    alt = _peak(60, amp=3.0)
    feats = dict(
        zip(FEATURES_PER_MODALITY, modality_features(TrackProfile("RNA", 1, ref, alt)))
    )
    assert feats["mass_delta"] > 0
    assert feats["jordan_w1"] == pytest.approx(0.0)


def test_broaden_mechanism_width_ratio_above_one():
    # Equal-mass peaks isolate spread: width_ratio is the ratio of mass-weighted
    # support widths, so comparing equal-mass profiles removes the pedestal/mass
    # confound a non-area-preserving peak would introduce.
    c = np.arange(128, dtype=float)
    ref = np.exp(-0.5 * ((c - 60) / 3.0) ** 2)
    alt = np.exp(-0.5 * ((c - 60) / 9.0) ** 2)
    ref = ref / ref.sum()
    alt = alt / alt.sum()
    feats = dict(
        zip(FEATURES_PER_MODALITY, modality_features(TrackProfile("RNA", 1, ref, alt)))
    )
    assert feats["width_ratio"] > 1.3
    assert feats["mass_delta"] == pytest.approx(0.0, abs=1e-9)


def test_missing_modality_is_zero_block():
    ex = FingerprintExtractor(["RNA", "DNASE"])
    eff = _effect(_peak(60), _peak(64))  # only RNA
    vec = ex.transform_one(eff)
    assert np.allclose(vec[len(FEATURES_PER_MODALITY) :], 0.0)


def test_mad_scaler_centers_median_to_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, (200, 4))
    z = MADScaler().fit_transform(x)
    assert np.allclose(np.median(z, axis=0), 0.0, atol=1e-9)


def test_mad_scaler_constant_feature_no_blowup():
    x = np.ones((10, 3))
    z = MADScaler().fit_transform(x)
    assert np.all(np.isfinite(z))


def test_fingerprint_deterministic():
    eff = _effect(_peak(60), _peak(70))
    ex = FingerprintExtractor(["RNA"])
    a = ex.transform_one(eff)
    b = ex.transform_one(eff)
    assert np.array_equal(a, b)
