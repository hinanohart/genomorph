import numpy as np
import pytest

from genomorph.cluster import ari_bootstrap_ci, cluster_labels, paired_ari_diff_ci
from genomorph.eval import (
    baseline_raw_delta,
    baseline_vep_scalar,
    fingerprint_matrix,
    generate_cohort,
    run_benchmark,
)


def test_perfect_clustering_ari_one():
    x = np.vstack([np.zeros((20, 2)) + [0, 0], np.zeros((20, 2)) + [10, 10]])
    y = np.array([0] * 20 + [1] * 20)
    pred = cluster_labels(x, 2, seed=0)
    ci = ari_bootstrap_ci(y, pred, n_boot=200, seed=0)
    assert ci.point == pytest.approx(1.0)
    assert ci.excludes_zero_above


def test_paired_diff_zero_when_identical():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    d = paired_ari_diff_ci(y, pred, pred, n_boot=100, seed=0)
    assert d.point == pytest.approx(0.0)


def test_cohort_shapes_and_labels():
    effects, y = generate_cohort(n_per_mechanism=10, seed=0)
    assert len(effects) == 40
    assert set(y.tolist()) == {0, 1, 2, 3}
    assert effects[0].modalities == ["RNA", "DNASE", "H3K27ac"]


def test_matched_regime_makes_vep_magnitude_uninformative():
    effects, y = generate_cohort(n_per_mechanism=40, regime="matched", seed=0)
    x_vep = baseline_vep_scalar(effects, ["RNA", "DNASE", "H3K27ac"])
    pred = cluster_labels(x_vep, 4, seed=0)
    ci = ari_bootstrap_ci(y, pred, n_boot=200, seed=0)
    assert ci.point < 0.2  # magnitude alone cannot recover mechanism here


def test_baselines_have_expected_dims():
    effects, _ = generate_cohort(n_per_mechanism=5, seed=0)
    mods = ["RNA", "DNASE", "H3K27ac"]
    assert baseline_vep_scalar(effects, mods).shape == (20, 3)
    assert baseline_raw_delta(effects, mods).shape[0] == 20
    assert fingerprint_matrix(effects, mods).shape == (20, 15)


def test_claim_holds_matched_regime():
    r = run_benchmark(regime="matched", n_per_mechanism=50, n_boot=400, seed=0)
    assert r.claim_upheld
    assert r.ari_fingerprint["point"] > r.ari_vep_scalar["point"]
    assert r.diff_vs_vep["lo"] > 0
    assert r.mad_helps


def test_benchmark_deterministic():
    a = run_benchmark(regime="matched", n_per_mechanism=20, n_boot=200, seed=7)
    b = run_benchmark(regime="matched", n_per_mechanism=20, n_boot=200, seed=7)
    assert a.ari_fingerprint["point"] == b.ari_fingerprint["point"]
    assert a.diff_vs_vep["lo"] == b.diff_vs_vep["lo"]
