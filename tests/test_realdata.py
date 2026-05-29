import numpy as np

from genomorph.adapters import MockBackend
from genomorph.eval import load_eqtl_subset, run_real_eval


def test_eqtl_subset_loads():
    vids, labels, mechs = load_eqtl_subset()
    assert mechs == ["expression", "splicing"]
    assert len(vids) == len(labels) == 167
    assert set(labels.tolist()) == {0, 1}
    # variant ids are chr_pos_ref_alt (hg38)
    for v in vids[:5]:
        chrom, pos, ref, alt = v.split("_")
        assert chrom.startswith("chr") and pos.isdigit()
        assert ref and alt


def test_subset_class_balance_reasonable():
    _, labels, _ = load_eqtl_subset()
    counts = np.bincount(labels)
    assert counts.min() > 50  # both classes well represented


def test_real_eval_mock_is_labeled_smoke_only():
    r = run_real_eval(MockBackend(), n_boot=100, seed=0)
    assert r["backend_is_real"] is False
    assert r["claim_upheld"] is False  # mock can never uphold the CLAIM
    assert "WIRING SMOKE" in r["note"]
    assert r["n_variants"] == 167
    assert r["mechanisms"] == ["expression", "splicing"]


def test_real_eval_max_variants_subsets():
    r = run_real_eval(MockBackend(), n_boot=50, seed=0, max_variants=40)
    assert r["n_variants"] == 40


def test_real_eval_output_wellformed():
    r = run_real_eval(MockBackend(), n_boot=50, seed=1)
    for key in ("ari_fingerprint", "ari_vep_scalar", "diff_vs_vep"):
        assert {"point", "lo", "hi", "n_boot"} <= set(r[key])
