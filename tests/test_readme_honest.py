"""Mechanically enforce honest marketing.

The README must (a) cite only numbers that match the canonical results JSON,
(b) contain no placeholders, (c) contain no banned hype phrases, and (d) carry
the non-CLAIM disclaimers. If a future edit drifts the headline number or
slips in a placeholder/hype phrase, CI fails.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
RESULTS = json.loads((ROOT / "results" / "v0.1.0a1_separation.json").read_text())


def test_headline_numbers_match_results():
    m = RESULTS["synthetic"]["matched"]
    s0 = RESULTS["synthetic_single_seed0"]["matched"]
    expected = {
        "fingerprint mean": f"{m['ari_fingerprint']['mean']:.2f}",
        "vep mean": f"{m['ari_vep_scalar']['mean']:.2f}",
        "raw mean": f"{m['ari_raw_delta']['mean']:.2f}",
        "diff seed0": f"{s0['diff_vs_vep']['point']:.2f}",
        "ci lo": f"{s0['diff_vs_vep']['lo']:.2f}",
        "ci hi": f"{s0['diff_vs_vep']['hi']:.2f}",
    }
    missing = {k: v for k, v in expected.items() if v not in README}
    assert not missing, f"README missing/stale numbers: {missing}"


def test_claim_actually_upheld_in_results():
    # The README states the claim holds on all seeds; the results must agree.
    assert RESULTS["synthetic"]["matched"]["claim_upheld_all_seeds"] is True
    assert RESULTS["synthetic"]["matched"]["mad_helps_all_seeds"] is True


def test_no_placeholders():
    banned = [
        "MEASURED@",
        "TODO",
        "FIXME",
        "XXX",
        "lorem",
        "PLACEHOLDER",
        "<!--",
        "TBD",
        "coming soon",
    ]
    hits = [b for b in banned if b.lower() in README.lower()]
    assert not hits, f"placeholder tokens in README: {hits}"


def test_no_hype_phrases():
    banned = [
        "state-of-the-art",
        "state of the art",
        "revolutionary",
        "world's best",
        "fully automatic",
        "permanent",
        "guaranteed",
        "breakthrough",
        "unprecedented",
        "magic",
    ]
    hits = [b for b in banned if b in README.lower()]
    assert not hits, f"hype phrases in README: {hits}"


def test_disclaimers_present():
    for needed in [
        "no calibration",
        "no causal",
        "clinical",
        "synthetic",
        "non-commercial",
        "hypotheses",
        "not run in ci",
    ]:
        assert needed in README.lower(), f"missing disclaimer: {needed!r}"


def test_real_data_honesty_stated():
    low = README.lower()
    assert "wiring smoke" in low
    assert "reproducible" in low
    # must not claim a measured real-data ARI headline
    assert "real-data ari" not in low or "requires running a real backend" in low


@pytest.mark.parametrize("number", ["0.57", "0.02", "0.40"])
def test_specific_headline_present(number):
    assert number in README
