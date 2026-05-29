import json

import matplotlib
import numpy as np

matplotlib.use("Agg")

from genomorph.cli import main
from genomorph.eval import fingerprint_matrix, generate_cohort, run_benchmark_multiseed
from genomorph.viz import plot_fingerprints


def test_cli_fingerprint_returns_zero(capsys):
    assert main(["fingerprint", "chr1_1000_A_T", "--backend", "mock"]) == 0
    assert "RNA:w1_shape" in capsys.readouterr().out


def test_cli_benchmark_writes_json(tmp_path):
    out = tmp_path / "r.json"
    rc = main(
        [
            "benchmark",
            "--regime",
            "natural",
            "--n",
            "8",
            "--n-boot",
            "50",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["regime"] == "natural"
    assert "ari_fingerprint" in payload


def test_plot_fingerprints_returns_axes():
    effects, y = generate_cohort(n_per_mechanism=8, seed=0)
    x = fingerprint_matrix(effects, ["RNA", "DNASE", "H3K27ac"])
    ax = plot_fingerprints(x, y, title="t")
    assert ax.get_title() == "t"
    assert len(ax.collections) == len(np.unique(y))


def test_multiseed_claim_stable_matched():
    r = run_benchmark_multiseed(
        regime="matched", n_per_mechanism=30, seeds=(0, 1, 2), n_boot=200
    )
    assert r["claim_upheld_all_seeds"]
    assert r["ari_fingerprint"]["mean"] > r["ari_vep_scalar"]["mean"]
