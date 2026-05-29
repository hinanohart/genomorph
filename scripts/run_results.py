"""Produce the canonical results JSON for v0.1.0a1.

Runs the synthetic mechanism benchmark over several seeds in both regimes and
the real-data wiring smoke, stamps the environment, and writes
results/v0.1.0a1_separation.json. README numbers are read from this file only.

Run:  python scripts/run_results.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy
import scipy
import sklearn

import ot as pot
from genomorph import __version__
from genomorph.adapters import MockBackend
from genomorph.eval import run_benchmark, run_benchmark_multiseed, run_real_eval

OUT = Path(__file__).resolve().parent.parent / "results" / "v0.1.0a1_separation.json"
SEEDS = (0, 1, 2, 3, 4)
N_PER_MECH = 60
N_BOOT = 1000


def main() -> int:
    env = {
        "genomorph": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "POT": pot.__version__,
    }
    payload = {
        "env": env,
        "config": {
            "seeds": list(SEEDS),
            "n_per_mechanism": N_PER_MECH,
            "n_boot": N_BOOT,
            "n_clusters": 4,
        },
        "synthetic": {},
        "synthetic_single_seed0": {},
        "real_smoke": run_real_eval(MockBackend(), n_boot=N_BOOT, seed=0),
    }
    for regime in ("matched", "natural"):
        payload["synthetic"][regime] = run_benchmark_multiseed(
            regime=regime, n_per_mechanism=N_PER_MECH, seeds=SEEDS, n_boot=N_BOOT
        )
        payload["synthetic_single_seed0"][regime] = run_benchmark(
            regime=regime, n_per_mechanism=N_PER_MECH, n_boot=N_BOOT, seed=0
        ).to_dict()

    headline = payload["synthetic"]["matched"]
    payload["claim"] = {
        "headline_regime": "matched",
        "ari_fingerprint_mean": headline["ari_fingerprint"]["mean"],
        "ari_vep_scalar_mean": headline["ari_vep_scalar"]["mean"],
        "ari_raw_delta_mean": headline["ari_raw_delta"]["mean"],
        "diff_vs_vep_seed0_ci": payload["synthetic_single_seed0"]["matched"][
            "diff_vs_vep"
        ],
        "upheld_all_seeds": headline["claim_upheld_all_seeds"],
        "mad_helps_all_seeds": headline["mad_helps_all_seeds"],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    sys.stderr.write(f"wrote {OUT}\n")
    c = payload["claim"]
    sys.stderr.write(
        f"matched: fingerprint ARI={c['ari_fingerprint_mean']:.3f} "
        f"vep={c['ari_vep_scalar_mean']:.3f} raw={c['ari_raw_delta_mean']:.3f} "
        f"upheld_all_seeds={c['upheld_all_seeds']}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
