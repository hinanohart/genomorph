"""CI assertions over CLI-produced JSON (run end-to-end, locally reproducible).

Usage:
    python scripts/ci_assert.py synthetic bench.json
    python scripts/ci_assert.py real-smoke real.json
"""

from __future__ import annotations

import json
import sys


def _synthetic(path: str) -> None:
    d = json.load(open(path))
    assert d["claim_upheld"], "synthetic CLAIM not upheld"
    assert d["diff_vs_vep"]["lo"] > 0, "paired diff CI crosses zero"
    assert d["ari_fingerprint"]["point"] > d["ari_vep_scalar"]["point"], (
        "fingerprint <= vep"
    )
    print(
        "synthetic CLAIM upheld:",
        round(d["ari_fingerprint"]["point"], 3),
        ">",
        round(d["ari_vep_scalar"]["point"], 3),
    )


def _real_smoke(path: str) -> None:
    d = json.load(open(path))
    assert d["backend_is_real"] is False, "mock smoke must report backend_is_real=False"
    assert d["claim_upheld"] is False, "mock smoke must not uphold the CLAIM"
    assert d["n_variants"] == 167, f"expected 167 real variants, got {d['n_variants']}"
    print("real-smoke wiring ok:", d["note"])


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write(__doc__ or "")
        return 2
    kind, path = sys.argv[1], sys.argv[2]
    if kind == "synthetic":
        _synthetic(path)
    elif kind == "real-smoke":
        _real_smoke(path)
    else:
        sys.stderr.write(f"unknown kind {kind!r}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
