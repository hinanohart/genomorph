"""Evaluation harness: synthetic mechanism benchmark + fair baselines.

The CLAIM is tested on a controlled synthetic cohort where each variant carries
a known regulatory *mechanism* (the ground-truth label) injected on top of
nuisance variation (peak position jitter, amplitude variation, per-bin noise).

Two baselines are clustered against the same labels for a fair comparison that
uses the *same* backend outputs and differs only in the representation:

* ``vep_scalar``  -- per-modality L2 magnitude of the ALT-REF difference, i.e.
  the standard variant-effect-prediction summary (Borzoi/Enformer collapse each
  track to a magnitude). This is what the field uses.
* ``raw_delta``   -- the full concatenated per-bin difference (a strong control:
  no information is discarded, only un-summarised).

The genomorph fingerprint sits between them in dimension but encodes the
*geometry* of the change. The headline result is the paired bootstrap CI of
``ARI(fingerprint) - ARI(vep_scalar)``.

To pre-empt a "magnitude was rigged useless" critique we run two regimes:
``matched`` (total change equalised across mechanisms, so magnitude is
uninformative by construction -- the case standard VEP cannot handle) and
``natural`` (magnitudes left as each mechanism produces them).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .cluster import ari_bootstrap_ci, cluster_labels, paired_ari_diff_ci
from .fingerprint import FingerprintExtractor, MADScaler
from .measures import total_variation
from .types import TrackProfile, VariantEffect

__all__ = [
    "MECHANISMS",
    "generate_cohort",
    "baseline_vep_scalar",
    "baseline_raw_delta",
    "fingerprint_matrix",
    "run_benchmark",
    "run_benchmark_multiseed",
    "BenchmarkResult",
    "load_eqtl_subset",
    "run_real_eval",
]

MECHANISMS = ("shift", "loss", "gain", "broaden")


def _gaussian(
    coords: np.ndarray, centre: float, width: float, amp: float
) -> np.ndarray:
    return amp * np.exp(-0.5 * ((coords - centre) / width) ** 2)


def _inject(
    mechanism: str, coords: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Build (ref, alt) for one variant under a mechanism, with nuisance jitter."""
    n = coords.shape[0]
    base = 0.05
    centre = rng.uniform(0.35, 0.65) * n  # nuisance: peak position varies
    width = rng.uniform(4.0, 7.0)
    amp = rng.uniform(1.5, 3.5)  # nuisance: amplitude varies (shared across mechs)
    ref = _gaussian(coords, centre, width, amp) + base

    # Parameter ranges are deliberately small and overlapping so mechanisms are
    # genuinely confusable under noise (a small shift can mimic asymmetric
    # broadening); the benchmark is not trivially separable.
    if mechanism == "shift":
        d = rng.choice([-1, 1]) * rng.uniform(3.0, 8.0)
        alt = _gaussian(coords, centre + d, width, amp) + base
    elif mechanism == "loss":
        f = rng.uniform(0.25, 0.6)  # peak attenuated (overlaps broaden's area loss)
        alt = _gaussian(coords, centre, width, amp * f) + base
    elif mechanism == "gain":
        # new secondary peak appears nearby; primary unchanged
        off = rng.choice([-1, 1]) * rng.uniform(8.0, 16.0)
        alt = ref + _gaussian(coords, centre + off, width, amp * rng.uniform(0.4, 0.9))
    elif mechanism == "broaden":
        g = rng.uniform(1.3, 1.9)  # peak spreads, amplitude drops to ~preserve area
        alt = _gaussian(coords, centre, width * g, amp / np.sqrt(g)) + base
    else:  # pragma: no cover - guarded by caller
        raise ValueError(mechanism)
    return ref, alt


def generate_cohort(
    *,
    n_per_mechanism: int = 60,
    modalities: tuple[str, ...] = ("RNA", "DNASE", "H3K27ac"),
    n_bins: int = 128,
    bin_size: int = 128,
    noise: float = 0.18,
    regime: str = "matched",
    seed: int = 0,
) -> tuple[list[VariantEffect], np.ndarray]:
    """Generate a labelled synthetic cohort.

    ``regime='matched'`` rescales every variant's per-modality difference to a
    shared target total-variation drawn from one distribution, so the standard
    VEP magnitude is uninformative about mechanism by construction.
    ``regime='natural'`` leaves magnitudes as produced.
    """
    if regime not in ("matched", "natural"):
        raise ValueError("regime must be 'matched' or 'natural'")
    rng = np.random.default_rng(seed)
    coords = np.arange(n_bins, dtype=np.float64)
    effects: list[VariantEffect] = []
    labels: list[int] = []

    for label, mech in enumerate(MECHANISMS):
        for _ in range(n_per_mechanism):
            target_tv = rng.uniform(8.0, 20.0)  # shared across mechanisms
            tracks = []
            for m in modalities:
                ref, alt = _inject(mech, coords, rng)
                # per-bin nuisance noise on both profiles
                ref = np.clip(ref + rng.normal(0, noise, n_bins), 0.0, None)
                alt = np.clip(alt + rng.normal(0, noise, n_bins), 0.0, None)
                if regime == "matched":
                    tv = total_variation(ref, alt)
                    if tv > 1e-9:
                        alt = np.clip(ref + (alt - ref) * (target_tv / tv), 0.0, None)
                tracks.append(
                    TrackProfile(modality=m, bin_size=bin_size, ref=ref, alt=alt)
                )
            effects.append(
                VariantEffect(variant_id=f"{mech}_{len(effects)}", tracks=tracks)
            )
            labels.append(label)

    return effects, np.asarray(labels)


def baseline_vep_scalar(
    effects: list[VariantEffect], modalities: list[str]
) -> np.ndarray:
    """Standard VEP representation: per-modality L2 magnitude of the difference."""
    rows = []
    for e in effects:
        present = {t.modality: t for t in e.tracks}
        rows.append(
            [
                float(np.linalg.norm(present[m].delta)) if m in present else 0.0
                for m in modalities
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def baseline_raw_delta(
    effects: list[VariantEffect], modalities: list[str]
) -> np.ndarray:
    """Strong control: full concatenated per-bin difference profiles."""
    rows = []
    for e in effects:
        present = {t.modality: t for t in e.tracks}
        blocks = []
        for m in modalities:
            if m in present:
                blocks.append(present[m].delta)
            else:
                blocks.append(np.zeros(present[next(iter(present))].n_bins))
        rows.append(np.concatenate(blocks))
    return np.asarray(rows, dtype=np.float64)


def fingerprint_matrix(
    effects: list[VariantEffect], modalities: list[str], *, scale: bool = True
) -> np.ndarray:
    """genomorph fingerprint matrix, MAD-scaled across the cohort by default."""
    extractor = FingerprintExtractor(modalities)
    x = extractor.transform(effects)
    if scale:
        x = MADScaler().fit_transform(x)
    return x


@dataclass
class BenchmarkResult:
    regime: str
    n: int
    n_clusters: int
    ari_fingerprint: dict
    ari_vep_scalar: dict
    ari_raw_delta: dict
    ari_fingerprint_unscaled: dict
    diff_vs_vep: dict
    diff_vs_raw: dict
    claim_upheld: bool
    mad_helps: bool

    def to_dict(self) -> dict:
        return asdict(self)


def run_benchmark(
    *,
    regime: str = "matched",
    n_per_mechanism: int = 60,
    modalities: tuple[str, ...] = ("RNA", "DNASE", "H3K27ac"),
    n_boot: int = 1000,
    seed: int = 0,
) -> BenchmarkResult:
    """Run the full synthetic benchmark and evaluate the CLAIM condition."""
    effects, y = generate_cohort(
        n_per_mechanism=n_per_mechanism,
        modalities=modalities,
        regime=regime,
        seed=seed,
    )
    mods = list(modalities)
    k = len(MECHANISMS)

    x_fp = fingerprint_matrix(effects, mods, scale=True)
    x_fp_raw = fingerprint_matrix(effects, mods, scale=False)
    x_vep = baseline_vep_scalar(effects, mods)
    x_raw = baseline_raw_delta(effects, mods)

    pred_fp = cluster_labels(x_fp, k, seed=seed)
    pred_fp_unscaled = cluster_labels(x_fp_raw, k, seed=seed)
    pred_vep = cluster_labels(x_vep, k, seed=seed)
    pred_raw = cluster_labels(x_raw, k, seed=seed)

    ci_fp = ari_bootstrap_ci(y, pred_fp, n_boot=n_boot, seed=seed)
    ci_fp_unscaled = ari_bootstrap_ci(y, pred_fp_unscaled, n_boot=n_boot, seed=seed)
    ci_vep = ari_bootstrap_ci(y, pred_vep, n_boot=n_boot, seed=seed)
    ci_raw = ari_bootstrap_ci(y, pred_raw, n_boot=n_boot, seed=seed)
    diff_vep = paired_ari_diff_ci(y, pred_fp, pred_vep, n_boot=n_boot, seed=seed)
    diff_raw = paired_ari_diff_ci(y, pred_fp, pred_raw, n_boot=n_boot, seed=seed)

    return BenchmarkResult(
        regime=regime,
        n=len(effects),
        n_clusters=k,
        ari_fingerprint=vars(ci_fp),
        ari_vep_scalar=vars(ci_vep),
        ari_raw_delta=vars(ci_raw),
        ari_fingerprint_unscaled=vars(ci_fp_unscaled),
        diff_vs_vep=vars(diff_vep),
        diff_vs_raw=vars(diff_raw),
        claim_upheld=bool(diff_vep.excludes_zero_above),
        mad_helps=bool(ci_fp.point > ci_fp_unscaled.point),
    )


def run_benchmark_multiseed(
    *,
    regime: str = "matched",
    n_per_mechanism: int = 60,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    n_boot: int = 1000,
) -> dict:
    """Run the benchmark over several seeds and summarise stability.

    Reports the per-seed point estimates and the CLAIM condition holding on
    *every* seed, so the headline cannot rest on a cherry-picked seed.
    """
    results = [
        run_benchmark(
            regime=regime, n_per_mechanism=n_per_mechanism, n_boot=n_boot, seed=s
        )
        for s in seeds
    ]
    fp = np.array([r.ari_fingerprint["point"] for r in results])
    vep = np.array([r.ari_vep_scalar["point"] for r in results])
    raw = np.array([r.ari_raw_delta["point"] for r in results])
    diff = np.array([r.diff_vs_vep["point"] for r in results])
    return {
        "regime": regime,
        "seeds": list(seeds),
        "n_per_seed": results[0].n,
        "ari_fingerprint": {
            "mean": float(fp.mean()),
            "min": float(fp.min()),
            "max": float(fp.max()),
            "per_seed": fp.round(4).tolist(),
        },
        "ari_vep_scalar": {
            "mean": float(vep.mean()),
            "min": float(vep.min()),
            "max": float(vep.max()),
            "per_seed": vep.round(4).tolist(),
        },
        "ari_raw_delta": {
            "mean": float(raw.mean()),
            "min": float(raw.min()),
            "max": float(raw.max()),
            "per_seed": raw.round(4).tolist(),
        },
        "diff_vs_vep_point": {
            "mean": float(diff.mean()),
            "min": float(diff.min()),
            "max": float(diff.max()),
        },
        "claim_upheld_all_seeds": bool(all(r.claim_upheld for r in results)),
        "mad_helps_all_seeds": bool(all(r.mad_helps for r in results)),
    }


_EQTL_SUBSET = "macrophage_expr_vs_splice.tsv"


def load_eqtl_subset() -> tuple[list[str], np.ndarray, list[str]]:
    """Load the embedded eQTL Catalogue labelled subset (CC-BY-4.0).

    Returns ``(variant_ids, labels, mechanism_names)`` where labels index
    ``mechanism_names`` (``expression`` vs ``splicing``). Variant ids are
    ``chr_pos_ref_alt`` (hg38), ready to feed to any backend.
    """
    data_dir = Path(__file__).resolve().parent / "data" / "eqtl_subset"
    text = (data_dir / _EQTL_SUBSET).read_text()
    rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    mech_names = sorted({r["mechanism"] for r in rows})
    idx = {m: i for i, m in enumerate(mech_names)}
    variant_ids = [r["variant"] for r in rows]
    labels = np.array([idx[r["mechanism"]] for r in rows])
    return variant_ids, labels, mech_names


def run_real_eval(
    backend,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    max_variants: int | None = None,
) -> dict:
    """Run the real-data pipeline: backend -> fingerprint/baseline -> ARI.

    Backend-agnostic. With the ``mock`` backend this is a *wiring smoke* only:
    the mock effects carry no biology, so the reported ARI is meaningless and
    is labelled ``backend_is_real=False``. A genuine result requires a real
    sequence-to-function backend (Borzoi/Enformer) with weights and a reference
    genome; that run is reproducible by the user and is NOT performed in CI.
    """
    variant_ids, labels, mech_names = load_eqtl_subset()
    if max_variants is not None:
        variant_ids = variant_ids[:max_variants]
        labels = labels[:max_variants]
    effects = [backend.predict(v) for v in variant_ids]
    mods = list(backend.modalities)
    k = len(mech_names)

    x_fp = fingerprint_matrix(effects, mods, scale=True)
    x_vep = baseline_vep_scalar(effects, mods)
    pred_fp = cluster_labels(x_fp, k, seed=seed)
    pred_vep = cluster_labels(x_vep, k, seed=seed)

    ci_fp = ari_bootstrap_ci(labels, pred_fp, n_boot=n_boot, seed=seed)
    ci_vep = ari_bootstrap_ci(labels, pred_vep, n_boot=n_boot, seed=seed)
    diff = paired_ari_diff_ci(labels, pred_fp, pred_vep, n_boot=n_boot, seed=seed)

    is_real = getattr(backend, "name", "") not in ("mock", "")
    return {
        "backend": getattr(backend, "name", "?"),
        "backend_is_real": bool(is_real),
        "n_variants": len(variant_ids),
        "mechanisms": mech_names,
        "modalities": mods,
        "ari_fingerprint": vars(ci_fp),
        "ari_vep_scalar": vars(ci_vep),
        "diff_vs_vep": vars(diff),
        "claim_upheld": bool(is_real and diff.excludes_zero_above),
        "note": (
            "REAL result"
            if is_real
            else "WIRING SMOKE ONLY (mock backend has no biology; ARI is meaningless)"
        ),
    }
