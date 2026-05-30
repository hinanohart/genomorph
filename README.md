# genomorph

**Optimal-transport fingerprints of regulatory-variant effect *mechanisms*.**
Pre-alpha (v0.1.0a1) · MIT · CPU-only core · backend-agnostic.

A sequence-to-function model (Borzoi, Enformer, AlphaGenome, …) predicts, for a
variant, a REF and an ALT coverage profile per assay. The field then collapses
each track to a single effect-size scalar. genomorph keeps the **shape**: it
treats the REF→ALT difference as a measure on genomic coordinates and
decomposes it into

* a **shape** term — the 1-D Wasserstein-1 distance between the mass-normalised
  REF and ALT profiles (how the signal *redistributed*), and
* a signed **mass** term — the net change in total signal,

plus the spatial separation of *gain* from *loss* (a Jordan decomposition of the
difference) and a support-width ratio. The result is a small fixed-dimension
**fingerprint** that distinguishes mechanisms a magnitude scalar conflates:
a peak **shift** vs a **loss** vs a **gain** vs a **broadening**.

> genomorph never predicts variant effects itself and bundles no model weights.
> It is a *representation + evaluation* layer on top of whatever backend you use.

## The one claim

> On a controlled synthetic benchmark where each variant carries a known
> regulatory mechanism, the genomorph fingerprint clusters variants by mechanism
> at a **higher Adjusted Rand Index (ARI)** than the standard variant-effect
> summary (per-modality L2 magnitude, "vep-scalar"), using the *same* backend
> outputs. The claim is upheld only when the 95 % bootstrap CI of
> `ARI(fingerprint) − ARI(vep-scalar)` excludes zero.

### Measured (from `results/v0.1.0a1_separation.json`, 5 seeds, n=240, 1000 bootstraps)

In the **matched** regime the total amount of change is equalised across
mechanisms, so the magnitude scalar is *uninformative by construction* — this
isolates the contribution of shape:

| representation | ARI (mean over 5 seeds) |
|----------------|-------------------------|
| genomorph fingerprint | **0.57** (per-seed 0.52–0.64) |
| vep-scalar (standard)  | 0.02 |
| raw per-bin delta (strong control) | 0.40 |

Paired bootstrap `ARI(fingerprint) − ARI(vep-scalar)` (seed 0) = **0.62**,
95 % CI **[0.55, 0.68]** → excludes zero; the claim holds on **all 5 seeds**,
and MAD scaling helps on all 5 seeds.

In the **natural** regime (magnitudes left as each mechanism produces them, the
easy case) the fingerprint reaches ARI ≈ 0.95 vs 0.23 for the vep-scalar.

### What is *not* claimed

* **No calibration, no causal proof, no clinical or diagnostic use.** Mechanisms
  are *hypotheses* about the predicted effect's geometry.
* **The headline number is on synthetic data.** A real eQTL-Catalogue labelled
  subset (167 variants) is embedded and a reproducible harness is provided, but
  a genuine real-data ARI requires running a real backend (weights + reference
  genome) and is **not run in CI** — it is reproducible by you (see below). The
  in-CI real-data run uses the `mock` backend and is a *wiring smoke only* whose
  ARI is meaningless and labelled as such.
* **No backend weights or backend outputs are bundled or redistributed.**
* The AlphaGenome backend is **opt-in and non-commercial** (see NOTICE).

## Install

```bash
pip install genomorph                      # core (numpy/scipy/scikit-learn/POT)
pip install "genomorph[borzoi]"            # + Borzoi backend (weights MIT)
pip install "genomorph[enformer]"          # + Enformer backend (MIT)
```

## Quickstart

```bash
# Reproduce the headline synthetic benchmark
genomorph benchmark --regime matched --out results.json

# Fingerprint one variant (mock backend = no weights needed, for wiring/demo)
genomorph fingerprint chr1_108004887_G_T --backend mock

# Real eQTL-Catalogue evaluation:
#   mock  -> wiring smoke (fast, ARI meaningless, clearly labelled)
genomorph eval-real --backend mock
#   borzoi/enformer -> genuine (needs weights + hg38 FASTA)
genomorph eval-real --backend enformer --genome-fasta /path/hg38.fa
```

```python
import genomorph as gm

backend = gm.get_backend("mock")                 # or "borzoi" / "enformer"
effect = backend.predict("chr1_108004887_G_T")   # -> VariantEffect (REF/ALT tracks)
fp = gm.FingerprintExtractor(backend.modalities).transform_one(effect)
```

## How the benchmark is fair

The baseline is the *same backend output*, summarised the way the field
summarises it (per-modality L2 magnitude). A second, stronger control clusters
the **full per-bin difference** (no information discarded). The fingerprint beats
both. To pre-empt a "magnitude was rigged useless" objection we report two
regimes — `matched` (magnitude uninformative by construction) and `natural`.
All numbers come from `results/v0.1.0a1_separation.json`; none are hand-typed.

## Scope

* **v0.1.0a (now):** OT shape+mass fingerprint · Borzoi(default)/Enformer/mock
  backends · synthetic benchmark · embedded real eQTL labels + harness · Oklab
  visualisation.
* **v0.2:** splicing-mechanism block · Gromov–Wasserstein cross-modality ·
  information-geometry features.
* **v0.3:** contact-map topological (persistent-homology) features.

## License & data

Code: MIT. Backends and the eQTL Catalogue evaluation subset carry their
own licenses — see  for the full matrix. The embedded eQTL
Catalogue subset is redistributed under CC-BY-4.0 (Kerimov et al., Nat Genet
2021).
