# genomorph

> Pre-alpha (v0.1.0a1). Backend-agnostic optimal-transport fingerprints of
> regulatory-variant effect **mechanisms** — shape + signed mass — CPU-only.

genomorph takes a sequence-to-function model's REF→ALT per-track difference and
decomposes it into a **shape** component (1-D Wasserstein distance between the
mass-normalised profiles) and a signed **mass** component, producing a
fixed-dimension fingerprint that separates regulatory mechanisms a single
effect-size scalar conflates (peak shift vs loss vs gain vs broadening).

<!-- MEASURED@S7: headline numbers filled from results/ JSON in S7 -->

## Status

Scaffold + core under active build. Numbers and full documentation are written
in stage S7 from measured `results/` JSON only — no hand-typed metrics.
