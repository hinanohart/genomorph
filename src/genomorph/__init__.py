"""genomorph: optimal-transport fingerprints of regulatory-variant mechanisms.

genomorph decomposes a sequence-to-function model's REF->ALT per-track
difference into a *shape* component (1-D Wasserstein distance between the
mass-normalised profiles) and a signed *mass* component, producing a fixed-
dimension fingerprint that separates regulatory mechanisms a single effect-size
scalar conflates. It is backend-agnostic (Borzoi/Enformer/AlphaGenome or any
custom backend) and CPU-only at its core.
"""

from __future__ import annotations

from .adapters import MockBackend, VariantEffectBackend, get_backend
from .cluster import ari_bootstrap_ci, cluster_labels, paired_ari_diff_ci
from .fingerprint import FingerprintExtractor, MADScaler
from .ot import wasserstein1d
from .types import TrackProfile, VariantEffect

__version__ = "0.1.0a1"

__all__ = [
    "__version__",
    "VariantEffect",
    "TrackProfile",
    "FingerprintExtractor",
    "MADScaler",
    "wasserstein1d",
    "cluster_labels",
    "ari_bootstrap_ci",
    "paired_ari_diff_ci",
    "VariantEffectBackend",
    "MockBackend",
    "get_backend",
]
