"""A deterministic, dependency-free backend for tests and demos.

The mock backend turns a variant id into a reproducible synthetic effect by
seeding a generator from a hash of the id. It carries no biological meaning --
it exists so the end-to-end pipeline (predict -> fingerprint -> cluster) can be
exercised in CI without downloading multi-gigabyte weights or a reference
genome. It is NOT a variant-effect predictor.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..types import TrackProfile, VariantEffect

__all__ = ["MockBackend"]

_DEFAULT_MODALITIES = ("RNA", "DNASE", "H3K27ac")


class MockBackend:
    name = "mock"

    def __init__(
        self,
        modalities: tuple[str, ...] = _DEFAULT_MODALITIES,
        n_bins: int = 128,
        bin_size: int = 128,
    ):
        self._modalities = tuple(modalities)
        self.n_bins = n_bins
        self.bin_size = bin_size

    @property
    def modalities(self) -> list[str]:
        return list(self._modalities)

    def _seed(self, variant_id: str) -> int:
        h = hashlib.sha256(variant_id.encode()).digest()
        return int.from_bytes(h[:8], "little")

    def predict(self, variant_id: str) -> VariantEffect:
        rng = np.random.default_rng(self._seed(variant_id))
        coords = np.arange(self.n_bins)
        tracks = []
        for m in self._modalities:
            centre = rng.uniform(0.3, 0.7) * self.n_bins
            width = rng.uniform(3, 8)
            amp = rng.uniform(1.0, 4.0)
            ref = amp * np.exp(-0.5 * ((coords - centre) / width) ** 2) + 0.05
            # a small reproducible perturbation between REF and ALT
            shift = rng.normal(0, 2.0)
            scale = rng.uniform(0.7, 1.3)
            alt = (
                scale * amp * np.exp(-0.5 * ((coords - centre - shift) / width) ** 2)
                + 0.05
            )
            tracks.append(
                TrackProfile(
                    modality=m,
                    bin_size=self.bin_size,
                    ref=ref,
                    alt=alt,
                    signed=False,
                )
            )
        return VariantEffect(variant_id=variant_id, tracks=tracks)
