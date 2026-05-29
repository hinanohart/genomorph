"""Core data structures shared across genomorph.

A regulatory variant's predicted effect is, per modality, a pair of coverage
profiles (REF and ALT) over a window of genomic bins. genomorph treats the
*difference between these profiles* as a (possibly signed) measure on genomic
coordinates and fingerprints its *shape* and *mass*, rather than collapsing it
to a single scalar effect size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["TrackProfile", "VariantEffect"]


@dataclass
class TrackProfile:
    """REF and ALT coverage for a single assay/modality over a binned window.

    Parameters
    ----------
    modality:
        Free-form modality name, e.g. ``"RNA"``, ``"DNASE"``, ``"CAGE"``,
        ``"H3K27ac"``, ``"splice"``, ``"contact"``.
    bin_size:
        Base pairs represented by each bin. Used to put profiles from
        different-resolution assays onto a common bp ground metric.
    ref, alt:
        1-D arrays of equal length giving predicted coverage at REF and ALT.
    signed:
        Whether the assay can take negative values (e.g. log fold-change or
        contact-difference tracks). Signed tracks are handled via a Jordan
        decomposition (see :mod:`genomorph.measures`); unsigned tracks are
        non-negative coverage.
    start_bp:
        Genomic coordinate (bp) of the first bin's left edge. Optional; only
        affects the absolute scale of the ground metric, not relative shape.
    """

    modality: str
    bin_size: int
    ref: np.ndarray
    alt: np.ndarray
    signed: bool = False
    start_bp: int = 0

    def __post_init__(self) -> None:
        self.ref = np.asarray(self.ref, dtype=np.float64)
        self.alt = np.asarray(self.alt, dtype=np.float64)
        if self.ref.ndim != 1 or self.alt.ndim != 1:
            raise ValueError(f"{self.modality}: ref/alt must be 1-D")
        if self.ref.shape != self.alt.shape:
            raise ValueError(
                f"{self.modality}: ref {self.ref.shape} != alt {self.alt.shape}"
            )
        if self.bin_size <= 0:
            raise ValueError(f"{self.modality}: bin_size must be positive")
        if not self.signed and (np.any(self.ref < 0) or np.any(self.alt < 0)):
            raise ValueError(
                f"{self.modality}: negative values in an unsigned track; "
                "set signed=True for log-fold-change/contact tracks"
            )

    @property
    def n_bins(self) -> int:
        return int(self.ref.shape[0])

    @property
    def coords_bp(self) -> np.ndarray:
        """Bin-centre coordinates in bp (the OT ground metric support)."""
        return self.start_bp + (np.arange(self.n_bins) + 0.5) * self.bin_size

    @property
    def delta(self) -> np.ndarray:
        """Signed ALT-minus-REF profile."""
        return self.alt - self.ref


@dataclass
class VariantEffect:
    """A variant's multi-modality predicted effect, as produced by a backend."""

    variant_id: str
    tracks: list[TrackProfile] = field(default_factory=list)

    def modality(self, name: str) -> TrackProfile:
        for t in self.tracks:
            if t.modality == name:
                return t
        raise KeyError(f"modality {name!r} not present in {self.variant_id}")

    @property
    def modalities(self) -> list[str]:
        return [t.modality for t in self.tracks]
