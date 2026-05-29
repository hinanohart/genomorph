"""The backend contract.

genomorph never predicts variant effects itself; it consumes the REF/ALT track
profiles a sequence-to-function model produces and fingerprints their shape.
Any object satisfying :class:`VariantEffectBackend` can be plugged in.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import VariantEffect

__all__ = ["VariantEffectBackend"]


@runtime_checkable
class VariantEffectBackend(Protocol):
    """Produce a :class:`~genomorph.types.VariantEffect` for a variant id.

    ``variant_id`` is ``chr_pos_ref_alt`` on the backend's assembly (e.g.
    ``chr1_108004887_G_T`` for hg38), matching the eQTL Catalogue convention.
    Implementations download their own weights at first use and must never
    bundle or redistribute them through genomorph.
    """

    name: str

    def predict(self, variant_id: str) -> VariantEffect: ...

    @property
    def modalities(self) -> list[str]: ...
