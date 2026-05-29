"""Variant-effect backends.

The mock backend is always available. Heavy model backends are imported lazily
so that ``import genomorph`` and the synthetic benchmark never require torch,
model weights, or a reference genome.
"""

from __future__ import annotations

from .base import VariantEffectBackend
from .mock import MockBackend

__all__ = ["VariantEffectBackend", "MockBackend", "get_backend"]


def get_backend(name: str, **kwargs) -> VariantEffectBackend:
    """Construct a backend by name, importing heavy deps only when requested."""
    name = name.lower()
    if name == "mock":
        return MockBackend(**kwargs)
    if name == "borzoi":
        from .borzoi import BorzoiBackend

        return BorzoiBackend(**kwargs)
    if name == "enformer":
        from .enformer import EnformerBackend

        return EnformerBackend(**kwargs)
    if name == "alphagenome":
        from .alphagenome import AlphaGenomeBackend

        return AlphaGenomeBackend(**kwargs)
    raise ValueError(
        f"unknown backend {name!r}; choices: mock, borzoi, enformer, alphagenome"
    )
