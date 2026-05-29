"""Topological component of the fingerprint — DEFERRED to v0.3.

Persistent homology of contact-map difference tracks (via ripser/persim, MIT)
is planned as a sparse fingerprint component for contact/Hi-C modalities only.
It is intentionally NOT part of the v0.1.0a core CLAIM, which rests on the
1-D optimal-transport features. This module is a placeholder so the public
surface and roadmap are explicit; calling it raises rather than silently
returning a fake feature.
"""

from __future__ import annotations

import numpy as np

__all__ = ["contact_persistence_features"]


def contact_persistence_features(contact_delta: np.ndarray) -> np.ndarray:
    """Planned for v0.3; not implemented in v0.1.0a."""
    del contact_delta  # interface placeholder; no v0.1.0a implementation
    raise NotImplementedError(
        "TDA (contact-map persistent homology) is a v0.3 backlog feature; "
        "the v0.1.0a fingerprint uses only the optimal-transport components."
    )
