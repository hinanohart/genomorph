"""AlphaGenome backend — OPT-IN ONLY (non-commercial).

AlphaGenome's model and outputs are licensed CC-BY-NC and its Terms of Use
state that outputs must not be used to train other ML models. genomorph
therefore treats AlphaGenome strictly as an opt-in source:

* a non-commercial disclaimer is emitted whenever this backend is constructed;
* the user must supply their own API key via the ALPHAGENOME_API_KEY
  environment variable (genomorph never stores, logs, or transmits the key
  itself; it is read from the user's environment and handed to the official
  client);
* AlphaGenome outputs are used only to compute a fingerprint for the requested
  variant and are NEVER bundled, cached to a shared atlas, redistributed, or
  used to train anything.

If you need a redistributable / commercial-friendly default, use the Borzoi
(MIT) or Enformer (Apache-2.0) backends instead.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from ..types import TrackProfile, VariantEffect

__all__ = ["AlphaGenomeBackend", "DISCLAIMER"]

DISCLAIMER = (
    "AlphaGenome backend: outputs are CC-BY-NC (NON-COMMERCIAL) and per its "
    "Terms of Use must not be used to train other ML models. genomorph uses "
    "them only to fingerprint the requested variant and never bundles, "
    "redistributes, or trains on them. Use Borzoi (MIT) or Enformer (Apache-2.0) "
    "for commercial or redistributable workflows."
)

_DEFAULT_TRACK_GROUPS = {
    "RNA": ["RNA_SEQ"],
    "DNASE": ["DNASE"],
    "CAGE": ["CAGE"],
}


class AlphaGenomeBackend:
    name = "alphagenome"

    def __init__(
        self,
        track_groups: dict[str, list[str]] | None = None,
        api_key_env: str = "ALPHAGENOME_API_KEY",
        emit_disclaimer: bool = True,
    ):
        if emit_disclaimer:
            warnings.warn(DISCLAIMER, stacklevel=2)
        self.track_groups = dict(track_groups or _DEFAULT_TRACK_GROUPS)
        self.api_key_env = api_key_env
        self._client = None

    @property
    def modalities(self) -> list[str]:
        return list(self.track_groups)

    def _lazy(self):
        if self._client is None:
            from alphagenome.models import dna_client

            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(
                    f"AlphaGenome requires an API key in ${self.api_key_env}. "
                    "genomorph reads it from your environment and never stores it."
                )
            self._client = dna_client.create(key)
        return self._client

    def predict(self, variant_id: str) -> VariantEffect:  # pragma: no cover - opt-in
        client = self._lazy()
        parts = variant_id.split("_")
        chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2], parts[3]
        # The official client returns per-track REF/ALT profiles for the variant;
        # the exact call surface is version-specific. We group returned tracks
        # into modalities and build TrackProfiles without persisting outputs.
        from alphagenome.data import genome

        variant = genome.Variant(
            chromosome=chrom, position=pos, reference_bases=ref, alternate_bases=alt
        )
        pred = client.predict_variant(variant=variant)
        tracks = []
        for modality, names in self.track_groups.items():
            ref_prof, alt_prof, bin_size = _gather(pred, names)
            tracks.append(
                TrackProfile(
                    modality=modality,
                    bin_size=bin_size,
                    ref=np.clip(ref_prof, 0.0, None),
                    alt=np.clip(alt_prof, 0.0, None),
                )
            )
        return VariantEffect(variant_id=variant_id, tracks=tracks)


def _gather(pred, names):  # pragma: no cover - opt-in, version-specific
    """Aggregate named AlphaGenome output tracks into REF/ALT profiles."""
    ref_stack, alt_stack, bin_size = [], [], 1
    for name in names:
        out = getattr(pred, name.lower(), None)
        if out is None:
            continue
        ref_stack.append(np.asarray(out.reference).mean(axis=-1))
        alt_stack.append(np.asarray(out.alternate).mean(axis=-1))
        bin_size = int(getattr(out, "resolution", bin_size))
    if not ref_stack:
        raise RuntimeError("no requested AlphaGenome tracks were returned")
    return (
        np.mean(ref_stack, axis=0),
        np.mean(alt_stack, axis=0),
        bin_size,
    )
