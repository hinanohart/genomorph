# pyright: reportMissingImports=false, reportPrivateImportUsage=false
# (optional heavy extras: torch / enformer-pytorch / pyfaidx are installed only
#  with the ``enformer`` extra; the lazy imports below are guarded by that extra.)
"""Enformer backend (optional, code+weights Apache-2.0).

Requires the ``enformer`` extra (``torch``, ``enformer-pytorch``) and a local
hg38 FASTA. Weights are downloaded from the Hugging Face Hub at first use and
are never bundled by genomorph. Not exercised in CI (heavy / multi-GB).

Enformer takes a 196,608 bp sequence and predicts 5,313 human tracks over 896
central bins (128 bp each). genomorph crops a window around the variant and
groups a user-chosen set of track indices into modalities.
"""

from __future__ import annotations

import numpy as np

from ..types import TrackProfile, VariantEffect

__all__ = ["EnformerBackend"]

_SEQ_LEN = 196_608
_BIN_SIZE = 128
_N_BINS = 896

# A small, documented default track selection (indices into Enformer's human
# head). Users should override with indices appropriate to their analysis.
_DEFAULT_TRACK_GROUPS = {
    "DNASE": [0, 1, 2],
    "CAGE": [4828, 4829, 4830],
    "H3K27ac": [2000, 2001, 2002],
}


class EnformerBackend:
    name = "enformer"

    def __init__(
        self,
        genome_fasta: str,
        model_id: str = "EleutherAI/enformer-official-rough",
        track_groups: dict[str, list[int]] | None = None,
        window_bins: int = 128,
        device: str = "cpu",
    ):
        self.genome_fasta = genome_fasta
        self.model_id = model_id
        self.track_groups = dict(track_groups or _DEFAULT_TRACK_GROUPS)
        self.window_bins = window_bins
        self.device = device
        self._model = None
        self._fasta = None

    @property
    def modalities(self) -> list[str]:
        return list(self.track_groups)

    def _lazy(self):
        if self._model is None:
            from enformer_pytorch import Enformer

            self._model = Enformer.from_pretrained(self.model_id).eval()
        if self._fasta is None:
            import pyfaidx

            self._fasta = pyfaidx.Fasta(self.genome_fasta)
        return self._model, self._fasta

    def _one_hot(self, seq: str) -> np.ndarray:
        lut = {"A": 0, "C": 1, "G": 2, "T": 3}
        oh = np.zeros((len(seq), 4), dtype=np.float32)
        for i, ch in enumerate(seq.upper()):
            j = lut.get(ch)
            if j is not None:
                oh[i, j] = 1.0
        return oh

    def predict(self, variant_id: str) -> VariantEffect:
        import torch

        model, fasta = self._lazy()
        parts = variant_id.split("_")
        chrom, pos, alt = parts[0], int(parts[1]), parts[3]
        half = _SEQ_LEN // 2
        start = pos - half
        seq = str(fasta[chrom][start : start + _SEQ_LEN]).upper()
        ref_seq = list(seq)
        alt_seq = list(seq)
        # place ALT allele at the variant position (SNV; first base for indels)
        alt_seq[half] = alt[0]
        out_ref = model(torch.tensor(self._one_hot("".join(ref_seq)))[None])["human"][0]
        out_alt = model(torch.tensor(self._one_hot("".join(alt_seq)))[None])["human"][0]
        out_ref = out_ref.detach().cpu().numpy()
        out_alt = out_alt.detach().cpu().numpy()

        c = _N_BINS // 2
        w = self.window_bins // 2
        sl = slice(c - w, c + w)
        tracks = []
        for modality, idxs in self.track_groups.items():
            ref_prof = out_ref[sl][:, idxs].mean(axis=1)
            alt_prof = out_alt[sl][:, idxs].mean(axis=1)
            tracks.append(
                TrackProfile(
                    modality=modality,
                    bin_size=_BIN_SIZE,
                    ref=np.clip(ref_prof, 0.0, None),
                    alt=np.clip(alt_prof, 0.0, None),
                    start_bp=start + (c - w) * _BIN_SIZE,
                )
            )
        return VariantEffect(variant_id=variant_id, tracks=tracks)
