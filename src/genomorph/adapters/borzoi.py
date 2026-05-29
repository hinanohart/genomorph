"""Borzoi backend (optional, first-class default; weights MIT).

Requires the ``borzoi`` extra (``torch``, ``borzoi-pytorch``) and a local hg38
FASTA. Weights are downloaded from the Hugging Face Hub (``johahi/borzoi-*``,
MIT) at first use and are never bundled by genomorph. Not exercised in CI
(heavy / multi-GB).

Borzoi takes a 524,288 bp sequence and predicts ~7,600 human tracks over 16,384
central bins (32 bp each, after cropping). genomorph crops a window around the
variant and groups a user-chosen set of track indices into modalities. Track
indices depend on the model release; defaults below are placeholders the user
should map to their target track set (see Borzoi's targets file).
"""

from __future__ import annotations

import numpy as np

from ..types import TrackProfile, VariantEffect

__all__ = ["BorzoiBackend"]

_SEQ_LEN = 524_288
_BIN_SIZE = 32  # bp per output bin after cropping

_DEFAULT_TRACK_GROUPS = {
    "RNA": [0, 1, 2],
    "DNASE": [100, 101, 102],
    "CAGE": [200, 201, 202],
}


class BorzoiBackend:
    name = "borzoi"

    def __init__(
        self,
        genome_fasta: str,
        model_id: str = "johahi/borzoi-replicate-0",
        track_groups: dict[str, list[int]] | None = None,
        window_bins: int = 256,
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
            from borzoi_pytorch import Borzoi

            self._model = Borzoi.from_pretrained(self.model_id).eval()
        if self._fasta is None:
            import pyfaidx

            self._fasta = pyfaidx.Fasta(self.genome_fasta)
        return self._model, self._fasta

    @staticmethod
    def _one_hot_cl(seq: str) -> np.ndarray:
        """One-hot in (4, L) channel-first layout expected by Borzoi."""
        lut = {"A": 0, "C": 1, "G": 2, "T": 3}
        oh = np.zeros((4, len(seq)), dtype=np.float32)
        for i, ch in enumerate(seq.upper()):
            j = lut.get(ch)
            if j is not None:
                oh[j, i] = 1.0
        return oh

    def predict(self, variant_id: str) -> VariantEffect:
        import torch

        model, fasta = self._lazy()
        parts = variant_id.split("_")
        chrom, pos, alt = parts[0], int(parts[1]), parts[3]
        half = _SEQ_LEN // 2
        start = pos - half
        seq = list(str(fasta[chrom][start : start + _SEQ_LEN]).upper())
        alt_seq = list(seq)
        alt_seq[half] = alt[0]

        with torch.no_grad():
            ref_in = torch.from_numpy(self._one_hot_cl("".join(seq)))[None]
            alt_in = torch.from_numpy(self._one_hot_cl("".join(alt_seq)))[None]
            out_ref = model(ref_in)[0].detach().cpu().numpy()
            out_alt = model(alt_in)[0].detach().cpu().numpy()
        # out_* shape: (tracks, bins)
        n_bins = out_ref.shape[1]
        c = n_bins // 2
        w = self.window_bins // 2
        sl = slice(c - w, c + w)
        tracks = []
        for modality, idxs in self.track_groups.items():
            ref_prof = out_ref[idxs][:, sl].mean(axis=0)
            alt_prof = out_alt[idxs][:, sl].mean(axis=0)
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
