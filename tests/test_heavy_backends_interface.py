"""Interface-only tests for heavy backends.

These exercise the static surface (construction, modality mapping, one-hot
encoding) WITHOUT importing torch, model weights, or a reference genome -- those
paths are integration-tested by users with the relevant extras installed and
are intentionally not run in CI.
"""

import numpy as np

from genomorph.adapters.base import VariantEffectBackend
from genomorph.adapters.borzoi import BorzoiBackend
from genomorph.adapters.enformer import EnformerBackend


def test_borzoi_interface_without_weights():
    b = BorzoiBackend(genome_fasta="/nonexistent.fa", track_groups={"RNA": [0, 1]})
    assert b.name == "borzoi"
    assert b.modalities == ["RNA"]
    assert isinstance(b, VariantEffectBackend)


def test_enformer_interface_without_weights():
    e = EnformerBackend(genome_fasta="/nonexistent.fa")
    assert e.name == "enformer"
    assert set(e.modalities) == {"DNASE", "CAGE", "H3K27ac"}
    assert isinstance(e, VariantEffectBackend)


def test_borzoi_one_hot_channel_first():
    oh = BorzoiBackend._one_hot_cl("ACGTN")
    assert oh.shape == (4, 5)
    assert np.array_equal(oh[:, 0], [1, 0, 0, 0])  # A
    assert np.array_equal(oh[:, 3], [0, 0, 0, 1])  # T
    assert np.array_equal(oh[:, 4], [0, 0, 0, 0])  # N -> all zero


def test_enformer_one_hot_position_major():
    e = EnformerBackend(genome_fasta="/nonexistent.fa")
    oh = e._one_hot("ACGT")
    assert oh.shape == (4, 4)
    assert np.array_equal(oh[0], [1, 0, 0, 0])
    assert np.array_equal(oh[2], [0, 0, 1, 0])  # G
