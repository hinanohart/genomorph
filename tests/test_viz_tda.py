import numpy as np
import pytest

from genomorph import viz
from genomorph.tda import contact_persistence_features


def test_oklab_to_srgb_in_gamut():
    rgb = viz.oklab_to_srgb([[0.7, 0.0, 0.0]])
    assert rgb.shape == (1, 3)
    assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)


def test_palette_distinct_colors():
    pal = viz.cluster_palette(4)
    assert pal.shape == (4, 3)
    # all four colours differ
    assert len({tuple(np.round(c, 3)) for c in pal}) == 4


def test_oklab_roundtrip_grayscale():
    # Oklab L with zero a,b is achromatic -> equal-ish RGB channels.
    rgb = viz.oklab_to_srgb([[0.5, 0.0, 0.0]])[0]
    assert rgb.std() < 1e-3


def test_tda_is_v03_not_implemented():
    with pytest.raises(NotImplementedError):
        contact_persistence_features(np.zeros((4, 4)))
