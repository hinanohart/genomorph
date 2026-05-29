"""Perceptual (Oklab) visualisation of mechanism fingerprints.

Mechanism clusters are projected to 2-D and coloured in Oklab, a perceptually
uniform colour space (Björn Ottosson), so that equal visual distances reflect
roughly equal fingerprint distances. matplotlib is an optional dependency
(``viz`` extra); the colour maths is pure numpy and always importable.
"""

from __future__ import annotations

import numpy as np

__all__ = ["oklab_to_srgb", "cluster_palette", "plot_fingerprints"]

# linear-sRGB <-> Oklab matrices (Ottosson 2020)
_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_M2_INV = np.linalg.inv(_M2)
_M1_INV = np.linalg.inv(_M1)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert Oklab (L, a, b) rows to sRGB in [0, 1]."""
    lab = np.atleast_2d(np.asarray(lab, dtype=np.float64))
    lms = lab @ _M2_INV.T
    lms_cubed = lms**3
    rgb_lin = lms_cubed @ _M1_INV.T
    return _linear_to_srgb(rgb_lin)


def cluster_palette(
    n: int, *, lightness: float = 0.7, chroma: float = 0.12
) -> np.ndarray:
    """``n`` perceptually even sRGB colours by sweeping hue at fixed L, C in Oklab."""
    hues = np.linspace(0, 2 * np.pi, n, endpoint=False)
    lab = np.column_stack(
        [np.full(n, lightness), chroma * np.cos(hues), chroma * np.sin(hues)]
    )
    return np.clip(oklab_to_srgb(lab), 0.0, 1.0)


def plot_fingerprints(
    x: np.ndarray, labels: np.ndarray, *, title: str = "genomorph", ax=None
):
    """Scatter MAD-scaled fingerprints in 2-D (PCA), coloured per cluster in Oklab.

    Returns the matplotlib Axes. Requires the ``viz`` extra.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels)
    coords = PCA(n_components=2, random_state=0).fit_transform(x)
    uniq = np.unique(labels)
    palette = cluster_palette(len(uniq))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    for color, lab in zip(palette, uniq):
        m = labels == lab
        ax.scatter(coords[m, 0], coords[m, 1], s=18, color=color, label=str(lab))
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="mechanism", fontsize=8)
    return ax
