import subprocess
import sys

import numpy as np
import pytest

from genomorph.adapters import MockBackend, VariantEffectBackend, get_backend


def test_mock_backend_deterministic():
    b = MockBackend()
    e1 = b.predict("chr1_1000_A_T")
    e2 = b.predict("chr1_1000_A_T")
    for t1, t2 in zip(e1.tracks, e2.tracks):
        assert np.array_equal(t1.ref, t2.ref)
        assert np.array_equal(t1.alt, t2.alt)


def test_mock_backend_distinct_variants_differ():
    b = MockBackend()
    e1 = b.predict("chr1_1000_A_T")
    e2 = b.predict("chr2_5000_G_C")
    assert not np.array_equal(e1.tracks[0].alt, e2.tracks[0].alt)


def test_mock_satisfies_protocol():
    assert isinstance(MockBackend(), VariantEffectBackend)


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("not-a-backend")


def test_alphagenome_disclaimer_constant_present():
    # Import the module without constructing the client (no SDK / key needed).
    from genomorph.adapters import alphagenome

    assert "NON-COMMERCIAL" in alphagenome.DISCLAIMER
    assert "train other ML models" in alphagenome.DISCLAIMER


def test_alphagenome_construction_warns_noncommercial():
    pytest.importorskip("numpy")
    from genomorph.adapters.alphagenome import AlphaGenomeBackend

    with pytest.warns(UserWarning, match="NON-COMMERCIAL"):
        AlphaGenomeBackend()


def test_cli_version():
    out = subprocess.run(
        [sys.executable, "-m", "genomorph.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert "genomorph 0.1.0a1" in out.stdout


def test_cli_fingerprint_mock():
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "genomorph.cli",
            "fingerprint",
            "chr1_108004887_G_T",
            "--backend",
            "mock",
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0
    assert "fingerprint" in out.stdout
    assert "RNA:w1_shape" in out.stdout
