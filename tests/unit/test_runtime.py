"""The OpenMP workaround is verified numerically, not trusted.

Intel documents ``KMP_DUPLICATE_LIB_OK`` as unsupported and warns it "may
silently produce incorrect results". Silently producing incorrect results is the
one failure mode this project cannot tolerate -- every reported metric flows
through the MKL matmul and eigendecomposition that the flag affects. So the
claim is tested rather than assumed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from pipeline.runtime import OPENMP_ENV, configure_openmp

# Exactly the kernel the PCA path runs: a centred covariance followed by a
# symmetric eigendecomposition, both MKL-backed.
KERNEL = textwrap.dedent(
    """
    import numpy as np, hashlib
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20000, 60)) + np.linspace(0.2, 0.6, 60)
    c = a - a.mean(axis=0)
    cov = (c.T @ c) / (a.shape[0] - 1)
    w, v = np.linalg.eigh(cov)
    print(hashlib.sha256(cov.tobytes()).hexdigest())
    print(hashlib.sha256(w.tobytes()).hexdigest())
    print(hashlib.sha256(np.abs(v).tobytes()).hexdigest())
    """
)


def _run(source: str, env: dict) -> str:
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip()


def test_configure_openmp_is_idempotent_and_non_clobbering(monkeypatch):
    for key in OPENMP_ENV:
        monkeypatch.setenv(key, "user-set")
    configure_openmp()
    configure_openmp()
    for key in OPENMP_ENV:
        assert os.environ[key] == "user-set"


@pytest.mark.slow
def test_duplicate_openmp_runtime_does_not_change_the_numbers():
    pytest.importorskip("torch", reason="deep path is optional (PRD §10)")

    clean = {k: v for k, v in os.environ.items() if k not in OPENMP_ENV}
    reference = _run(KERNEL, clean)

    workaround = _run("import torch\n" + KERNEL, dict(os.environ, **OPENMP_ENV))

    assert workaround == reference, (
        "KMP_DUPLICATE_LIB_OK changed an MKL result. The workaround in "
        "pipeline.runtime is no longer safe -- switch to a single OpenMP runtime "
        "before trusting any metric from this machine."
    )


@pytest.mark.slow
def test_the_workaround_is_actually_load_bearing():
    # If this ever fails, the environment was fixed properly and
    # pipeline.runtime can be deleted rather than carried around.
    pytest.importorskip("torch", reason="deep path is optional (PRD §10)")

    clean = {k: v for k, v in os.environ.items() if k not in OPENMP_ENV}
    result = subprocess.run(
        [sys.executable, "-c", "import torch\n" + KERNEL],
        capture_output=True,
        text=True,
        env=clean,
    )
    if result.returncode == 0:
        pytest.skip(
            "torch and NumPy now share one OpenMP runtime; pipeline.runtime is "
            "no longer needed on this machine."
        )
    assert "OMP" in result.stderr or result.returncode != 0
