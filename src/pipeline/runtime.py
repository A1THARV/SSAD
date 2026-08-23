"""Process-level setup that must happen before NumPy or torch load their DLLs.

Exists for one concrete failure. On this Windows/Anaconda machine two different
builds of the Intel OpenMP runtime are installed:

* ``anaconda3/Library/bin/libiomp5md.dll``        (2023-01-18, shipped with MKL)
* ``anaconda3/Lib/site-packages/torch/lib/...``   (2024-03-20, shipped with torch)

Importing torch and then calling any MKL-backed NumPy routine -- ``a.T @ a``,
``np.linalg.eigh`` -- aborts the interpreter with ``OMP: Error #15``. That is
every line of the PCA path, so without this the deep branch and the classical
branch cannot run in one process.
"""

from __future__ import annotations

import os

__all__ = ["configure_openmp", "OPENMP_ENV"]


# Intel documents KMP_DUPLICATE_LIB_OK as unsupported and warns it "may silently
# produce incorrect results". That warning is taken seriously rather than waved
# through: tests/unit/test_runtime.py runs the corpus-covariance kernel in two
# subprocesses -- one with torch loaded under this flag, one with NumPy alone --
# and asserts the covariance and its eigenvalues are *bit*-identical. The
# workaround stays only as long as that test passes.
OPENMP_ENV = {"KMP_DUPLICATE_LIB_OK": "TRUE"}


def configure_openmp() -> None:
    """Permit the duplicate OpenMP runtime. Call before importing torch.

    Idempotent, and never overrides a value the caller already set -- someone
    debugging a threading problem may have set it deliberately.
    """
    for key, value in OPENMP_ENV.items():
        os.environ.setdefault(key, value)
