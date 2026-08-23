"""Runs before any test module is imported, which is the point.

``configure_openmp`` must set its environment variable before torch loads its
bundled OpenMP DLL; by the time a test module's ``import torch`` executes it is
already too late. See :mod:`pipeline.runtime`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()
