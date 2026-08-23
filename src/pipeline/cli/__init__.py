"""Command-line entry points.

:mod:`pipeline.runtime` is applied here rather than in each module, because the
OpenMP fix has to land before torch loads its DLLs and these are the processes
that load torch.
"""

from ..runtime import configure_openmp

configure_openmp()
