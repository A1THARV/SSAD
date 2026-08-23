"""Typed exception hierarchy for the pipeline.

PRD §10 ("No silent failures"): any module that cannot produce a valid result
must raise a clear, typed exception -- never return zeros or NaNs disguised as
a valid score. Every "must raise a clear error" requirement in the PRD resolves
to one of the types below, so call sites raise an existing type rather than
inventing one.

The hierarchy distinguishes two situations that look similar but must be
handled differently:

* **Refuse** -- the requested computation cannot be made meaningful, so raise
  (non-radiometric input to Planck, unregistered input to pixel-level fusion,
  a class map where a binary mask is required).
* **Degrade gracefully** -- a documented fallback exists, so return a flagged
  result instead of raising (SAM low confidence, sequence shorter than the
  LSTM minimum, unknown emissivity, a scene with zero anomalies).

Only the first group appears here. The second group is expressed as flags and
fallbacks in the modules themselves, which is why there is no
``LowConfidenceError`` or ``ShortSequenceError``.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "PipelineError",
    "ConfigError",
    "DataUnavailableError",
    "DegenerateInputError",
    "IncompatibleInputError",
    "MissingGroundTruthError",
    "NonRadiometricInputError",
    "UnregisteredInputError",
    "ModelNotFittedError",
]


class PipelineError(Exception):
    """Base class for every error this pipeline raises deliberately.

    Catching ``PipelineError`` catches all actionable, expected failures while
    still letting genuine bugs (``KeyError``, ``IndexError``, ...) surface.
    """


class ConfigError(PipelineError):
    """A configuration file is malformed, incomplete, or contains unknown keys.

    Unknown keys are an error rather than a warning: a silently ignored typo in
    a config is a reproducibility hazard (PRD §10).
    """


class DataUnavailableError(PipelineError):
    """A required data file is missing, truncated, or fails its checksum.

    PRD §11 edge case 14 requires pipeline scripts to fail fast with an
    actionable message naming the expected path and how to obtain the file,
    rather than a stack trace from deep inside ``scipy.io.loadmat``. That
    obligation is carried by the type rather than by each call site's prose:
    ``path`` and ``remedy`` are required arguments, so an unhelpful message is
    not constructible.

    Attributes:
        reason: What is wrong, in one clause.
        path: Where the file was expected.
        remedy: The concrete next action -- a URL, a script to run, a command.
    """

    def __init__(self, reason: str, path: str | Path, remedy: str) -> None:
        self.reason = reason
        self.path = Path(path)
        self.remedy = remedy
        super().__init__(f"{reason}\n  expected at: {self.path}\n  to fix     : {remedy}")


class DegenerateInputError(PipelineError):
    """One input is structurally valid but cannot yield a meaningful result.

    Examples: a scene with a single distinct spectrum, an all-NaN score map, a
    class map supplied where a binary mask is required, a sequence with no
    usable frames.

    Contrast :class:`IncompatibleInputError`, which is about a *pair* of inputs
    that are each individually fine.
    """


class IncompatibleInputError(PipelineError):
    """Two inputs are each valid but cannot be combined.

    The data is fine; the pairing is not. Covers:

    * a query spectrum and a library spectrum on different wavelength grids
      (spectral libraries ship on their own grids, not the sensor's, so
      resampling must happen before :func:`spectral_angle` is called);
    * a band-count mismatch between scenes (§11 edge case 5 -- ABU alone spans
      seven band counts from 102 to 207);
    * cubes with mismatched spatial dimensions handed to fusion.
    """


class MissingGroundTruthError(PipelineError):
    """Metrics were requested for a scene that carries no labels.

    ``HSICube.ground_truth`` is optional (PRD §9) while FR-9.11 computes
    AUC-ROC against it, so this case needs a decided home. It gets its own type
    rather than folding into :class:`DegenerateInputError` because a caller can
    legitimately recover from it: running an unlabelled scene should still
    produce an anomaly map and figures, just no metrics table. Catching
    ``DegenerateInputError`` to allow that would be far too broad a net.
    """


class NonRadiometricInputError(PipelineError):
    """Quantitative radiometry was requested on non-radiometric input.

    PRD FR-9.7 / §11 edge case 9 -- an 8-bit false-colour thermal image carries
    no true temperature values, so the Planck/emissivity module must refuse
    rather than compute a fabricated temperature.
    """


class UnregisteredInputError(PipelineError):
    """Pixel-level fusion was requested on inputs that are not co-registered.

    PRD FR-9.10 / §11 edge case 12 -- the honest default is independent mode;
    fabricating pixel alignment between a hyperspectral scene and an unrelated
    thermal sequence is not permitted.
    """


class ModelNotFittedError(PipelineError):
    """A transform was requested from an estimator that has not been fitted.

    PRD FR-9.3 -- PCA must be fit once and reused, so training and inference
    cannot silently refit on different data.
    """
