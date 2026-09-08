"""Calibration, clipped-pixel handling and scaling for thermal frames (FR-9.7).

Three jobs, in the order they must happen:

1. **Digital numbers to temperature**, where the file did not already carry it.
   FLAME 3 ships degrees Celsius directly, so this path exists for datasets that
   do not -- and it refuses to run on data that is already calibrated, because
   applying a gain twice is silent and irreversible.
2. **Clipped pixels**, which FLAME 3 has in quantity: roughly 2% of a typical
   frame sits at exactly the sensor's cold floor. Those pixels are real
   measurements that saturated, so they carry no gradient and must not be read
   as structure.
3. **Scaling for a model**, which is where the interesting mistake lives -- see
   :func:`normalise`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ThermalPreprocessConfig
from ..errors import DegenerateInputError, IncompatibleInputError
from .loader import ThermalFrame

__all__ = [
    "PHYSICAL_SCALE_C",
    "PHYSICAL_REFERENCE_C",
    "ClipReport",
    "detect_clipped",
    "to_temperature",
    "normalise",
]

#: Reference point and span for physical normalisation, in degrees Celsius.
#: Chosen to bracket FLAME 3's observed range (-40.6 to 608.7 C) with margin,
#: and fixed as constants rather than derived per dataset so that two runs are
#: comparable and a model trained on one scene can be applied to another.
PHYSICAL_REFERENCE_C = 0.0
PHYSICAL_SCALE_C = 500.0


@dataclass(frozen=True)
class ClipReport:
    """Where a frame hit the sensor's limits."""

    low_value: float
    high_value: float
    n_low: int
    n_high: int
    n_pixels: int

    @property
    def fraction_clipped(self) -> float:
        return (self.n_low + self.n_high) / self.n_pixels

    def as_labels(self) -> dict[str, str]:
        return {
            "clipped_fraction": f"{self.fraction_clipped:.4f}",
            "clipped_low_value_c": f"{self.low_value:.2f}",
        }


def detect_clipped(
    data: np.ndarray, tolerance: float = 1e-3
) -> ClipReport:
    """Find pixels sitting exactly at the frame's extremes.

    A saturated pixel is a real measurement that ran out of range, so its value
    is a bound rather than a reading. FLAME 3's cold floor is the common case --
    sky against a 640x512 sensor puts ~2% of a typical frame at exactly the same
    negative value, repeated bit-for-bit across frames and across both splits,
    which is the signature of a clip rather than of physics.

    Left as a report rather than a repair. Masking or interpolating those pixels
    is a modelling choice that belongs to whatever consumes the frame, and doing
    it here would hide how much of the frame was affected.
    """
    if data.ndim != 2:
        raise IncompatibleInputError(f"Expected a 2-D frame, got shape {data.shape}.")
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise DegenerateInputError("Frame holds no finite values.")

    lo, hi = float(finite.min()), float(finite.max())
    return ClipReport(
        low_value=lo,
        high_value=hi,
        n_low=int(np.count_nonzero(np.abs(data - lo) <= tolerance)),
        n_high=int(np.count_nonzero(np.abs(data - hi) <= tolerance)),
        n_pixels=int(data.size),
    )


def to_temperature(frame: ThermalFrame, cfg: ThermalPreprocessConfig) -> np.ndarray:
    """Apply a documented gain and offset to digital numbers.

    Refuses to touch a frame that already carries temperature. Applying a gain
    to calibrated degrees produces a plausible-looking array with no error
    anywhere downstream, which is precisely the failure this module exists to
    make impossible.
    """
    if frame.radiometry.units in {"celsius", "kelvin"}:
        raise IncompatibleInputError(
            f"{frame.name or 'frame'} already carries {frame.radiometry.units} "
            "temperature, so a gain/offset must not be applied to it -- doing so "
            "would silently rescale real degrees. Calibration is for datasets whose "
            "files hold raw counts."
        )
    if cfg.gain == 0:
        raise DegenerateInputError("A gain of zero would erase the frame.")
    return frame.data * cfg.gain + cfg.offset


def normalise(
    data: np.ndarray,
    mode: str = "physical",
    reference: float = PHYSICAL_REFERENCE_C,
    scale: float = PHYSICAL_SCALE_C,
) -> np.ndarray:
    """Scale a temperature field for a network.

    ``physical`` maps ``(T - reference) / scale`` with **fixed constants**.
    ``per_frame`` min-max scales each frame onto ``[0, 1]`` independently.

    **The default is not a preference.** On this data, per-frame normalisation
    destroys the signal outright. FLAME 3's fire and no-fire frames are
    separated entirely by absolute temperature -- fire frames peak at 108-609 C,
    no-fire frames at 7-47 C, with no overlap -- and min-max scaling maps the
    hottest pixel of *both* to exactly 1.0. A model given per-frame normalised
    input cannot see the one feature that distinguishes the classes, and would
    be reduced to inferring fire from texture.

    That is the classic form of this bug: the preprocessing looks reasonable,
    the model trains without complaint, and the accuracy loss is blamed on the
    architecture. ``per_frame`` is kept only so the comparison can be measured
    and so shape-only experiments remain expressible.

    Args:
        data: Temperature field in degrees Celsius.
        mode: ``physical`` or ``per_frame``.
        reference / scale: Constants for ``physical`` mode.
    """
    arr = np.asarray(data, dtype=np.float32)
    if not np.any(np.isfinite(arr)):
        raise DegenerateInputError("Cannot normalise a frame with no finite values.")

    if mode == "physical":
        if scale == 0:
            raise DegenerateInputError("Physical normalisation needs a non-zero scale.")
        return (arr - reference) / scale

    if mode == "per_frame":
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi - lo <= 0:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    raise DegenerateInputError(
        f"Unknown normalisation mode {mode!r} (expected 'physical' or 'per_frame')."
    )
