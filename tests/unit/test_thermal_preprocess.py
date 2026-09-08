"""Thermal calibration, clipping and scaling."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import ThermalPreprocessConfig
from pipeline.errors import DegenerateInputError, IncompatibleInputError
from pipeline.thermal.loader import Radiometry, ThermalFrame
from pipeline.thermal.preprocess import (
    PHYSICAL_SCALE_C,
    detect_clipped,
    normalise,
    to_temperature,
)


def _frame(data, units="celsius", radiometric=True):
    return ThermalFrame(
        data=np.asarray(data, dtype=np.float32),
        radiometry=Radiometry(radiometric, units, "test"),
    )


# --------------------------------------------------------------------------
# Clipping
# --------------------------------------------------------------------------


def test_clipped_pixels_are_counted_at_both_ends():
    data = np.full((10, 10), 20.0, dtype=np.float32)
    data[0, :5] = -22.79
    data[1, :3] = 500.0
    report = detect_clipped(data)
    assert report.n_low == 5 and report.n_high == 3
    assert report.fraction_clipped == pytest.approx(8 / 100)


def test_clip_report_reaches_the_run_record():
    labels = detect_clipped(np.linspace(0, 50, 100).reshape(10, 10)).as_labels()
    assert "clipped_fraction" in labels and "clipped_low_value_c" in labels


def test_clipping_is_reported_not_repaired():
    # Masking or interpolating is the consumer's choice; doing it here would
    # hide how much of the frame was affected.
    data = np.full((4, 4), -40.56, dtype=np.float32)
    data[0, 0] = 100.0
    assert detect_clipped(data).n_low == 15
    assert data.min() == pytest.approx(-40.56), "input must not be modified"


def test_empty_frame_is_refused():
    with pytest.raises(DegenerateInputError):
        detect_clipped(np.full((4, 4), np.nan))


def test_non_2d_input_is_refused():
    with pytest.raises(IncompatibleInputError):
        detect_clipped(np.zeros((4, 4, 3)))


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_gain_and_offset_convert_counts_to_degrees():
    frame = _frame(np.full((2, 2), 1000.0), units="digital_number", radiometric=False)
    cfg = ThermalPreprocessConfig(gain=0.04, offset=-273.15)
    assert to_temperature(frame, cfg) == pytest.approx(1000 * 0.04 - 273.15)


def test_calibrating_already_calibrated_data_is_refused():
    # Applying a gain to real degrees produces a plausible array and no error
    # anywhere downstream -- the exact failure this module exists to prevent.
    with pytest.raises(IncompatibleInputError, match="already carries"):
        to_temperature(_frame(np.full((2, 2), 20.0)), ThermalPreprocessConfig(gain=0.04))


def test_zero_gain_is_refused():
    frame = _frame(np.full((2, 2), 1000.0), units="digital_number", radiometric=False)
    with pytest.raises(DegenerateInputError):
        to_temperature(frame, ThermalPreprocessConfig(gain=0.0))


# --------------------------------------------------------------------------
# Normalisation -- where the interesting mistake lives
# --------------------------------------------------------------------------


def test_physical_normalisation_preserves_absolute_temperature():
    hot = normalise(np.full((2, 2), 500.0))
    cold = normalise(np.full((2, 2), 25.0))
    assert hot.mean() > cold.mean()
    assert hot.mean() == pytest.approx(500.0 / PHYSICAL_SCALE_C)


def test_per_frame_normalisation_destroys_the_fire_signal():
    # The reason `physical` is the default. FLAME 3's classes are separated
    # entirely by absolute temperature; min-max scaling maps the hottest pixel
    # of a 600 C fire and a 25 C field to exactly the same value.
    fire = np.linspace(10.0, 600.0, 64).reshape(8, 8)
    calm = np.linspace(10.0, 25.0, 64).reshape(8, 8)

    assert normalise(fire, "per_frame").max() == pytest.approx(
        normalise(calm, "per_frame").max()
    )
    assert normalise(fire, "physical").max() > normalise(calm, "physical").max()


# Same 200 C pixel in two frames that differ only in how hot their hottest
# pixel is. Compared at the MIDDLE element: the extremes are pinned to 0 and 1
# by construction under per-frame scaling, so only an interior value can show
# whether the mapping depends on the rest of the frame.
_SAME_PIXEL_TWO_FRAMES = (
    np.array([[100.0, 200.0, 300.0]], dtype=np.float32),
    np.array([[100.0, 200.0, 600.0]], dtype=np.float32),
)


def test_physical_normalisation_is_frame_independent():
    # Two frames of the same scene must scale identically, or a model trained on
    # one cannot be applied to another.
    a, b = _SAME_PIXEL_TWO_FRAMES
    assert normalise(a)[0, 1] == pytest.approx(normalise(b)[0, 1])


def test_per_frame_normalisation_is_not_frame_independent():
    a, b = _SAME_PIXEL_TWO_FRAMES
    # 200 C becomes 0.5 in one frame and 0.2 in the other, purely because
    # something elsewhere in the frame got hotter.
    assert normalise(a, "per_frame")[0, 1] == pytest.approx(0.5)
    assert normalise(b, "per_frame")[0, 1] == pytest.approx(0.2)


def test_constant_frame_does_not_divide_by_zero():
    assert np.all(normalise(np.full((4, 4), 20.0), "per_frame") == 0.0)


def test_unknown_mode_is_refused():
    with pytest.raises(DegenerateInputError, match="per_frame"):
        normalise(np.zeros((2, 2)), "zscore")


def test_all_nan_frame_is_refused():
    with pytest.raises(DegenerateInputError):
        normalise(np.full((2, 2), np.nan))
