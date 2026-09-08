"""Thermal loading, and the radiometric/display classification it turns on.

The classification is the whole point of this module (PRD §11 edge case 9), so
most of these tests are about it rather than about parsing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.config import DatasetConfig, ThermalPreprocessConfig
from pipeline.errors import (
    DataUnavailableError,
    DegenerateInputError,
    IncompatibleInputError,
    MissingGroundTruthError,
    NonRadiometricInputError,
)
from pipeline.thermal.loader import (
    FLAME3_DISPLAY_SUBDIR,
    FLAME3_RADIOMETRIC_SUBDIR,
    KELVIN_OFFSET,
    Radiometry,
    ThermalFrame,
    classify_radiometry,
    list_frames,
    load_dataset,
    load_frame,
)

Image = pytest.importorskip("PIL.Image", reason="Pillow reads every thermal format")

FLAME3 = Path("data/thermal/flame3_cv_subset")
EQUIP = Path("data/thermal/thermal_images_equip")


# --------------------------------------------------------------------------
# Classification -- synthetic, so the rules are pinned independently of data
# --------------------------------------------------------------------------


def test_float_single_channel_is_radiometric():
    field = np.linspace(10.0, 400.0, 64, dtype=np.float32).reshape(8, 8)
    verdict = classify_radiometry(field)
    assert verdict.radiometric
    assert verdict.units == "celsius"


def test_three_channel_is_a_picture_whatever_its_dtype():
    # Float RGB is still a colormap: the mapping is lossy and non-invertible.
    rgb = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    verdict = classify_radiometry(rgb)
    assert not verdict.radiometric
    assert "3-channel" in verdict.reason


def test_uint8_is_a_display_product():
    verdict = classify_radiometry(np.zeros((8, 8), dtype=np.uint8))
    assert not verdict.radiometric
    assert "256 levels" in verdict.reason


def test_uint16_is_digital_numbers_not_temperature():
    # Genuinely ambiguous input: raw counts may be calibratable, but an
    # uncalibrated count is not a temperature.
    verdict = classify_radiometry(np.full((8, 8), 12000, dtype=np.uint16))
    assert not verdict.radiometric
    assert verdict.units == "digital_number"


def test_config_can_assert_radiometry_for_documented_integer_data():
    cfg = ThermalPreprocessConfig(assume_radiometric=True)
    verdict = classify_radiometry(np.full((8, 8), 300, dtype=np.uint16), cfg=cfg)
    assert verdict.radiometric
    assert not verdict.detected, "an asserted verdict must not claim to be detected"


def test_config_cannot_assert_radiometry_onto_8_bit():
    # assume_radiometric closes a calibration gap; it cannot restore information
    # the file never held.
    cfg = ThermalPreprocessConfig(assume_radiometric=True)
    with pytest.raises(NonRadiometricInputError, match="256 levels"):
        classify_radiometry(np.zeros((8, 8), dtype=np.uint8), cfg=cfg)


def test_tiff_sample_format_tag_is_believed_over_dtype():
    tags = {339: (3,)}
    verdict = classify_radiometry(
        np.linspace(5.0, 90.0, 64).reshape(8, 8), tiff_tags=tags
    )
    assert verdict.radiometric
    assert "SampleFormat" in verdict.reason


def test_normalised_float_display_data_is_refused():
    # Floats confined to [0, 1] are a picture scaled for display, not degrees.
    # The dtype test alone would wave this through, and the plausible-range test
    # cannot catch it either -- 0..1 C is a valid temperature interval.
    verdict = classify_radiometry(np.random.default_rng(1).random((8, 8)))
    assert not verdict.radiometric
    assert "normalised for display" in verdict.reason


def test_absurd_magnitudes_are_refused():
    verdict = classify_radiometry(np.full((8, 8), 5e4, dtype=np.float32))
    assert not verdict.radiometric


def test_all_nan_frame_is_refused():
    with pytest.raises(DegenerateInputError):
        classify_radiometry(np.full((8, 8), np.nan, dtype=np.float32))


def test_one_dimensional_input_is_refused():
    with pytest.raises(DegenerateInputError):
        classify_radiometry(np.zeros(8, dtype=np.float32))


def test_verdict_carries_its_evidence_into_the_run_record():
    labels = classify_radiometry(np.full((4, 4), 25.0, dtype=np.float32)).as_labels()
    assert labels["radiometric"] == "true"
    assert labels["radiometry_basis"] == "detected"
    assert labels["radiometry_reason"]


# --------------------------------------------------------------------------
# ThermalFrame
# --------------------------------------------------------------------------


def _frame(radiometric=True, **kw) -> ThermalFrame:
    verdict = Radiometry(radiometric, "celsius" if radiometric else "digital_number", "test")
    return ThermalFrame(data=np.full((4, 4), 20.0, dtype=np.float32), radiometry=verdict, **kw)


def test_require_radiometric_returns_the_field_when_it_is_one():
    assert _frame().require_radiometric().shape == (4, 4)


def test_require_radiometric_refuses_a_display_product_with_a_remedy():
    with pytest.raises(NonRadiometricInputError, match="to fix"):
        _frame(radiometric=False).require_radiometric()


def test_to_kelvin_converts_from_celsius():
    assert _frame().to_kelvin() == pytest.approx(20.0 + KELVIN_OFFSET)


def test_to_kelvin_routes_through_the_refusal():
    # Eq. 5 must not be reachable from a display product by any path.
    with pytest.raises(NonRadiometricInputError):
        _frame(radiometric=False).to_kelvin()


def test_kelvin_data_is_not_shifted_twice():
    verdict = Radiometry(True, "kelvin", "test")
    frame = ThermalFrame(np.full((2, 2), 300.0, dtype=np.float32), verdict)
    assert frame.to_kelvin() == pytest.approx(300.0)


def test_image_label_and_pixel_mask_are_different_things():
    # Collapsing them would let a scalar reach code expecting a mask.
    frame = _frame(label=True)
    assert frame.require_label() is True
    with pytest.raises(MissingGroundTruthError, match="image-level"):
        frame.require_ground_truth()


def test_mask_must_match_the_frame_grid():
    verdict = Radiometry(True, "celsius", "test")
    with pytest.raises(IncompatibleInputError):
        ThermalFrame(
            np.zeros((4, 4), dtype=np.float32), verdict, ground_truth=np.zeros((8, 8), bool)
        )


def test_three_dimensional_frame_is_refused():
    verdict = Radiometry(True, "celsius", "test")
    with pytest.raises(DegenerateInputError):
        ThermalFrame(np.zeros((4, 4, 3), dtype=np.float32), verdict)


def test_missing_file_names_the_path_and_the_fix():
    with pytest.raises(DataUnavailableError) as exc:
        load_frame("data/thermal/does_not_exist.tiff")
    assert "download_data.py" in str(exc.value)


# --------------------------------------------------------------------------
# Against the real datasets
# --------------------------------------------------------------------------

needs_flame3 = pytest.mark.skipif(
    not (FLAME3 / "Fire" / FLAME3_RADIOMETRIC_SUBDIR).is_dir(),
    reason="FLAME 3 not present; run scripts/download_data.py --verify --only flame3",
)
needs_equip = pytest.mark.skipif(
    not EQUIP.is_dir(), reason="thermal_images_equip negative fixture not present"
)


@needs_flame3
def test_flame3_celsius_tiff_is_detected_as_radiometric():
    path = next(iter(list_frames(FLAME3 / "Fire" / FLAME3_RADIOMETRIC_SUBDIR)))
    frame = load_frame(path, label=True)
    assert frame.is_radiometric
    assert frame.radiometry.units == "celsius"
    assert frame.shape == (512, 640)


@needs_flame3
def test_flame3_raw_jpg_of_the_same_scene_is_refused():
    # The strongest form of this test: same dataset, same scene, one radiometric
    # export and one display export. A classifier that separates these is not
    # keying on something incidental to two unrelated datasets.
    radiometric = load_frame(
        next(iter(list_frames(FLAME3 / "Fire" / FLAME3_RADIOMETRIC_SUBDIR)))
    )
    display = load_frame(next(iter(list_frames(FLAME3 / "Fire" / FLAME3_DISPLAY_SUBDIR))))

    assert radiometric.is_radiometric
    assert not display.is_radiometric
    with pytest.raises(NonRadiometricInputError):
        display.to_kelvin()


@needs_flame3
def test_fire_frames_are_hotter_than_no_fire_frames():
    # Sanity on the physics, not just the plumbing: if these agreed, either the
    # labels or the radiometry would be wrong.
    fire = load_frame(next(iter(list_frames(FLAME3 / "Fire" / FLAME3_RADIOMETRIC_SUBDIR))))
    calm = load_frame(next(iter(list_frames(FLAME3 / "No Fire" / FLAME3_RADIOMETRIC_SUBDIR))))
    assert fire.data.max() > 100.0
    assert calm.data.max() < 100.0


@needs_flame3
def test_load_dataset_labels_from_the_folder_split():
    frames = load_dataset(
        DatasetConfig(name="flame3", kind="thermal", path=str(FLAME3)), limit=3
    )
    assert len(frames) == 6
    assert sum(f.label for f in frames) == 3
    assert all(f.is_radiometric for f in frames)
    assert all(f.labels["label_granularity"] == "image" for f in frames)


@needs_flame3
def test_load_dataset_can_read_the_display_export_for_the_refusal_path():
    frames = load_dataset(
        DatasetConfig(name="flame3", kind="thermal", path=str(FLAME3)),
        limit=2,
        subdir=FLAME3_DISPLAY_SUBDIR,
    )
    assert not any(f.is_radiometric for f in frames)


@needs_equip
def test_equipment_fixture_is_the_negative_case_it_is_kept_for():
    # Its only role under change C8: proving Planck refuses rather than
    # fabricating a temperature from 8-bit false colour.
    path = next(iter(list_frames(EQUIP, "*.bmp")))
    frame = load_frame(path)
    assert not frame.is_radiometric
    with pytest.raises(NonRadiometricInputError):
        frame.require_radiometric()


def test_a_real_cold_scene_is_not_mistaken_for_normalised_data():
    # The discriminator is physical: a real scene has thermal contrast. A frost
    # scene spanning -5..8 C is unambiguous temperature, and must survive.
    field = np.linspace(-5.0, 8.0, 64, dtype=np.float32).reshape(8, 8)
    assert classify_radiometry(field).radiometric


def test_a_sub_kelvin_frame_is_treated_as_display_data():
    # Under one kelvin of contrast across a whole frame is not a scene. This is
    # deliberately refused rather than guessed at, with assume_radiometric as
    # the documented override -- see test_normalised_data_can_be_overruled.
    field = np.linspace(0.2, 0.8, 64, dtype=np.float32).reshape(8, 8)
    assert not classify_radiometry(field).radiometric


def test_normalised_data_can_be_overruled_when_documented():
    cfg = ThermalPreprocessConfig(assume_radiometric=True)
    verdict = classify_radiometry(np.linspace(0.0, 1.0, 64).reshape(8, 8), cfg=cfg)
    assert verdict.radiometric
    assert not verdict.detected
