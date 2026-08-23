"""Published band selections (PRD v2 changes C1/C3)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.errors import ConfigError
from pipeline.hyperspectral.band_presets import (
    BAND_PRESETS,
    PRESET_SOURCE_BANDS,
    good_band_indices,
    preset_band_count,
)


def test_preset_widths_match_had100() -> None:
    """Transcription check against HAD100's published band_select()."""
    assert preset_band_count("had100_aviris_ng") == 276
    assert preset_band_count("had100_aviris") == 162


def test_ng_preset_is_exactly_the_published_ranges() -> None:
    expected = np.r_[15:109, 118:145, 158:187, 227:274, 328:407]
    assert np.array_equal(BAND_PRESETS["had100_aviris_ng"], expected)


def test_classic_preset_is_exactly_the_published_ranges() -> None:
    expected = np.r_[7:57, 65:79, 85:104, 122:149, 172:224]
    assert np.array_equal(BAND_PRESETS["had100_aviris"], expected)


@pytest.mark.parametrize("preset", sorted(BAND_PRESETS))
def test_indices_are_in_range_for_their_sensor(preset: str) -> None:
    idx = BAND_PRESETS[preset]
    assert idx.min() >= 0
    assert idx.max() < PRESET_SOURCE_BANDS[preset]


@pytest.mark.parametrize("preset", sorted(BAND_PRESETS))
def test_indices_are_sorted_and_unique(preset: str) -> None:
    idx = BAND_PRESETS[preset]
    assert np.array_equal(idx, np.unique(idx))


def test_selection_applies_to_a_real_shaped_cube() -> None:
    cube = np.zeros((64, 64, 425), dtype=np.float32)
    idx = good_band_indices("had100_aviris_ng", cube.shape[2])
    assert cube[:, :, idx].shape == (64, 64, 276)


def test_wrong_sensor_is_refused_not_clipped() -> None:
    """AVIRIS-Classic indices on an AVIRIS-NG cube select real but wrong bands.

    Every index is in range, so nothing downstream would catch it -- which is
    exactly why it has to be refused here.
    """
    with pytest.raises(ConfigError, match="224-band sensor"):
        good_band_indices("had100_aviris", 425)


def test_ng_preset_refused_on_classic_cube() -> None:
    with pytest.raises(ConfigError, match="425-band sensor"):
        good_band_indices("had100_aviris_ng", 224)


def test_unknown_preset_lists_the_known_ones() -> None:
    with pytest.raises(ConfigError) as exc:
        good_band_indices("had100_hyperion", 224)
    assert "had100_aviris" in str(exc.value)


def test_the_two_sensors_are_not_band_compatible() -> None:
    """The C2 consequence: one autoencoder cannot consume both corpora.

    Documented as a test so the constraint is discovered here rather than as a
    shape error mid-training.
    """
    assert preset_band_count("had100_aviris") != preset_band_count("had100_aviris_ng")


# --------------------------------------------------------------------------
# HAD100's published band-count protocol, recovered by measurement
# --------------------------------------------------------------------------


def test_band_count_presets_take_the_first_n_bands():
    # Not a uniform subsample and not a stride. Established by reproducing the
    # benchmark's own GRX baseline: first-N matched 0.9799/0.9714 exactly at
    # band50/band100, where linspace gave 0.9742/0.9710 and stride 0.9727/0.9701.
    assert good_band_indices("had100_band50", 276).tolist() == list(range(50))
    assert good_band_indices("had100_band100", 276).tolist() == list(range(100))
    assert good_band_indices("had100_band200", 276).tolist() == list(range(200))


def test_band_count_presets_apply_to_selected_crops_not_raw_scenes():
    # 276 is the width AFTER HAD100's main.py has run. Handing this preset a raw
    # 425-band scene would take the first 50 raw bands -- a different, wrong
    # wavelength window that nothing downstream could detect.
    with pytest.raises(ConfigError, match="276-band"):
        good_band_indices("had100_band50", 425)


def test_band_count_presets_are_nested():
    fifty = good_band_indices("had100_band50", 276)
    hundred = good_band_indices("had100_band100", 276)
    assert fifty.tolist() == hundred[:50].tolist()
