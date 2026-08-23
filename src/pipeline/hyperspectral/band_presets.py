"""Published band selections, reproduced exactly for comparability.

FR-9.1 derives a good-band mask from wavelengths (water absorption, low SNR).
That is the right default for a scene we score on our own terms. It is the
*wrong* thing to do when scoring against someone else's published table.

PRD v2 change C3 positions the pipeline within HAD100's 17 published baselines.
Every one of those numbers was produced after HAD100's own band selection, so
using a different mask -- even a defensible one -- would compare our detector on
our bands against their detectors on theirs, and the ranking would be
meaningless. These presets are transcribed from HAD100's ``main.py``
(``band_select``) so the comparison is like-for-like.

Note that the two sensors survive selection at different widths, 162 against
276. That is not a detail: a single autoencoder cannot consume both without
resampling to a common grid, which is why :class:`IncompatibleInputError`
exists (change C6).
"""

from __future__ import annotations

import numpy as np

from ..errors import ConfigError

__all__ = ["BAND_PRESETS", "PRESET_SOURCE_BANDS", "good_band_indices", "preset_band_count"]

#: Band index selections, keyed by preset name.
#:
#: Transcribed verbatim from HAD100 ``main.py::band_select``:
#:
#: * ``aviris_ng``  -> ``np.r_[15:109, 118:145, 158:187, 227:274, 328:407]``
#: * ``aviris``     -> ``np.r_[7:57, 65:79, 85:104, 122:149, 172:224]``
BAND_PRESETS: dict[str, np.ndarray] = {
    "had100_aviris_ng": np.r_[15:109, 118:145, 158:187, 227:274, 328:407],
    "had100_aviris": np.r_[7:57, 65:79, 85:104, 122:149, 172:224],
    # HAD100's published baseline table has three columns -- band50, band100,
    # band200 -- and its shipped code does not say what they mean. Recovered by
    # measurement: plain RX (Eq. 2) on the FIRST N of the 276 already-selected
    # bands reproduces the benchmark's own GRX baseline to four decimal places
    # (0.9799 / 0.9714 / 0.9653 against their 0.9799 / 0.9714 / 0.9649).
    # Uniform subsampling and strided selection both miss it; see
    # scripts/had100_baselines.py and RESULTS.md.
    #
    # These apply to the 276-band .npy crops main.py writes, NOT to a raw
    # 425-band scene -- hence the separate source-band entry below.
    "had100_band50": np.arange(50),
    "had100_band100": np.arange(100),
    "had100_band200": np.arange(200),
}

#: Band count each preset expects its *input* cube to have, before selection.
#: A cube of any other width means the preset is being applied to the wrong
#: sensor, which would silently select meaningless bands.
PRESET_SOURCE_BANDS: dict[str, int] = {
    "had100_aviris_ng": 425,
    "had100_aviris": 224,
    "had100_band50": 276,
    "had100_band100": 276,
    "had100_band200": 276,
}


def good_band_indices(preset: str, n_bands: int) -> np.ndarray:
    """Band indices kept by a named preset.

    Args:
        preset: A key of :data:`BAND_PRESETS`.
        n_bands: Band count of the cube the preset is about to be applied to,
            checked against the sensor the preset was written for.

    Returns:
        Sorted integer indices into the cube's band axis.

    Raises:
        ConfigError: the preset is unknown, or the cube's band count does not
            match the sensor it describes. Applying AVIRIS-Classic's indices to
            an AVIRIS-NG cube would select real but wrong bands and fail
            silently, so it is refused rather than clipped.
    """
    if preset not in BAND_PRESETS:
        known = ", ".join(sorted(BAND_PRESETS))
        raise ConfigError(f"Unknown band preset {preset!r}. Known presets: {known}.")

    expected = PRESET_SOURCE_BANDS[preset]
    if n_bands != expected:
        raise ConfigError(
            f"Band preset {preset!r} describes a {expected}-band sensor, but the cube "
            f"has {n_bands} bands. Applying it anyway would select valid indices from "
            "the wrong instrument, which no downstream check would catch. Set "
            "dataset.sensor correctly, or use good_bands_preset: none."
        )
    return BAND_PRESETS[preset]


def preset_band_count(preset: str) -> int:
    """How many bands a preset keeps."""
    if preset not in BAND_PRESETS:
        known = ", ".join(sorted(BAND_PRESETS))
        raise ConfigError(f"Unknown band preset {preset!r}. Known presets: {known}.")
    return int(BAND_PRESETS[preset].size)
