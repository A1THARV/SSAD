"""Invariants of the controlled mixing experiment.

The experiment is synthetic, so its conclusions are only as good as its
construction. These pin the three construction errors that were actually made
while building it, each of which produced a plausible but meaningless table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from sam_mixing_experiment import (  # noqa: E402
    ABLATION_SNRS_DB,
    BAND_SUBSETS,
    add_noise,
    background_spectrum,
    resample_library,
    run,
    sensor_wavelengths_nm,
)


def test_sensor_grid_matches_had100s_band_selection():
    wl = sensor_wavelengths_nm()
    assert wl.size == 276
    assert 450 < wl[0] < 460 and 2400 < wl[-1] < 2420


def test_band_subsets_select_the_intended_windows():
    wl = sensor_wavelengths_nm()
    counts = {b.name: int(b.mask(wl).sum()) for b in BAND_SUBSETS}
    assert counts["full"] == 276
    assert counts["visible"] == 50, "the §3 window is the first 50 bands"
    assert counts["swir"] > counts["visible"]


# --------------------------------------------------------------------------
# Construction error 1: mismatched units make the abundance axis a lie
# --------------------------------------------------------------------------


def test_background_is_rescaled_to_the_library_units():
    # HAD100's crops are not reflectance fractions (mean ~2.5 against a library
    # mean of ~0.6). Unscaled, a nominal a=0.75 pixel takes MORE signal from the
    # "25%" background than from the target, so the axis labels would be wrong.
    wl = sensor_wavelengths_nm()
    bg, _ = background_spectrum(wl, target_mean=0.6)
    assert bg.mean() == pytest.approx(0.6, rel=1e-9)


def test_rescaling_preserves_spectral_shape():
    # Only the magnitude may change: SAM sees shape, and altering it would make
    # the background a different material.
    wl = sensor_wavelengths_nm()
    a, _ = background_spectrum(wl, target_mean=0.6)
    b, _ = background_spectrum(wl, target_mean=6.0)
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# Construction error 2: a noiseless experiment is exactly solvable
# --------------------------------------------------------------------------


def _tiny_setup(n_materials=6, n_bands=None):
    wl = sensor_wavelengths_nm()
    rng = np.random.default_rng(0)
    lib_nm = np.linspace(400.0, 2500.0, 128)
    library = np.abs(rng.normal(0.5, 0.15, (n_materials, lib_nm.size)))
    materials = np.array([f"m{i}" for i in range(n_materials)])
    bg, _ = background_spectrum(wl, target_mean=float(library.mean()))
    return lib_nm, library, materials, wl, bg


def test_noiseless_background_removed_matching_is_exactly_solvable():
    # observed - bg = a*(target - bg) is the reference scaled by a, and SAM is
    # magnitude-invariant, so the angle to the right entry is identically zero.
    # A noiseless band ablation therefore scores 1.000 everywhere and would
    # appear to REFUTE §3 while measuring nothing.
    lib_nm, library, materials, wl, bg = _tiny_setup()
    resampled = resample_library(library, lib_nm, wl)
    acc = run(lib_nm, library, materials, wl, bg, BAND_SUBSETS[1], 0.05,
              np.inf, 40, np.random.default_rng(1),
              remove_background=True, resampled=resampled)
    assert acc == 1.0


def test_the_band_ablation_never_runs_noiseless():
    # Guards the above: if noiseless were ever added back, the ablation would
    # silently become uninformative rather than fail.
    assert all(np.isfinite(s) for s in ABLATION_SNRS_DB)
    assert ABLATION_SNRS_DB, "the ablation needs at least one noise level"


# --------------------------------------------------------------------------
# Construction error 3: matching the raw pixel is a different question
# --------------------------------------------------------------------------


def test_raw_matching_collapses_for_a_minority_target():
    # Not a flaw in SAM: as abundance falls every mixture converges on the
    # background regardless of what was mixed in, so the nearest library entry
    # is whichever sits closest to the background, the same one every trial.
    lib_nm, library, materials, wl, bg = _tiny_setup()
    resampled = resample_library(library, lib_nm, wl)
    raw = run(lib_nm, library, materials, wl, bg, BAND_SUBSETS[0], 0.05,
              np.inf, 60, np.random.default_rng(2),
              remove_background=False, resampled=resampled)
    removed = run(lib_nm, library, materials, wl, bg, BAND_SUBSETS[0], 0.05,
                  np.inf, 60, np.random.default_rng(2),
                  remove_background=True, resampled=resampled)
    assert raw < removed
    assert removed == 1.0


def test_noise_is_applied_at_the_stated_snr():
    rng = np.random.default_rng(0)
    signal = np.full(4096, 2.0)
    noisy = add_noise(signal, 20.0, rng)
    measured = 10 * np.log10(np.mean(signal**2) / np.mean((noisy - signal) ** 2))
    assert measured == pytest.approx(20.0, abs=0.5)


def test_infinite_snr_is_a_no_op():
    signal = np.linspace(0.1, 1.0, 32)
    assert add_noise(signal, np.inf, np.random.default_rng(0)) is signal


def test_scoring_is_by_material_not_by_spectrum_index():
    # The library holds 51 quartz entries; requiring the exact one would measure
    # sample recall rather than material identification.
    lib_nm = np.linspace(400.0, 2500.0, 64)
    rng = np.random.default_rng(3)
    base = np.abs(rng.normal(0.5, 0.1, lib_nm.size))
    # Two near-identical spectra sharing a material label.
    library = np.vstack([base, base * 1.001, np.abs(rng.normal(0.5, 0.1, lib_nm.size))])
    materials = np.array(["quartz", "quartz", "calcite"])
    wl = sensor_wavelengths_nm()
    bg, _ = background_spectrum(wl, target_mean=float(library.mean()))
    resampled = resample_library(library, lib_nm, wl)
    acc = run(lib_nm, library, materials, wl, bg, BAND_SUBSETS[0], 0.5,
              30.0, 40, np.random.default_rng(4),
              remove_background=True, resampled=resampled)
    # Picking either quartz for a quartz target must count as correct.
    assert acc > 0.8
