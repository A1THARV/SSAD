"""Spectral Angle Mapper, paper Eq. 4 (FR-9.6)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.errors import DegenerateInputError, IncompatibleInputError
from pipeline.hyperspectral.sam import (
    MaterialIdentification,
    match_material,
    resample_to_grid,
    spectral_angle,
    spectral_angle_batch,
    spectral_angle_paired,
)


# --------------------------------------------------------------------------
# Closed-form angles -- FR-9.6's acceptance test
# --------------------------------------------------------------------------


def test_identical_spectra_give_zero_angle() -> None:
    """FR-9.6: feeding a reference back against itself returns theta ~ 0.

    Tolerance is tight deliberately. The arccos form of Eq. 4 cannot reach it --
    it bottoms out near 1.5e-8 because arccos is ill-conditioned at zero angle --
    so this pins the well-conditioned atan2 formulation.
    """
    rng = np.random.default_rng(0)
    spectrum = np.abs(rng.normal(size=50)) + 0.1
    assert spectral_angle(spectrum, spectrum) == pytest.approx(0.0, abs=1e-15)


def test_scaled_spectrum_gives_zero_angle() -> None:
    """SAM compares shape, not magnitude -- the property §III-B relies on.

    A spectrum scaled by illumination must be identified as the same material.
    """
    rng = np.random.default_rng(1)
    spectrum = np.abs(rng.normal(size=40)) + 0.1
    assert spectral_angle(spectrum, spectrum * 0.37) == pytest.approx(0.0, abs=1e-9)


def test_orthogonal_spectra_give_ninety_degrees() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert spectral_angle(a, b) == pytest.approx(np.pi / 2)


def test_hand_computed_forty_five_degrees() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 1.0])
    assert spectral_angle(a, b) == pytest.approx(np.pi / 4)


def test_hand_computed_sixty_degrees() -> None:
    """cos(theta) = 0.5 for these two unit-length vectors."""
    a = np.array([1.0, 0.0])
    b = np.array([0.5, np.sqrt(3) / 2])
    assert spectral_angle(a, b) == pytest.approx(np.pi / 3)


def test_dissimilar_materials_give_a_large_angle() -> None:
    """FR-9.6: two maximally dissimilar references must be far apart.

    Water falls steeply across the SWIR; a bright flat metal does not.
    """
    wavelengths = np.linspace(400, 2400, 60)
    water = np.exp(-(wavelengths - 400) / 400) + 0.01
    metal = np.full_like(wavelengths, 0.8)

    assert spectral_angle(water, metal) > 0.5


def test_angles_are_symmetric() -> None:
    rng = np.random.default_rng(2)
    a, b = np.abs(rng.normal(size=30)) + 0.1, np.abs(rng.normal(size=30)) + 0.1
    assert spectral_angle(a, b) == pytest.approx(spectral_angle(b, a))


def test_angles_stay_within_range() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(20, 12))
    b = rng.normal(size=(15, 12))
    angles = spectral_angle_batch(a, b)
    assert angles.shape == (20, 15)
    assert np.all(angles >= 0.0) and np.all(angles <= np.pi)


def test_batch_matches_scalar() -> None:
    """The vectorised form Eq. 3's loss uses must agree with the scalar one."""
    rng = np.random.default_rng(4)
    queries = np.abs(rng.normal(size=(5, 20))) + 0.1
    refs = np.abs(rng.normal(size=(4, 20))) + 0.1

    batch = spectral_angle_batch(queries, refs)
    for i in range(5):
        for j in range(4):
            assert batch[i, j] == pytest.approx(spectral_angle(queries[i], refs[j]))


def test_near_identical_spectra_do_not_produce_nan() -> None:
    """Float error can push the cosine past 1.0, where arccos returns NaN."""
    spectrum = np.linspace(0.1, 0.9, 200)
    perturbed = spectrum * (1.0 + 1e-16)
    angle = spectral_angle(spectrum, perturbed)
    assert np.isfinite(angle)
    assert angle == pytest.approx(0.0, abs=1e-7)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_mismatched_grids_are_refused_not_broadcast() -> None:
    """Library spectra ship on their own grids -- the C4 failure mode."""
    with pytest.raises(IncompatibleInputError, match="resample"):
        spectral_angle_batch(np.ones(50), np.ones(60))


def test_zero_spectrum_is_refused() -> None:
    with pytest.raises(DegenerateInputError, match="no direction"):
        spectral_angle(np.zeros(10), np.ones(10))


def test_nan_spectrum_is_refused() -> None:
    bad = np.ones(10)
    bad[3] = np.nan
    with pytest.raises(DegenerateInputError, match="non-finite"):
        spectral_angle(bad, np.ones(10))


def test_single_band_is_refused() -> None:
    """Every one-band vector is parallel, so the angle carries no information."""
    with pytest.raises(DegenerateInputError, match="at least 2"):
        spectral_angle(np.ones(1), np.ones(1))


# --------------------------------------------------------------------------
# Resampling -- PRD v2 change C4
# --------------------------------------------------------------------------


def test_resampling_puts_a_library_spectrum_on_the_sensor_grid() -> None:
    src_wl = np.linspace(400, 2500, 200)
    sensor_wl = np.linspace(450, 2400, 60)
    spectrum = np.sin(src_wl / 300.0) + 2.0

    resampled = resample_to_grid(spectrum, src_wl, sensor_wl)

    assert resampled.shape == (60,)
    assert np.all(np.isfinite(resampled))


def test_resampling_makes_sam_usable() -> None:
    """End to end: mismatched grids fail, resampled ones succeed."""
    src_wl = np.linspace(400, 2500, 173)
    sensor_wl = np.linspace(450, 2400, 60)
    library_spectrum = np.linspace(0.2, 0.8, 173)
    pixel = np.linspace(0.2, 0.8, 60)

    with pytest.raises(IncompatibleInputError):
        spectral_angle(pixel, library_spectrum)

    resampled = resample_to_grid(library_spectrum, src_wl, sensor_wl)
    assert spectral_angle(pixel, resampled) < 0.05


def test_resampling_is_exact_on_shared_points() -> None:
    src_wl = np.array([400.0, 500.0, 600.0, 700.0])
    spectrum = np.array([0.1, 0.5, 0.3, 0.9])
    got = resample_to_grid(spectrum, src_wl, np.array([500.0, 700.0]))
    assert got == pytest.approx([0.5, 0.9])


def test_extrapolation_is_refused_not_clamped() -> None:
    """Clamping would invent flat shape, which SAM reads as real similarity."""
    src_wl = np.linspace(1000, 2000, 50)
    sensor_wl = np.linspace(400, 2500, 60)
    with pytest.raises(IncompatibleInputError, match="Extrapolating"):
        resample_to_grid(np.ones(50), src_wl, sensor_wl)


def test_unsorted_source_grid_is_handled() -> None:
    src_wl = np.array([600.0, 400.0, 500.0])
    spectrum = np.array([0.3, 0.1, 0.5])
    got = resample_to_grid(spectrum, src_wl, np.array([500.0]))
    assert got == pytest.approx([0.5])


# --------------------------------------------------------------------------
# match_material -- FR-9.6 and §11 edge case 7
# --------------------------------------------------------------------------


@pytest.fixture()
def library() -> dict[str, np.ndarray]:
    wl = np.linspace(400, 2400, 40)
    return {
        "vegetation": np.where(wl < 700, 0.05, 0.45) + 0.01 * (wl / 2400),
        "water": np.exp(-(wl - 400) / 350) + 0.01,
        "concrete": np.full_like(wl, 0.35) + 0.00005 * wl,
        "asphalt": np.full_like(wl, 0.08) + 0.00002 * wl,
        "metal": np.full_like(wl, 0.85),
    }


def test_top_k_are_returned_nearest_first(library) -> None:
    result = match_material(library["water"], library, top_k=3)

    assert len(result.candidates) == 3
    angles = [c.angle_rad for c in result.candidates]
    assert angles == sorted(angles)


def test_exact_library_member_is_its_own_best_match(library) -> None:
    result = match_material(library["concrete"], library, top_k=3)
    assert result.best.name == "concrete"
    assert result.best.angle_rad == pytest.approx(0.0, abs=1e-9)
    assert result.low_confidence is False


def test_every_candidate_carries_its_angle(library) -> None:
    """FR-9.6 forbids a bare label with no confidence indication."""
    result = match_material(library["vegetation"], library, top_k=3)
    for c in result.candidates:
        assert isinstance(c.angle_rad, float)
        assert c.angle_deg == pytest.approx(np.degrees(c.angle_rad))


def test_unmatched_spectrum_is_flagged_low_confidence(library) -> None:
    """§11 edge case 7 -- never assert a match the library cannot support."""
    wl = np.linspace(400, 2400, 40)
    alien = np.abs(np.sin(wl / 40.0)) + 0.01  # oscillates unlike any entry

    result = match_material(alien, library, top_k=3, low_confidence_angle_rad=0.25)

    assert result.low_confidence is True
    assert result.best.low_confidence is True
    # The shortlist is still returned -- flagged, not withheld.
    assert len(result.candidates) == 3


def test_threshold_governs_the_flag(library) -> None:
    wl = np.linspace(400, 2400, 40)
    alien = np.abs(np.sin(wl / 40.0)) + 0.01

    strict = match_material(alien, library, low_confidence_angle_rad=0.01)
    lenient = match_material(alien, library, low_confidence_angle_rad=np.pi)

    assert strict.low_confidence is True
    assert lenient.low_confidence is False


def test_top_k_larger_than_library_is_clamped(library) -> None:
    result = match_material(library["metal"], library, top_k=99)
    assert len(result.candidates) == len(library)
    assert result.library_size == len(library)


def test_result_serialises_for_the_report(library) -> None:
    result = match_material(library["asphalt"], library, top_k=2)
    payload = result.to_dict()

    assert len(payload["candidates"]) == 2
    assert payload["candidates"][0]["name"] == "asphalt"
    assert "low_confidence" in payload
    assert payload["library_size"] == 5


def test_empty_library_is_actionable() -> None:
    with pytest.raises(DegenerateInputError, match="curated ECOSTRESS"):
        match_material(np.ones(10), {})


def test_ragged_library_is_refused() -> None:
    with pytest.raises(IncompatibleInputError, match="common grid"):
        match_material(np.ones(10), {"a": np.ones(10), "b": np.ones(12)})


def test_invalid_top_k_is_refused(library) -> None:
    with pytest.raises(DegenerateInputError, match="top_k"):
        match_material(library["water"], library, top_k=0)


def test_water_and_metal_rank_far_apart(library) -> None:
    """The FR-9.6 sanity check, through the matching path."""
    result = match_material(library["water"], library, top_k=5)
    by_name = {c.name: c.angle_rad for c in result.candidates}
    assert by_name["water"] < by_name["metal"]
    assert by_name["metal"] > 0.5


def test_mean_anomaly_spectrum_workflow(library) -> None:
    """How the CLI will use it: mean spectrum over a detected anomaly region."""
    wl = np.linspace(400, 2400, 40)
    region = np.stack([library["metal"] * s for s in (0.9, 1.0, 1.1)])
    mean_spectrum = region.mean(axis=0)

    result = match_material(mean_spectrum, library, top_k=3)

    assert result.best.name == "metal"
    assert result.low_confidence is False


# --------------------------------------------------------------------------
# Numerical conditioning -- matters because Eq. 3's loss lives near theta ~ 0
# --------------------------------------------------------------------------


def test_atan2_form_beats_arccos_near_zero_angle() -> None:
    """The reason spectral_angle_batch does not use arccos(u.v).

    arccos has infinite derivative at 1, so a cosine accurate to machine
    epsilon gives an angle accurate only to ~sqrt(eps) ~ 1.5e-8. Eq. 3's
    composite loss operates exactly there, where a well-reconstructed patch has
    a near-zero angle, so that error would become training noise.
    """
    rng = np.random.default_rng(10)
    spectrum = np.abs(rng.normal(size=100)) + 0.5
    unit = spectrum / np.linalg.norm(spectrum)

    arccos_angle = float(np.arccos(np.clip(unit @ unit, -1.0, 1.0)))
    ours = spectral_angle(spectrum, spectrum)

    assert ours <= arccos_angle
    assert ours < 1e-15


def test_tiny_angles_are_resolved_not_flattened_to_zero() -> None:
    """arccos would quantise these to 0; the atan2 form keeps them ordered."""
    base = np.ones(64)
    angles = []
    for eps in (1e-6, 1e-7, 1e-8):
        perturbed = base.copy()
        perturbed[0] += eps
        angles.append(spectral_angle(base, perturbed))

    assert all(a > 0 for a in angles)
    assert angles[0] > angles[1] > angles[2]


def test_paired_matches_batch_diagonal() -> None:
    rng = np.random.default_rng(11)
    a = np.abs(rng.normal(size=(6, 25))) + 0.1
    b = np.abs(rng.normal(size=(6, 25))) + 0.1

    paired = spectral_angle_paired(a, b)
    cross = spectral_angle_batch(a, b)

    assert paired.shape == (6,)
    assert paired == pytest.approx(np.diag(cross))


def test_paired_broadcasts_a_single_reference() -> None:
    rng = np.random.default_rng(12)
    stack = np.abs(rng.normal(size=(5, 20))) + 0.1
    one = np.abs(rng.normal(size=20)) + 0.1

    got = spectral_angle_paired(stack, one)

    assert got.shape == (5,)
    for i in range(5):
        assert got[i] == pytest.approx(spectral_angle(stack[i], one))


def test_paired_refuses_mismatched_stack_sizes() -> None:
    rng = np.random.default_rng(13)
    with pytest.raises(IncompatibleInputError, match="matching stack sizes"):
        spectral_angle_paired(rng.normal(size=(4, 10)) + 5, rng.normal(size=(3, 10)) + 5)


def test_paired_is_zero_for_perfect_reconstruction() -> None:
    """What Eq. 3's SAM term contributes when reconstruction is exact."""
    rng = np.random.default_rng(14)
    x = np.abs(rng.normal(size=(8, 30))) + 0.2
    assert spectral_angle_paired(x, x.copy()) == pytest.approx(np.zeros(8), abs=1e-15)


# --------------------------------------------------------------------------
# Magnitude invariance -- the property §III-B's justification rests on
# --------------------------------------------------------------------------
#
# theta = 2*atan2(||u-v||, ||u+v||) holds ONLY for unit vectors; it derives
# from ||u-v|| = 2 sin(theta/2) and ||u+v|| = 2 cos(theta/2). Applied to raw
# spectra it silently returns a wrong angle rather than failing, so the
# normalisation step needs tests that would fail without it.
#
# Note that test_identical_spectra_give_zero_angle does NOT discriminate here:
# u == v gives ||u-v|| == 0 whether or not either was normalised.


def test_parallel_spectra_of_different_magnitude_are_identical() -> None:
    """The counterexample: un-normalised this returns 0.6435, not 0."""
    u = np.array([1.0, 0.0])
    v = np.array([2.0, 0.0])
    assert spectral_angle(u, v) == pytest.approx(0.0, abs=1e-15)


def test_invariance_holds_at_non_zero_angles() -> None:
    """The stronger check: scaling must not move an angle that is not zero.

    A zero-angle test can be satisfied by accident; this one cannot.
    """
    u = np.array([1.0, 0.0])
    v = np.array([1.0, 1.0])
    base = spectral_angle(u, v)

    assert base == pytest.approx(np.pi / 4)
    for scale in (0.01, 3.0, 7.0, 1000.0):
        assert spectral_angle(u, scale * v) == pytest.approx(base, rel=1e-12)
        assert spectral_angle(scale * u, v) == pytest.approx(base, rel=1e-12)


def test_invariance_survives_independent_scaling_of_both() -> None:
    rng = np.random.default_rng(30)
    a = np.abs(rng.normal(size=64)) + 0.1
    b = np.abs(rng.normal(size=64)) + 0.1
    base = spectral_angle(a, b)
    assert spectral_angle(11.0 * a, 0.003 * b) == pytest.approx(base, rel=1e-12)


def test_invariance_holds_in_the_paired_form() -> None:
    """Eq. 3's loss uses this path, so it needs the property independently."""
    rng = np.random.default_rng(31)
    x = np.abs(rng.normal(size=(6, 30))) + 0.1
    y = np.abs(rng.normal(size=(6, 30))) + 0.1

    base = spectral_angle_paired(x, y)
    scaled = spectral_angle_paired(x * 250.0, y * 0.004)
    assert scaled == pytest.approx(base, rel=1e-12)


def test_invariance_holds_in_the_batch_form() -> None:
    rng = np.random.default_rng(32)
    x = np.abs(rng.normal(size=(4, 20))) + 0.1
    y = np.abs(rng.normal(size=(5, 20))) + 0.1

    base = spectral_angle_batch(x, y)
    scaled = spectral_angle_batch(x * 90.0, y * 0.02)
    assert scaled == pytest.approx(base, rel=1e-12)


def test_brightness_difference_alone_yields_no_angle() -> None:
    """Illumination and topographic shading scale a spectrum; SAM must ignore it.

    This is the concrete form of the paper's §III-B claim: a shaded and a
    sunlit pixel of the same material must be identified as the same material.
    """
    wl = np.linspace(400, 2400, 50)
    material = np.where(wl < 700, 0.05, 0.45)
    sunlit = material * 1.0
    shaded = material * 0.22  # same surface, deep shadow

    assert spectral_angle(sunlit, shaded) == pytest.approx(0.0, abs=1e-15)


def test_match_material_is_unaffected_by_query_brightness(library) -> None:
    """A shaded anomaly must still match its own material."""
    bright = match_material(library["vegetation"], library, top_k=3)
    dim = match_material(library["vegetation"] * 0.15, library, top_k=3)

    assert bright.best.name == dim.best.name == "vegetation"
    assert [c.angle_rad for c in dim.candidates] == pytest.approx(
        [c.angle_rad for c in bright.candidates], rel=1e-12
    )
