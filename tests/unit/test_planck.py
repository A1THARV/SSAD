"""Eq. 5, validated against an independent implementation.

The anchor test for M5. Self-consistency cannot catch a wrong physical constant
or a units error -- both produce smooth, positive, plausible output -- so the
radiance function is checked against ``astropy``'s BlackBody model, which was
written by different people from the same physics.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline.errors import DegenerateInputError
from pipeline.thermal.loader import Radiometry, ThermalFrame
from pipeline.thermal.planck import (
    BOLTZMANN_K,
    PLANCK_H,
    SPEED_OF_LIGHT,
    EmissivityEstimate,
    EmissivityTable,
    inverse_planck_temperature,
    planck_radiance,
    retrieve_temperature,
)

# --------------------------------------------------------------------------
# The external validation -- M5's exit criterion
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wavelength_um,temperature_k",
    [(10.0, 300.0), (10.0, 600.0), (4.0, 1200.0), (100.0, 50.0), (0.5, 5800.0)],
)
def test_radiance_matches_astropy_blackbody(wavelength_um, temperature_k):
    astropy_units = pytest.importorskip("astropy.units")
    from astropy.modeling.physical_models import BlackBody

    # astropy returns per-steradian spectral radiance; asking for it in SI
    # wavelength units makes the comparison direct rather than needing a
    # per-frequency-to-per-wavelength Jacobian, which is its own error source.
    reference = (
        BlackBody(temperature=temperature_k * astropy_units.K, scale=1.0)
        (wavelength_um * astropy_units.um)
        .to(
            astropy_units.W / astropy_units.m**2 / astropy_units.sr / astropy_units.m,
            equivalencies=astropy_units.spectral_density(
                wavelength_um * astropy_units.um
            ),
        )
        .value
    )
    ours = planck_radiance(wavelength_um * 1e-6, temperature_k)
    assert ours == pytest.approx(reference, rel=1e-9)


def test_constants_are_the_defined_si_values():
    # All three are exact by SI definition since 2019, so a mismatch is a typo
    # rather than a revision.
    assert PLANCK_H == 6.62607015e-34
    assert SPEED_OF_LIGHT == 2.99792458e8
    assert BOLTZMANN_K == 1.380649e-23


def test_wien_displacement_law_is_reproduced():
    # An independent physical check that does not go through astropy: the peak
    # of the spectral radiance curve must sit at b/T with b = 2.898e-3 m K.
    temperature = 1000.0
    lam = np.linspace(0.5e-6, 20e-6, 400_000)
    peak = lam[np.argmax(planck_radiance(lam, temperature))]
    assert peak * temperature == pytest.approx(2.897771955e-3, rel=1e-3)


#: np.trapezoid is NumPy >= 2.0; np.trapz is the 1.x spelling and is deprecated
#: in 2.x. Bound once so the test reads as physics rather than as version logic.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def test_stefan_boltzmann_is_reproduced_by_integration():
    # Integrating Eq. 5 over wavelength and solid angle must give sigma T^4.
    # A third independent check: astropy validates the curve pointwise, Wien
    # validates where its peak sits, and this validates the area under it.
    #
    # Log-spaced, and that is not a detail. Planck's curve spans decades of
    # wavelength, so linear sampling either misses the long tail or wastes
    # everything on it: 2e6 linear points to 200 um still truncated enough
    # tail to leave a 1.5e-4 error, while 2e5 log points to 1 cm give 4e-10.
    # A tolerance loosened to accommodate the linear version would have been a
    # test tuned to its own sampling error rather than to the physics.
    temperature = 500.0
    lam = np.logspace(-8, -2, 200_000)
    integrated = np.pi * _trapezoid(planck_radiance(lam, temperature), lam)
    sigma = 5.670374419e-8
    assert integrated == pytest.approx(sigma * temperature**4, rel=1e-8)


# --------------------------------------------------------------------------
# Units and refusals
# --------------------------------------------------------------------------


def test_celsius_temperature_is_refused():
    # 20 C passed as kelvin is merely wrong; -20 C is unphysical and catchable.
    with pytest.raises(DegenerateInputError, match="kelvin"):
        planck_radiance(10e-6, -20.0)


def test_zero_kelvin_is_refused():
    with pytest.raises(DegenerateInputError):
        planck_radiance(10e-6, 0.0)


def test_negative_wavelength_is_refused():
    with pytest.raises(DegenerateInputError, match="metres"):
        planck_radiance(-1e-6, 300.0)


def test_wavelength_in_micrometres_is_not_silently_accepted():
    # Passing 10 instead of 10e-6 asks for radiance at 10 metres. It does not
    # raise -- it cannot, 10 m is a valid wavelength -- so this documents the
    # magnitude of the mistake rather than claiming protection from it.
    assert planck_radiance(10.0, 300.0) < 1e-10
    assert planck_radiance(10e-6, 300.0) > 1e6


# --------------------------------------------------------------------------
# Inversion
# --------------------------------------------------------------------------


@pytest.mark.parametrize("temperature_k", [200.0, 300.0, 600.0, 1500.0])
def test_inversion_round_trips(temperature_k):
    radiance = planck_radiance(10e-6, temperature_k)
    assert inverse_planck_temperature(10e-6, radiance) == pytest.approx(
        temperature_k, rel=1e-9
    )


def test_inversion_is_vectorised():
    temps = np.array([250.0, 400.0, 900.0])
    back = inverse_planck_temperature(10e-6, planck_radiance(10e-6, temps))
    assert back == pytest.approx(temps, rel=1e-9)


def test_non_positive_radiance_is_refused():
    with pytest.raises(DegenerateInputError, match="positive"):
        inverse_planck_temperature(10e-6, 0.0)


# --------------------------------------------------------------------------
# Emissivity
# --------------------------------------------------------------------------


def test_unlisted_material_gets_a_range_not_a_point_estimate():
    estimate = EmissivityTable().get("unobtainium")
    assert estimate.is_assumed
    assert estimate.low < estimate.value < estimate.high


def test_listed_material_is_not_flagged_as_assumed():
    table = EmissivityTable({"concrete": {"emissivity": 0.92, "range": [0.90, 0.95]}})
    estimate = table.get("Concrete")  # case-insensitive
    assert estimate.value == 0.92
    assert not estimate.is_assumed


def test_missing_table_file_is_not_an_error(tmp_path):
    # Emissivity references are an optional supplement; absence means every
    # lookup is the flagged default, not a crash.
    table = EmissivityTable.load(tmp_path / "nope.json")
    assert table.get("steel").is_assumed


def test_table_loads_from_disk(tmp_path):
    path = tmp_path / "emissivity.json"
    path.write_text(json.dumps({"materials": {"steel": {"emissivity": 0.28,
                                                        "range": [0.1, 0.4]}}}))
    assert EmissivityTable.load(path).get("steel").value == 0.28


def test_emissivity_outside_the_unit_interval_is_refused():
    with pytest.raises(DegenerateInputError):
        EmissivityEstimate(1.4, 1.0, 1.5, "impossible", "table")


def test_range_must_contain_its_own_estimate():
    with pytest.raises(DegenerateInputError, match="does not contain"):
        EmissivityEstimate(0.95, 0.2, 0.5, "inconsistent", "table")


def test_default_must_lie_inside_its_stated_range():
    with pytest.raises(DegenerateInputError, match="outside its own stated range"):
        EmissivityTable(default=0.99, default_range=(0.85, 0.95))


def test_provenance_reaches_the_run_record():
    labels = EmissivityTable().get("unknown thing").as_labels()
    assert labels["emissivity_source"] == "default"
    assert "-" in labels["emissivity_range"]


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def test_emissivity_correction_raises_the_temperature():
    # A real surface emits less than a blackbody, so it must be HOTTER than the
    # camera's reading to produce the radiance measured.
    estimate = EmissivityEstimate(0.9, 0.85, 0.95, "test", "table")
    out = retrieve_temperature(400.0, estimate)
    assert out.temperature_k > out.brightness_temperature_k


def test_unit_emissivity_is_the_identity():
    estimate = EmissivityEstimate(1.0, 1.0, 1.0, "blackbody", "table")
    out = retrieve_temperature(400.0, estimate)
    assert out.temperature_k == pytest.approx(400.0, rel=1e-9)


def test_lower_emissivity_means_a_larger_correction():
    low = retrieve_temperature(400.0, EmissivityEstimate(0.5, 0.5, 0.5, "a", "t"))
    high = retrieve_temperature(400.0, EmissivityEstimate(0.95, 0.95, 0.95, "b", "t"))
    assert low.temperature_k > high.temperature_k


def test_error_bars_are_not_inverted():
    # Higher emissivity -> less correction -> LOWER temperature. Swapping these
    # would invert the bars while leaving the central estimate correct, which
    # nothing downstream would catch.
    estimate = EmissivityEstimate(0.9, 0.8, 1.0, "test", "table")
    out = retrieve_temperature(400.0, estimate)
    assert out.low_k < out.temperature_k < out.high_k


def test_assumed_emissivity_produces_a_visible_spread():
    # The point of §11 edge case 11: for an unlisted material the assumption is
    # most of the uncertainty, and the answer must show it.
    out = retrieve_temperature(400.0, EmissivityTable().get("mystery"))
    assert out.spread_k > 1.0


def test_retrieval_names_its_mode():
    out = retrieve_temperature(400.0, EmissivityTable().get("x"))
    assert "assumed-emissivity" in out.mode
    assert out.as_labels()["retrieval_mode"] == out.mode


def test_retrieval_is_vectorised_over_a_frame():
    field = np.full((4, 4), 400.0)
    out = retrieve_temperature(field, EmissivityEstimate(0.9, 0.85, 0.95, "t", "table"))
    assert np.asarray(out.temperature_k).shape == (4, 4)


def test_celsius_input_to_retrieval_is_refused():
    with pytest.raises(DegenerateInputError, match="to_kelvin"):
        retrieve_temperature(-10.0, EmissivityTable().get("x"))


# --------------------------------------------------------------------------
# The refusal path, end to end
# --------------------------------------------------------------------------


def test_planck_is_unreachable_from_a_display_product():
    # §11 edge case 9, joined up: a colormapped frame cannot reach Eq. 5 by any
    # route, because to_kelvin() is the only way in and it refuses first.
    from pipeline.errors import NonRadiometricInputError

    frame = ThermalFrame(
        data=np.full((4, 4), 200.0, dtype=np.float32),
        radiometry=Radiometry(False, "digital_number", "8-bit false colour"),
    )
    with pytest.raises(NonRadiometricInputError):
        retrieve_temperature(frame.to_kelvin(), EmissivityTable().get("steel"))


def test_radiometric_frame_reaches_planck_correctly():
    frame = ThermalFrame(
        data=np.full((2, 2), 300.0, dtype=np.float32),
        radiometry=Radiometry(True, "celsius", "float32 TIFF"),
    )
    out = retrieve_temperature(frame.to_kelvin(), EmissivityTable().get("unknown"))
    # 300 C is 573.15 K; an emissivity below 1 must push the estimate above it.
    assert np.all(np.asarray(out.temperature_k) > 573.15)
