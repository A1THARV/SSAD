"""Subset extraction from the ECOSTRESS library.

Most of these run against synthetic fixtures in the library's real format, so
the filters stay pinned without the 132 GB... 132 MB of real data. The few that
need the actual library skip when it is absent.

What is worth pinning here is the set of things that were *wrong* on the first
attempt, each of which would have produced a plausible-looking but incorrect
library: matching the wrong metadata field, matching against free-text prose,
attributing a specific material to a generic one, and deriving emissivity from
a geometry Kirchhoff does not apply to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from build_speclib import (  # noqa: E402
    EMISSIVITY,
    GENERIC_KEYWORDS,
    KIRCHHOFF_EXPECTATIONS,
    KNOWN_ABSENT,
    REFLECTANCE,
    SENSOR_BAND_UM,
    classify_measurement,
    fallback_note,
    is_hemispherical,
    load_library,
    parse_spectrum_file,
    select,
    to_emissivity,
    validate_kirchhoff,
    write_subset,
)

from pipeline.errors import DataUnavailableError  # noqa: E402

LIBRARY = Path("data/speclib/ecostress/ecospeclib-all")
needs_library = pytest.mark.skipif(
    not LIBRARY.is_dir(), reason="ECOSTRESS library absent; run build_speclib.py setup"
)


def _spectrum(tmp_path, name, *, measurement="Hemispherical reflectance",
              y_units="Reflectance (percent)", lo=7.5, hi=13.5, value=5.0,
              type_="vegetation", cls="Test", description="", filename=None):
    grid = np.linspace(lo, hi, 64)
    head = [
        f"Name: {name}", f"Type: {type_}", f"Class: {cls}",
        f"Measurement: {measurement}", f"Y Units: {y_units}",
        f"First X Value: {lo}", f"Last X Value: {hi}",
    ]
    if description:
        head.append(f"Description: {description}")
    body = "\n".join(f"  {w:.4f}\t{value:.4f}" for w in grid)
    path = tmp_path / (filename or f"{name.replace(' ', '_')}.spectrum.txt")
    path.write_text("\n".join(head) + "\n\n" + body + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Measurement classification -- the field that was wrong first time
# --------------------------------------------------------------------------


def test_measurement_comes_from_y_units_not_type():
    # `Type:` is the MATERIAL category (Mineral, vegetation, manmade). An
    # earlier version filtered on it for "reflectance", matched nothing, and
    # would have written two empty subsets while reporting "kept 0" -- which
    # reads as a library deficiency rather than a filter bug.
    meta = {"Type": "vegetation", "Y Units": "Reflectance (percent)"}
    assert classify_measurement(meta) == "reflectance"
    assert classify_measurement({"Type": "reflectance"}) == "unknown"


@pytest.mark.parametrize(
    "spelling", ["Reflectance (percent)", "Reflectance (percentage)", "Reflectence (percent)"]
)
def test_reflectance_typos_are_tolerated(spelling):
    # All three spellings occur in the real library.
    assert classify_measurement({"Y Units": spelling}) == "reflectance"


@pytest.mark.parametrize(
    "spelling",
    ["Transmittance (percent)", "Transmittence (percent)", "Transittance (percent)",
     "Transmission (percent)"],
)
def test_transmittance_is_distinguished(spelling):
    # Not a pedantic distinction: eps = 1 - R - T for a transmitting sample, so
    # running these through 1 - R would silently overstate emissivity.
    assert classify_measurement({"Y Units": spelling}) == "transmittance"


def test_hemispherical_variants_are_recognised():
    for m in ("Hemispherical reflectance", "Directional (10 Degree) Hemispherical Reflectance",
              "Directional hemispherical reflectance"):
        assert is_hemispherical({"Measurement": m})
    assert not is_hemispherical({"Measurement": "Bidirectional reflectance"})


# --------------------------------------------------------------------------
# Kirchhoff
# --------------------------------------------------------------------------


def test_kirchhoff_inverts_reflectance():
    assert to_emissivity(np.array([0.0, 0.05, 1.0])) == pytest.approx([1.0, 0.95, 0.0])


def test_emissivity_is_clipped_into_the_unit_interval():
    # Calibration noise can push a measured reflectance just outside [0, 1];
    # an emissivity outside it is unphysical and EmissivityEstimate refuses it.
    assert to_emissivity(np.array([-0.02, 1.03])) == pytest.approx([1.0, 0.0])


def test_bidirectional_entries_are_excluded_from_the_emissivity_subset(tmp_path):
    # Kirchhoff relates DIRECTIONAL emissivity to directional absorptivity, so
    # bidirectional reflectance does not qualify.
    _spectrum(tmp_path, "Hemi grass", measurement="Hemispherical reflectance")
    _spectrum(tmp_path, "Bidir grass", measurement="Bidirectional reflectance")
    kept, dropped = select(load_library(tmp_path), EMISSIVITY)
    assert [s.name for s in kept] == ["Hemi grass"]
    assert dropped["not_hemispherical"] == 1


def test_transmitting_samples_are_excluded(tmp_path):
    _spectrum(tmp_path, "Leafy grass", y_units="Transmittance (percent)")
    kept, dropped = select(load_library(tmp_path), EMISSIVITY)
    assert kept == [] and dropped["wrong_measurement"] == 1


# --------------------------------------------------------------------------
# Matching -- the two bugs the validation exposed
# --------------------------------------------------------------------------


def test_description_is_not_matched_against(tmp_path):
    # The case that proved it: two entries matched "aluminum" and derived
    # eps = 0.937, wildly wrong for a metal, because they are black gloss PAINT
    # ON an aluminium substrate. The word described the mounting.
    _spectrum(
        tmp_path, "Black gloss paint", type_="manmade", cls="Paint",
        description="A single coat of alkyd gloss paint on an aluminum surface.",
    )
    kept, _ = select(load_library(tmp_path), EMISSIVITY)
    assert kept == [], "Description must not supply keyword matches"


def test_specific_material_is_not_swallowed_by_a_generic_one(tmp_path):
    # ECOSTRESS files bark under `nonphotosyntheticvegetation.bark.*`, so a
    # first-match-wins scan with "vegetation" ahead of "bark" attributed all 18
    # bark entries to vegetation and reported bark as ABSENT -- a material that
    # is present being reported missing.
    _spectrum(
        tmp_path, "Abies concolor bark", cls="Bark",
        filename="nonphotosyntheticvegetation.bark.abies.concolor.tir.vh311.spectrum.txt",
    )
    kept, dropped = select(load_library(tmp_path), EMISSIVITY)
    assert kept[0].metadata["_matched_keyword"] == "bark"
    assert "bark" not in dropped["empty_keywords"]


def test_generic_keywords_are_declared_not_inferred():
    assert "vegetation" in GENERIC_KEYWORDS
    # Every generic keyword must appear after every specific one in the list
    # the subset actually uses, or the ordering guarantee is vacuous.
    specific = [k for k in EMISSIVITY.keywords if k not in GENERIC_KEYWORDS]
    generic = [k for k in EMISSIVITY.keywords if k in GENERIC_KEYWORDS]
    assert EMISSIVITY.keywords == specific + generic


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_entry_not_spanning_the_sensor_band_is_dropped(tmp_path):
    _spectrum(tmp_path, "Short grass", lo=8.0, hi=12.0)
    kept, dropped = select(load_library(tmp_path), EMISSIVITY)
    assert kept == []
    assert dropped["insufficient_coverage"][0]["coverage_um"] == [8.0, 12.0]


def test_entry_spanning_exactly_the_sensor_band_is_kept(tmp_path):
    _spectrum(tmp_path, "Exact grass", lo=7.5, hi=13.5)
    kept, dropped = select(load_library(tmp_path), EMISSIVITY)
    assert len(kept) == 1 and dropped["insufficient_coverage"] == []


def test_emissivity_coverage_matches_the_sensor_band_not_a_wider_window():
    # Requiring 8-14 um would reject a 7.5-13.5 entry for lacking wavelengths
    # FLAME 3's camera never observes, and the rejection would appear in the
    # coverage-drop list looking like a library property.
    assert EMISSIVITY.coverage_um == SENSOR_BAND_UM == (7.5, 13.5)


def test_configured_ranges_have_not_been_accidentally_overlapped():
    # A CONFIG guard, nothing more. It does NOT encode "one library cannot serve
    # both": the ranges are disjoint because they were configured that way, and
    # 54 entries in the real library span 0.4-13.5 um continuously. Kirchhoff
    # makes emissivity and reflectance complementary, so asserting they must be
    # disjoint would state the opposite of the physics.
    assert REFLECTANCE.coverage_um[1] <= EMISSIVITY.coverage_um[0], (
        "The two subsets' ranges now overlap -- not wrong in itself, but worth "
        "doing deliberately rather than by accident."
    )


# --------------------------------------------------------------------------
# Absences
# --------------------------------------------------------------------------


def test_absent_materials_carry_their_reasoning():
    assert "ash" in KNOWN_ABSENT and "steel" in KNOWN_ABSENT
    note = fallback_note(["ash"])[0]
    assert "active wildfire" in note.lower()
    assert "cite a published value" in note


def test_undocumented_absence_says_so():
    assert "undocumented" in fallback_note(["unobtainium"])[0]


def test_empty_keywords_are_reported(tmp_path):
    _spectrum(tmp_path, "Some grass")
    _, dropped = select(load_library(tmp_path), EMISSIVITY)
    assert "char" in dropped["empty_keywords"]
    assert dropped["per_keyword"]["grass"] == 1


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def test_emissivity_subset_is_written_as_emissivity(tmp_path):
    _spectrum(tmp_path, "Grass A", value=5.0)
    kept, _ = select(load_library(tmp_path), EMISSIVITY)
    blob = np.load(write_subset(kept, EMISSIVITY, tmp_path / "out"), allow_pickle=True)
    assert str(blob["quantity"]) == "emissivity"
    assert str(blob["derivation"]) == "kirchhoff"
    # 5% reflectance -> 0.95 emissivity.
    assert blob["spectra"].mean() == pytest.approx(0.95, abs=1e-6)


def test_reflectance_subset_is_left_as_measured(tmp_path):
    _spectrum(tmp_path, "Quartz sample", lo=0.4, hi=2.5, value=30.0,
              type_="Mineral", cls="Quartz")
    kept, _ = select(load_library(tmp_path), REFLECTANCE)
    blob = np.load(write_subset(kept, REFLECTANCE, tmp_path / "out"), allow_pickle=True)
    assert str(blob["quantity"]) == "reflectance"
    assert blob["spectra"].mean() == pytest.approx(0.30, abs=1e-6)


def test_ancillary_files_are_skipped(tmp_path):
    _spectrum(tmp_path, "Real grass")
    (tmp_path / "mineral.arsenate.none.coarse.tir.a-1a.ancillary.txt").write_text(
        "Name: Mimetite\nType: Mineral\nXRD Analysis:  Mimetite(A)\n", encoding="utf-8"
    )
    assert [s.name for s in load_library(tmp_path)] == ["Real grass"]


def test_missing_library_names_the_remedy(tmp_path):
    with pytest.raises(DataUnavailableError, match="speclib.jpl.nasa.gov"):
        load_library(tmp_path / "absent")


def test_directory_without_spectra_is_refused(tmp_path):
    (tmp_path / "readme.txt").write_text("nothing here", encoding="utf-8")
    with pytest.raises(DataUnavailableError, match="spectrum.txt"):
        load_library(tmp_path)


def test_file_with_headers_but_no_data_is_skipped(tmp_path):
    path = tmp_path / "empty.spectrum.txt"
    path.write_text("Name: Nothing\nType: Mineral\n", encoding="utf-8")
    assert parse_spectrum_file(path) is None


# --------------------------------------------------------------------------
# Against the real library
# --------------------------------------------------------------------------


@needs_library
def test_kirchhoff_validation_passes_on_the_real_library():
    # The check that caught the paint-as-aluminium false match. If a material
    # lands outside its band, the hemispherical filter is admitting something
    # it should not -- so this failing is informative, not flaky.
    results = validate_kirchhoff(load_library(LIBRARY), SENSOR_BAND_UM)
    outside = {m: r for m, r in results.items() if r.get("status") == "OUT OF RANGE"}
    assert not outside, f"derived emissivity outside expected range: {outside}"


@needs_library
def test_every_validation_material_is_actually_present():
    # A validation that silently matches nothing proves nothing.
    results = validate_kirchhoff(load_library(LIBRARY), SENSOR_BAND_UM)
    missing = [m for m, r in results.items() if r["n"] == 0]
    assert not missing, f"no entries found for {missing}; the check is vacuous for them"
    assert set(results) == set(KIRCHHOFF_EXPECTATIONS)


@needs_library
def test_the_library_holds_no_emissivity_spectra():
    # The fact that makes Kirchhoff necessary rather than optional. If this ever
    # fails, a direct measurement became available and should be preferred.
    kinds = {classify_measurement(s.metadata) for s in load_library(LIBRARY)}
    assert "emissivity" not in kinds
