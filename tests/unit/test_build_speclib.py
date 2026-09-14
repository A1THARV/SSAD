"""Subset extraction from ECOSTRESS, tested on a synthetic stand-in.

The real ``spectra_all.txt`` is a pending manual order, so these run against a
fixture in the same format. That is the right level anyway: what needs pinning
is the coverage check and the two-subset split, neither of which depends on the
real file's size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from build_speclib import (  # noqa: E402
    EMISSIVITY,
    REFLECTANCE,
    parse_ecostress,
    select,
    write_subset,
)

from pipeline.errors import DataUnavailableError  # noqa: E402


def _record(name, type_, lo_um, hi_um, n=64, value=30.0):
    grid = np.linspace(lo_um, hi_um, n)
    body = "\n".join(f"  {w:.4f}  {value:.4f}" for w in grid)
    return f"Name: {name}\nType: {type_}\nClass: Test\n\n{body}\n"


def _library(tmp_path: Path, records: list[str]) -> Path:
    path = tmp_path / "spectra_all.txt"
    path.write_text("\n".join(records), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parses_multiple_records(tmp_path):
    path = _library(tmp_path, [
        _record("Kaolinite KGa-1", "Reflectance", 0.35, 2.6),
        _record("Dry grass", "Emissivity", 7.5, 14.5),
    ])
    spectra = parse_ecostress(path)
    assert [s.name for s in spectra] == ["Kaolinite KGa-1", "Dry grass"]


def test_percent_is_converted_to_physical_units(tmp_path):
    path = _library(tmp_path, [_record("Kaolinite", "Reflectance", 0.4, 2.5, value=45.0)])
    assert parse_ecostress(path)[0].value[0] == pytest.approx(0.45)


def test_missing_source_names_the_order_page(tmp_path):
    with pytest.raises(DataUnavailableError, match="speclib.jpl.nasa.gov"):
        parse_ecostress(tmp_path / "absent.txt")


# --------------------------------------------------------------------------
# The coverage check -- the reason this runs at build time
# --------------------------------------------------------------------------


def test_entry_not_spanning_the_required_range_is_dropped(tmp_path):
    # An LWIR-only entry cannot serve the reflectance subset. Keeping it would
    # push a refusal into the middle of SAM matching instead.
    path = _library(tmp_path, [
        _record("Kaolinite good", "Reflectance", 0.35, 2.6),
        _record("Kaolinite truncated", "Reflectance", 0.4, 1.8),
    ])
    kept, dropped = select(parse_ecostress(path), REFLECTANCE)
    assert [s.name for s in kept] == ["Kaolinite good"]
    assert dropped["insufficient_coverage"][0]["name"] == "Kaolinite truncated"


def test_dropped_entries_report_their_actual_coverage(tmp_path):
    # "Dropped" alone is not actionable; the span says why.
    path = _library(tmp_path, [_record("Calcite short", "Reflectance", 0.4, 1.0)])
    _, dropped = select(parse_ecostress(path), REFLECTANCE)
    lo, hi = dropped["insufficient_coverage"][0]["coverage_um"]
    assert lo == pytest.approx(0.4) and hi == pytest.approx(1.0)


def test_coverage_is_checked_against_each_subset_separately(tmp_path):
    # The same library serves two subsets wanting opposite ends of the spectrum.
    path = _library(tmp_path, [
        _record("Kaolinite", "Reflectance", 0.35, 2.6),
        _record("Dry grass", "Emissivity", 7.5, 14.5),
    ])
    spectra = parse_ecostress(path)
    assert [s.name for s in select(spectra, REFLECTANCE)[0]] == ["Kaolinite"]
    assert [s.name for s in select(spectra, EMISSIVITY)[0]] == ["Dry grass"]


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def test_wrong_measurement_type_is_excluded(tmp_path):
    path = _library(tmp_path, [_record("Dry grass", "Reflectance", 7.5, 14.5)])
    kept, dropped = select(parse_ecostress(path), EMISSIVITY)
    assert kept == [] and dropped["wrong_type"] == 1


def test_unrelated_material_is_excluded(tmp_path):
    path = _library(tmp_path, [_record("Polystyrene", "Emissivity", 7.5, 14.5)])
    kept, dropped = select(parse_ecostress(path), EMISSIVITY)
    assert kept == [] and dropped["no_keyword"] == 1


def test_counterexamples_are_kept_and_flagged(tmp_path):
    # Steel is deliberately retained: it is what produces the 307 K
    # across-class error in RESULTS.md §13, so the library must carry it.
    path = _library(tmp_path, [
        _record("Dry grass", "Emissivity", 7.5, 14.5),
        _record("Stainless steel", "Emissivity", 7.5, 14.5),
    ])
    kept, _ = select(parse_ecostress(path), EMISSIVITY)
    flags = {s.name: s.metadata["_is_counterexample"] for s in kept}
    assert flags == {"Dry grass": "false", "Stainless steel": "true"}


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def test_subset_is_written_on_a_common_grid(tmp_path):
    path = _library(tmp_path, [
        _record("Dry grass", "Emissivity", 7.5, 14.5),
        _record("Charred wood", "Emissivity", 6.0, 15.0),
    ])
    kept, _ = select(parse_ecostress(path), EMISSIVITY)
    blob = np.load(write_subset(kept, EMISSIVITY, tmp_path / "out"), allow_pickle=True)

    assert blob["spectra"].shape == (2, blob["wavelength_um"].size)
    # Every kept entry spans the grid by construction, so resampling onto it
    # interpolates and never extrapolates.
    assert blob["wavelength_um"].min() >= EMISSIVITY.coverage_um[0]
    assert blob["wavelength_um"].max() <= EMISSIVITY.coverage_um[1]


def test_empty_subset_writes_a_well_formed_file(tmp_path):
    blob = np.load(write_subset([], EMISSIVITY, tmp_path / "out"), allow_pickle=True)
    assert blob["spectra"].shape[0] == 0
    assert blob["wavelength_um"].size > 0


def test_the_two_subsets_target_different_windows():
    # The whole reason for two subsets: SAM needs SWIR, Eq. 5 needs LWIR, and
    # no single spectrum is guaranteed to span both.
    assert REFLECTANCE.coverage_um[1] <= EMISSIVITY.coverage_um[0]
