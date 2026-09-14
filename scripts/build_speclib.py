"""Extract two curated subsets from the ECOSTRESS spectral library.

    python scripts/build_speclib.py

One download, two purposes, because M5 made a second subset load-bearing:

* **Reflectance, for Objective 2 (SAM).** Minerals carrying diagnostic SWIR
  absorption features -- clays, carbonates, sulfates. Sized for a
  SWIR-diagnostic scene per RESULTS.md §3, *not* for HAD100, whose signal sits
  at 455-700 nm where a spectral library has little to say.
* **LWIR emissivity, for Objective 4 (Eq. 5).** Vegetation, char, soil, bark --
  the wildfire material class -- plus a metal kept deliberately as the
  out-of-class counterexample behind the large retrieval error in §13.

The split is **pragmatic, not physical**: the two objectives need different
wavelength ranges and material sets. It is not that one spectrum could not serve
both -- 54 entries in this library span 0.4-13.5 um continuously.

What the library actually contains
----------------------------------
Established by survey before writing any of this, and it differs from the
documentation in ways that matter:

* It is a **directory of one file per spectrum**, not a single concatenated
  file: 3,451 ``*.spectrum.txt`` plus 2,692 ``*.ancillary.txt`` metadata
  companions that hold no spectra and must be skipped.
* ``Type:`` is the **material category** (Mineral, vegetation, Rock, soil,
  manmade), *not* the measurement type. An earlier version of this script
  filtered on it for "reflectance"/"emissivity", which matches nothing -- and
  would have written two empty subsets while reporting "kept 0", reading as a
  library deficiency rather than a filter bug. The measurement is in
  ``Y Units:`` and ``Measurement:``.
* **There are no emissivity spectra at all.** Every entry is reflectance
  (3,323) or transmittance (128). Emissivity is therefore derived, not read --
  see below.
* Field values are inconsistently cased (``vegetation``/``Vegetation``) and the
  units field carries typos (``Reflectence``, ``Transmittence``,
  ``Transittance``), so matching is normalised rather than exact.

Emissivity by Kirchhoff's law
-----------------------------
For an opaque surface at local thermodynamic equilibrium::

    epsilon(lambda) = 1 - R(lambda)

This is how ASTER and the LST&E literature derive emissivity routinely, so it
is sound rather than a compromise -- but it is an **inference from a published
measurement**, not a measurement, and is labelled that way everywhere it
surfaces.

Two filters the relation requires, both enforced here:

* **Hemispherical geometry only.** Kirchhoff relates directional emissivity to
  directional absorptivity, so bidirectional reflectance (892 entries) does not
  qualify and is excluded.
* **Opaque samples only.** For a transmitting sample eps = 1 - R - T, so the
  128 transmittance entries are excluded rather than run through 1 - R.

:func:`validate_kirchhoff` checks the conversion against materials whose LWIR
emissivity is well established, so "standard practice" becomes "verified on this
library".

Temperature, as a stated limitation
-----------------------------------
These are **laboratory spectra at ambient temperature**, applied to fire pixels
at 220-600 C. LWIR emissivity is reasonably stable with temperature for most of
these materials, but **char and soot during active combustion are the least
certain case, and char is the primary material** in a FLAME 3 scene. Recorded
in RESULTS.md §13 as a limitation rather than left implicit.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import numpy as np  # noqa: E402

from pipeline.errors import DataUnavailableError, DegenerateInputError  # noqa: E402

# --------------------------------------------------------------------------
# Library conventions, established by survey
# --------------------------------------------------------------------------

_HEADER = re.compile(r"^\s*([A-Za-z][A-Za-z .()/]*?)\s*:\s*(.*?)\s*$")

#: Only these geometries satisfy Kirchhoff. Matched as substrings of a
#: lowercased ``Measurement:`` field, which also covers the "Directional
#: (10 Degree) Hemispherical" and "Directional Hemispherical" variants.
HEMISPHERICAL = ("hemispher",)

#: Reflectance spellings present in the library, typos included.
_REFLECTANCE_TOKENS = ("reflectance", "reflectence", "reflect")
#: Transmittance spellings, likewise. These mark non-opaque samples.
_TRANSMITTANCE_TOKENS = ("transmittance", "transmittence", "transittance",
                         "transmission", "transmit")


def classify_measurement(meta: dict[str, str]) -> str:
    """``reflectance``, ``transmittance`` or ``unknown`` from the real fields.

    Reads ``Y Units`` and ``Measurement``, never ``Type`` -- ``Type`` is the
    material category, and using it here is the bug this function exists to
    make impossible to repeat.
    """
    blob = f"{meta.get('Y Units', '')} {meta.get('Measurement', '')}".lower()
    if any(t in blob for t in _TRANSMITTANCE_TOKENS):
        return "transmittance"
    if any(t in blob for t in _REFLECTANCE_TOKENS):
        return "reflectance"
    return "unknown"


def is_hemispherical(meta: dict[str, str]) -> bool:
    return any(t in meta.get("Measurement", "").lower() for t in HEMISPHERICAL)


# --------------------------------------------------------------------------
# What each subset needs
# --------------------------------------------------------------------------


@dataclass
class SubsetSpec:
    """One curated subset: what to keep, and the grid it must cover."""

    name: str
    purpose: str
    #: Required wavelength span in micrometres. An entry not covering it is
    #: dropped with a reason rather than extrapolated.
    coverage_um: tuple[float, float]
    keywords: list[str]
    #: ``reflectance`` throughout -- the library holds nothing else.
    measurement: str = "reflectance"
    #: ``kirchhoff`` converts reflectance to emissivity; ``None`` keeps it as is.
    derive: str | None = None
    #: Kirchhoff needs hemispherical geometry; SAM does not care.
    require_hemispherical: bool = False
    counterexamples: list[str] = field(default_factory=list)


#: SWIR diagnostics: OH/carbonate near 2.2-2.35 um, water near 1.4/1.9 um.
REFLECTANCE = SubsetSpec(
    name="reflectance_swir",
    purpose="Objective 2 -- SAM material matching on a SWIR-diagnostic scene",
    coverage_um=(0.4, 2.5),
    keywords=[
        "kaolinite", "montmorillonite", "illite", "muscovite", "chlorite",
        "calcite", "dolomite", "gypsum", "alunite", "jarosite",
        "hematite", "goethite", "quartz", "feldspar", "serpentine",
    ],
)

#: FLAME 3's thermal band, per the paper's Fig. 3. The requirement is the
#: SENSOR's band, deliberately not a wider nominal "LWIR" window: an entry
#: measured over exactly 7.5-13.5 um covers everything this camera sees, and
#: requiring 8-14 would reject it for lacking wavelengths no instrument here
#: observes -- a rejection that would appear in the coverage-drop list looking
#: like a property of the library rather than a mistake in the filter.
SENSOR_BAND_UM = (7.5, 13.5)

EMISSIVITY = SubsetSpec(
    name="emissivity_lwir",
    purpose="Objective 4 -- the epsilon prior for Eq. 5, via Kirchhoff",
    coverage_um=SENSOR_BAND_UM,
    measurement="reflectance",
    derive="kirchhoff",
    require_hemispherical=True,
    keywords=["char", "grass", "conifer", "deciduous", "leaf", "bark",
              "soil", "sand", "vegetation", "wood"],
    # Steel is ABSENT from this library at LWIR coverage, and so is metallic
    # aluminium -- the only "aluminum" matches were paint on an aluminium
    # substrate. Brass and gold ARE present and measured, at eps ~ 0.10, which
    # is *lower* than steel's assumed 0.28. The out-of-class retrieval error is
    # therefore larger than the figure §13 first reported, so that number is
    # RE-MEASURED from these spectra rather than carried across.
    counterexamples=["brass", "gold"],
)

#: Catch-all categories that subsume more specific ones. Tested last, so a
#: bark or leaf spectrum is attributed to bark or leaf rather than swallowed.
GENERIC_KEYWORDS = {"vegetation", "wood"}

SUBSETS = [REFLECTANCE, EMISSIVITY]

#: Materials requested but absent, with the reason it is acceptable and what to
#: do instead. Ash is the only genuine gap after the survey.
KNOWN_ABSENT = {
    "ash": (
        "No LWIR-capable entry. Acceptable here: FLAME 3 is ACTIVE wildfire -- "
        "char, flame and unburned vegetation -- whereas ash dominates post-burn "
        "scenes, which is not what this work scores against. If ash is ever "
        "needed, cite a published value explicitly rather than substituting "
        "vegetation; green canopy (~0.98) is not a stand-in for ash."
    ),
    "steel": (
        "No LWIR-capable entry, and neither has metallic aluminium -- its only "
        "matches were paint on an aluminium substrate. BRASS substitutes as the "
        "out-of-class counterexample: present, measured, hemispherical, and at "
        "eps ~ 0.10 it is lower than steel's assumed 0.28, so the retrieval error "
        "it produces is LARGER. §13's figure is re-measured from these spectra "
        "rather than inherited."
    ),
}

#: Materials whose LWIR emissivity is well established, used to check that the
#: Kirchhoff conversion lands where physics says it should. If one falls
#: outside, the hemispherical filter is admitting something it should not.
KIRCHHOFF_EXPECTATIONS = {
    "water": (0.96, 1.00),
    "vegetation": (0.94, 1.00),
    "concrete": (0.85, 0.97),
    # Polished metal: the only low-emissivity class in this library, and the
    # one that exercises the bottom of the range. "aluminum" was tried first and
    # failed validation at 0.937 -- correctly, because its only matches were
    # paint on an aluminium substrate. That failure is why Description is no
    # longer matched against.
    "brass": (0.02, 0.30),
}


def fallback_note(missing: list[str]) -> list[str]:
    notes = []
    for name in missing:
        reason = KNOWN_ABSENT.get(name)
        notes.append(
            f"{name!r}: {reason}" if reason
            else f"{name!r} absent and undocumented -- widen the keywords or record why."
        )
    return notes


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


@dataclass
class Spectrum:
    name: str
    metadata: dict[str, str]
    wavelength_um: np.ndarray
    value: np.ndarray
    source_file: str = ""

    @property
    def coverage(self) -> tuple[float, float]:
        return float(self.wavelength_um.min()), float(self.wavelength_um.max())

    def covers(self, span: tuple[float, float]) -> bool:
        lo, hi = self.coverage
        return lo <= span[0] and hi >= span[1]


def parse_spectrum_file(path: Path) -> Spectrum | None:
    """Read one ``*.spectrum.txt``. Returns ``None`` if it holds no data."""
    meta: dict[str, str] = {}
    rows: list[tuple[float, float]] = []

    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) == 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                    continue
                except ValueError:
                    pass
            header = _HEADER.match(stripped)
            if header:
                meta[header.group(1).strip()] = header.group(2).strip()

    if len(rows) < 2:
        return None

    arr = np.array(sorted(rows), dtype=np.float64)
    # Y is percent throughout this library, for both reflectance and
    # transmittance, so subsets are written in physical units rather than
    # percent -- Kirchhoff's 1 - R is only meaningful on a fraction.
    return Spectrum(
        name=meta.get("Name", path.stem),
        metadata=meta,
        wavelength_um=arr[:, 0],
        value=arr[:, 1] / 100.0,
        source_file=path.name,
    )


def load_library(root: Path) -> list[Spectrum]:
    """Read every spectrum under ``root``, skipping ancillary metadata files."""
    if not root.is_dir():
        raise DataUnavailableError(
            reason="ECOSTRESS library directory not found",
            path=root,
            remedy=(
                "Order the library from https://speclib.jpl.nasa.gov/download and "
                "extract its 'ecospeclib-all' folder here. Drop the __MACOSX folder; "
                "it is archive cruft."
            ),
        )
    paths = sorted(root.rglob("*.spectrum.txt"))
    if not paths:
        raise DataUnavailableError(
            reason="No '*.spectrum.txt' files found",
            path=root,
            remedy=(
                "The bundle should contain ~3,451 *.spectrum.txt files alongside "
                "*.ancillary.txt metadata companions. Check the extraction completed."
            ),
        )
    out = [s for s in (parse_spectrum_file(p) for p in paths) if s is not None]
    if not out:
        raise DegenerateInputError(f"{len(paths)} files found but none held spectra.")
    return out


#: Structured fields only. ``Description`` is deliberately EXCLUDED: it is prose
#: that mentions substrates, mounting and provenance, so matching against it
#: produces false positives that look authoritative. The case that proved it --
#: two entries matched "aluminum" and derived eps = 0.937, wildly wrong for a
#: metal, because they are *black gloss paint on an aluminium substrate*. The
#: word described what the sample was mounted on, not what was measured.
_MATCH_FIELDS = ("Name", "Class", "Subclass", "Genus", "Species", "Mineral")


def _haystack(spectrum: Spectrum) -> str:
    """Text a keyword may match against: filename taxonomy plus typed fields."""
    return " ".join(
        [spectrum.name, spectrum.source_file,
         *(spectrum.metadata.get(k, "") for k in _MATCH_FIELDS)]
    ).lower()


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def select(spectra: list[Spectrum], spec: SubsetSpec) -> tuple[list[Spectrum], dict]:
    """Apply a subset's filters, reporting what was dropped and why."""
    # Specific keywords are tested before generic ones, and that ordering is
    # load-bearing rather than cosmetic. ECOSTRESS files a tree's bark under
    # `nonphotosyntheticvegetation.bark.*`, so a first-match-wins scan with
    # "vegetation" ahead of "bark" attributes all 18 bark entries to vegetation
    # and reports bark as ABSENT -- a missing material that is actually present,
    # which is the worst way to be wrong here.
    specific = [k for k in spec.keywords if k not in GENERIC_KEYWORDS]
    generic = [k for k in spec.keywords if k in GENERIC_KEYWORDS]
    terms = (
        [(k, True) for k in spec.counterexamples]
        + [(k, False) for k in specific]
        + [(k, False) for k in generic]
    )
    kept: list[Spectrum] = []
    dropped = {
        "wrong_measurement": 0,
        "not_hemispherical": 0,
        "no_keyword": 0,
        "insufficient_coverage": [],
    }
    per_keyword = {k: 0 for k, _ in terms}

    for s in spectra:
        if classify_measurement(s.metadata) != spec.measurement:
            dropped["wrong_measurement"] += 1
            continue
        if spec.require_hemispherical and not is_hemispherical(s.metadata):
            dropped["not_hemispherical"] += 1
            continue
        text = _haystack(s)
        match = next((k for k, _ in terms if k in text), None)
        if match is None:
            dropped["no_keyword"] += 1
            continue
        if not s.covers(spec.coverage_um):
            lo, hi = s.coverage
            dropped["insufficient_coverage"].append(
                {"name": s.name, "coverage_um": [round(lo, 3), round(hi, 3)]}
            )
            continue
        s.metadata["_matched_keyword"] = match
        s.metadata["_is_counterexample"] = str(
            any(k == match for k in spec.counterexamples)
        ).lower()
        per_keyword[match] += 1
        kept.append(s)

    dropped["per_keyword"] = per_keyword
    dropped["empty_keywords"] = sorted(k for k, n in per_keyword.items() if n == 0)
    return kept, dropped


def to_emissivity(reflectance: np.ndarray) -> np.ndarray:
    """Kirchhoff's law: eps = 1 - R, for an opaque surface at LTE.

    Input is a fraction, not a percentage. Values are clipped into [0, 1]: a
    measured reflectance can exceed 1 or fall below 0 by a hair through
    calibration noise, and an emissivity outside the unit interval is
    unphysical -- :class:`EmissivityEstimate` would refuse it downstream.
    """
    return np.clip(1.0 - np.asarray(reflectance, dtype=np.float64), 0.0, 1.0)


def validate_kirchhoff(spectra: list[Spectrum], band: tuple[float, float]) -> dict:
    """Check derived emissivity against materials with well-known LWIR values.

    Turns "standard practice" into "verified on this library". A material
    landing outside its expected band means the hemispherical filter is
    admitting something it should not -- a bidirectional entry, or a
    transmitting sample -- rather than that Kirchhoff is wrong.
    """
    results = {}
    for material, (lo_expect, hi_expect) in KIRCHHOFF_EXPECTATIONS.items():
        values = []
        for s in spectra:
            if classify_measurement(s.metadata) != "reflectance":
                continue
            if not is_hemispherical(s.metadata) or not s.covers(band):
                continue
            if material not in _haystack(s):
                continue
            mask = (s.wavelength_um >= band[0]) & (s.wavelength_um <= band[1])
            if mask.any():
                values.append(float(np.mean(to_emissivity(s.value[mask]))))
        if not values:
            results[material] = {"n": 0, "status": "no entries"}
            continue
        mean = float(np.mean(values))
        results[material] = {
            "n": len(values),
            "mean_emissivity": round(mean, 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "expected": [lo_expect, hi_expect],
            "status": "ok" if lo_expect <= mean <= hi_expect else "OUT OF RANGE",
        }
    return results


def write_subset(kept: list[Spectrum], spec: SubsetSpec, out_dir: Path) -> Path:
    """Save one subset as a single ``.npz`` on a common grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.npz"

    # A shared grid inside the subset, so SAM resamples once at load rather than
    # per comparison. Bounds are the subset's required coverage, which every
    # kept entry spans by construction -- so this interpolates, never
    # extrapolates.
    grid = np.linspace(*spec.coverage_um, 512)
    if kept:
        rows = [np.interp(grid, s.wavelength_um, s.value) for s in kept]
        stacked = np.vstack(rows)
        if spec.derive == "kirchhoff":
            stacked = to_emissivity(stacked)
    else:
        stacked = np.zeros((0, grid.size))

    np.savez_compressed(
        path,
        wavelength_um=grid,
        spectra=stacked,
        names=np.array([s.name for s in kept], dtype=object),
        source_files=np.array([s.source_file for s in kept], dtype=object),
        counterexample=np.array(
            [s.metadata.get("_is_counterexample", "false") == "true" for s in kept]
        ),
        quantity="emissivity" if spec.derive == "kirchhoff" else "reflectance",
        derivation=spec.derive or "none",
        purpose=spec.purpose,
    )
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="data/speclib/ecostress/ecospeclib-all")
    p.add_argument("--out", default="data/speclib/curated")
    p.add_argument("--report", default="results/speclib_build.json")
    args = p.parse_args(argv)

    source = Path(args.source)
    print(f"reading {source} ...", flush=True)
    spectra = load_library(source)
    kinds = {}
    for s in spectra:
        kinds[classify_measurement(s.metadata)] = kinds.get(
            classify_measurement(s.metadata), 0
        ) + 1
    print(f"  parsed {len(spectra):,} spectra  {kinds}\n")

    print("Kirchhoff validation (eps = 1 - R against well-known LWIR values)")
    validation = validate_kirchhoff(spectra, SENSOR_BAND_UM)
    for material, row in validation.items():
        if row["n"] == 0:
            print(f"  {material:<12} no entries")
            continue
        flag = "" if row["status"] == "ok" else "   <-- " + row["status"]
        print(
            f"  {material:<12} eps {row['mean_emissivity']:.3f} "
            f"(n={row['n']}, range {row['min']:.3f}-{row['max']:.3f}), "
            f"expected {row['expected'][0]}-{row['expected'][1]}{flag}"
        )
    print()

    report = {
        "source": str(source),
        "n_parsed": len(spectra),
        "measurement_kinds": kinds,
        "kirchhoff_validation": validation,
        "subsets": {},
    }

    for spec in SUBSETS:
        kept, dropped = select(spectra, spec)
        out = write_subset(kept, spec, Path(args.out))
        n_counter = sum(s.metadata.get("_is_counterexample") == "true" for s in kept)
        print(f"{spec.name}  ({spec.purpose})")
        print(f"  coverage required  : {spec.coverage_um[0]}-{spec.coverage_um[1]} um")
        print(f"  derivation         : {spec.derive or 'none (reflectance as measured)'}")
        print(f"  kept               : {len(kept)}  ({n_counter} counterexample)")
        print(f"  dropped, measurement: {dropped['wrong_measurement']:,}")
        if spec.require_hemispherical:
            print(f"  dropped, geometry  : {dropped['not_hemispherical']:,} "
                  "(bidirectional -- Kirchhoff needs hemispherical)")
        print(f"  dropped, keyword   : {dropped['no_keyword']:,}")
        print(f"  dropped, coverage  : {len(dropped['insufficient_coverage'])}")
        if dropped["empty_keywords"]:
            print(f"  MATERIALS WITH NO ENTRIES: {', '.join(dropped['empty_keywords'])}")
            for note in fallback_note(dropped["empty_keywords"]):
                print(f"      ! {note}")
        top = sorted(dropped["per_keyword"].items(), key=lambda kv: -kv[1])[:6]
        print("  kept by material   : "
              + ", ".join(f"{k}={n}" for k, n in top if n))
        print(f"  -> {out}\n")
        report["subsets"][spec.name] = {
            "purpose": spec.purpose,
            "coverage_um": list(spec.coverage_um),
            "derivation": spec.derive,
            "n_kept": len(kept),
            "n_counterexamples": n_counter,
            "names": [s.name for s in kept],
            "source_files": [s.source_file for s in kept],
            "dropped": dropped,
            "path": str(out),
        }

    out_report = Path(args.report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report written to {out_report}")

    failures = [m for m, r in validation.items() if r.get("status") == "OUT OF RANGE"]
    if failures:
        print(f"\nWARNING: Kirchhoff validation out of range for {failures}. "
              "The hemispherical filter may be admitting entries it should not.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
