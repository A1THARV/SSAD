"""Extract two curated subsets from ECOSTRESS ``spectra_all.txt``.

    python scripts/build_speclib.py --source data/speclib/ecostress/spectra_all.txt

One download, two purposes, because M5 made a second subset load-bearing:

* **Reflectance, for Objective 2 (SAM).** Minerals carrying diagnostic SWIR
  absorption features -- clays, carbonates, sulfates. Sized for a
  SWIR-diagnostic scene per RESULTS.md §3, *not* for HAD100, whose signal sits
  at 455-700 nm where a spectral library has little to say.
* **LWIR emissivity, for Objective 4 (Eq. 5).** Vegetation, char, ash, soil --
  the wildfire material class -- plus one or two metals kept deliberately as the
  out-of-class counterexample that produces the 307 K error in §13.

The second subset is the point of doing this now: it turns the emissivity prior
from a hardcoded constant into a cited measurement.

**Coverage is checked per entry, at build time.** ``spectra_all.txt`` bundles
the JHU, JPL and USGS sub-libraries, which do not share a wavelength grid or a
sample convention -- many LWIR entries never reach SWIR and vice versa. Since
:func:`sam.resample_to_grid` refuses to extrapolate (and should -- edge-clamping
would invent flat spectral shape that SAM, being a shape comparison, reads as
genuine similarity), an out-of-range entry would otherwise surface as a refusal
in the middle of matching. Checking here converts that runtime surprise into a
build-time report naming which entries were dropped and why.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import numpy as np  # noqa: E402

from pipeline.errors import DataUnavailableError  # noqa: E402

# --------------------------------------------------------------------------
# What each subset needs
# --------------------------------------------------------------------------


@dataclass
class SubsetSpec:
    """One curated subset: what to keep, and the grid it must cover."""

    name: str
    purpose: str
    #: Required wavelength span in micrometres. An entry that does not cover it
    #: is dropped with a reason rather than extrapolated.
    coverage_um: tuple[float, float]
    #: Substrings matched case-insensitively against name/class/subclass.
    keywords: list[str]
    #: Measurement type as ECOSTRESS labels it.
    spectrum_type: str
    #: Kept deliberately as counterexamples, not as scene members.
    counterexamples: list[str] = field(default_factory=list)


#: SWIR diagnostics: OH/carbonate near 2200-2350 nm, water near 1400/1900 nm.
REFLECTANCE = SubsetSpec(
    name="reflectance_swir",
    purpose="Objective 2 -- SAM material matching on a SWIR-diagnostic scene",
    coverage_um=(0.4, 2.5),
    spectrum_type="reflectance",
    keywords=[
        "kaolinite", "montmorillonite", "illite", "muscovite", "chlorite",
        "calcite", "dolomite", "gypsum", "alunite", "jarosite",
        "hematite", "goethite", "quartz", "feldspar", "serpentine",
    ],
)

#: LWIR emissivity for Eq. 5's prior. 8-14 um is the thermal camera's window.
EMISSIVITY = SubsetSpec(
    name="emissivity_lwir",
    purpose="Objective 4 -- the epsilon prior for Eq. 5 retrieval",
    coverage_um=(8.0, 14.0),
    spectrum_type="emissivity",
    keywords=["grass", "vegetation", "conifer", "deciduous", "leaf",
              "char", "ash", "soil", "sand", "bark", "wood"],
    counterexamples=["steel", "aluminum", "aluminium", "copper", "galvanized"],
)

SUBSETS = [REFLECTANCE, EMISSIVITY]


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


@dataclass
class Spectrum:
    name: str
    metadata: dict[str, str]
    wavelength_um: np.ndarray
    value: np.ndarray

    @property
    def coverage(self) -> tuple[float, float]:
        return float(self.wavelength_um.min()), float(self.wavelength_um.max())

    def covers(self, span: tuple[float, float]) -> bool:
        lo, hi = self.coverage
        return lo <= span[0] and hi >= span[1]


_HEADER = re.compile(r"^\s*([A-Za-z][A-Za-z ./]*?)\s*:\s*(.*?)\s*$")


def parse_ecostress(path: Path) -> list[Spectrum]:
    """Read ECOSTRESS's concatenated ASCII export.

    Each record is a block of ``Key: Value`` headers followed by two numeric
    columns. Wavelength is micrometres; the second column is percent for both
    reflectance and emissivity, so it is divided by 100 here and the subsets are
    written in physical units rather than percent.
    """
    if not path.is_file():
        raise DataUnavailableError(
            reason="ECOSTRESS spectra_all.txt not found",
            path=path,
            remedy=(
                "Order 'spectra_all.txt' from https://speclib.jpl.nasa.gov/download "
                "and place the file at this path. The ordering page emails a link "
                "when the order is processed."
            ),
        )

    spectra: list[Spectrum] = []
    meta: dict[str, str] = {}
    rows: list[tuple[float, float]] = []

    def flush() -> None:
        if rows and meta:
            arr = np.array(sorted(rows), dtype=np.float64)
            spectra.append(
                Spectrum(
                    name=meta.get("Name", meta.get("Sample No", f"entry_{len(spectra)}")),
                    metadata=dict(meta),
                    wavelength_um=arr[:, 0],
                    value=arr[:, 1] / 100.0,
                )
            )

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        header = _HEADER.match(line)
        numeric = None
        if not header:
            parts = line.split()
            if len(parts) == 2:
                try:
                    numeric = (float(parts[0]), float(parts[1]))
                except ValueError:
                    numeric = None

        if numeric is not None:
            rows.append(numeric)
        elif header:
            # A header line after data means the previous record ended.
            if rows:
                flush()
                meta, rows = {}, []
            meta[header.group(1).strip()] = header.group(2).strip()

    flush()
    return spectra


def _haystack(spectrum: Spectrum) -> str:
    keys = ("Name", "Type", "Class", "Subclass", "Genus", "Species", "Sample No")
    return " ".join([spectrum.name, *(spectrum.metadata.get(k, "") for k in keys)]).lower()


def select(spectra: list[Spectrum], spec: SubsetSpec) -> tuple[list[Spectrum], dict]:
    """Apply a subset's filters, reporting what was dropped and why."""
    wanted_type = spec.spectrum_type.lower()
    terms = [(k, False) for k in spec.keywords] + [(k, True) for k in spec.counterexamples]

    kept: list[Spectrum] = []
    dropped = {"wrong_type": 0, "no_keyword": 0, "insufficient_coverage": []}

    for s in spectra:
        text = _haystack(s)
        declared = s.metadata.get("Type", "").lower()
        if wanted_type not in declared and wanted_type not in text:
            dropped["wrong_type"] += 1
            continue
        match = next((k for k, _ in terms if k in text), None)
        if match is None:
            dropped["no_keyword"] += 1
            continue
        # The check that would otherwise fire at match time.
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
        kept.append(s)

    return kept, dropped


def write_subset(kept: list[Spectrum], spec: SubsetSpec, out_dir: Path) -> Path:
    """Save one subset as a single ``.npz`` on a common grid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.name}.npz"

    # A shared grid inside the subset, so SAM resamples once at load rather than
    # per comparison. Bounds are the subset's required coverage, which every
    # kept entry spans by construction -- so this interpolates and never
    # extrapolates.
    grid = np.linspace(*spec.coverage_um, 512)
    stacked = np.vstack(
        [np.interp(grid, s.wavelength_um, s.value) for s in kept]
    ) if kept else np.zeros((0, grid.size))

    np.savez_compressed(
        path,
        wavelength_um=grid,
        spectra=stacked,
        names=np.array([s.name for s in kept], dtype=object),
        counterexample=np.array(
            [s.metadata.get("_is_counterexample", "false") == "true" for s in kept]
        ),
        purpose=spec.purpose,
    )
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", default="data/speclib/ecostress/spectra_all.txt")
    p.add_argument("--out", default="data/speclib/curated")
    p.add_argument("--report", default="results/speclib_build.json")
    args = p.parse_args(argv)

    source = Path(args.source)
    print(f"reading {source} ...", flush=True)
    spectra = parse_ecostress(source)
    print(f"  parsed {len(spectra):,} spectra\n")

    report = {"source": str(source), "n_parsed": len(spectra), "subsets": {}}
    for spec in SUBSETS:
        kept, dropped = select(spectra, spec)
        out = write_subset(kept, spec, Path(args.out))
        n_counter = sum(
            s.metadata.get("_is_counterexample") == "true" for s in kept
        )
        print(f"{spec.name}  ({spec.purpose})")
        print(f"  coverage required : {spec.coverage_um[0]}-{spec.coverage_um[1]} um")
        print(f"  kept              : {len(kept)}  ({n_counter} counterexample)")
        print(f"  dropped, type     : {dropped['wrong_type']:,}")
        print(f"  dropped, keyword  : {dropped['no_keyword']:,}")
        print(f"  dropped, coverage : {len(dropped['insufficient_coverage'])}")
        for entry in dropped["insufficient_coverage"][:5]:
            print(f"      {entry['name'][:52]:<52} {entry['coverage_um']} um")
        if len(dropped["insufficient_coverage"]) > 5:
            print(f"      ... and {len(dropped['insufficient_coverage']) - 5} more")
        print(f"  -> {out}\n")
        report["subsets"][spec.name] = {
            "purpose": spec.purpose,
            "coverage_um": list(spec.coverage_um),
            "n_kept": len(kept),
            "n_counterexamples": n_counter,
            "names": [s.name for s in kept],
            "dropped": dropped,
            "path": str(out),
        }

    out_report = Path(args.report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report written to {out_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
