"""M5: emissivity-aware temperature retrieval on real FLAME 3 frames (Eq. 5).

    python scripts/thermal_retrieval.py

The thermal branch's substantive result, and the argument for it is M4's
triviality: a threshold on absolute temperature detects fire perfectly
(RESULTS.md §11) but cannot say what an anomaly is *made of*. Emissivity
retrieval needs the radiometry in a way detection demonstrably does not.

What this produces, per anomalous region:

* the camera's **brightness** temperature -- what a blackbody would have to be;
* the **retrieved** surface temperature under an assumed emissivity;
* the **spread** those bounds imply, which for an unlisted material is most of
  the uncertainty and is reported rather than dropped (§11 edge case 11).

It also runs the refusal path on the negative fixture, so the edge-case-9
guarantee is exercised end to end on real files rather than only in unit tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import numpy as np  # noqa: E402

from pipeline.config import PlanckConfig  # noqa: E402
from pipeline.errors import DataUnavailableError  # noqa: E402
from pipeline.errors import NonRadiometricInputError  # noqa: E402
from pipeline.thermal.loader import list_frames, load_frame  # noqa: E402
from pipeline.thermal.planck import (  # noqa: E402
    EmissivityTable,
    retrieve_temperature,
)

#: Material classes a wildfire scene plausibly contains, mapped to the substring
#: their spectra carry in the curated subset. Emissivity is no longer written
#: here as a constant -- it is derived from ECOSTRESS hemispherical reflectance
#: via Kirchhoff's law and read from ``emissivity_lwir.npz``.
#:
#: Vegetation resolves to ECOSTRESS's ``nonphotosyntheticvegetation.*`` branch,
#: not to every entry mentioning vegetation. A burning scene is dry and
#: senescent material, so averaging green canopy into the prior would describe a
#: scene that is not on fire.
#:
#: Measured, the correction turns out to be small -- non-photosynthetic
#: vegetation gives eps 0.9525 against 0.9537 for photosynthetic, a difference
#: of 0.001 -- and that is itself informative. Green-versus-dry is a VNIR
#: reflectance phenomenon (chlorophyll absorption, the red edge); LWIR
#: emissivity is dominated by water and cellulose vibrational features and by
#: surface structure, which the two share. The distinction that dominates
#: Objective 2 barely registers for Objective 4. The selection is still made
#: explicitly, because a prior that is right by coincidence is not provenance.
SCENE_MATERIALS = ("nonphotosyntheticvegetation", "char", "soil", "grass", "bark", "sand")

#: Display names, so the report does not print ECOSTRESS path fragments.
DISPLAY_NAMES = {"nonphotosyntheticvegetation": "dry vegetation"}

#: Kept deliberately OUT of class: it is what produces the large retrieval error
#: that shows why epsilon must be constrained to the right material class.
COUNTEREXAMPLE = "brass"

CURATED = Path("data/speclib/curated/emissivity_lwir.npz")


def load_emissivity_table(path: Path = CURATED) -> tuple[EmissivityTable, dict]:
    """Build an emissivity table from the curated ECOSTRESS subset.

    Each material's value is the mean over its spectra across the sensor band,
    and its **range is the observed spread across those spectra** rather than a
    guessed interval -- so the uncertainty reported with a retrieval is measured
    variability between real samples of that material.
    """
    if not path.is_file():
        raise DataUnavailableError(
            reason="Curated emissivity subset not found",
            path=path,
            remedy="python scripts/build_speclib.py",
        )
    blob = np.load(path, allow_pickle=True)
    names = [str(n).lower() for n in blob["names"]]
    files = [str(f).lower() for f in blob["source_files"]]
    spectra = blob["spectra"]

    materials, provenance = {}, {}
    for keyword in (*SCENE_MATERIALS, COUNTEREXAMPLE):
        mask = np.array(
            [keyword in n or keyword in f for n, f in zip(names, files)]
        )
        if not mask.any():
            continue
        per_entry = spectra[mask].mean(axis=1)
        materials[keyword] = {
            "emissivity": float(per_entry.mean()),
            "range": [float(per_entry.min()), float(per_entry.max())],
            "source": f"ECOSTRESS, Kirchhoff eps=1-R, n={int(mask.sum())}",
        }
        provenance[keyword] = int(mask.sum())

    # The default for an UNLISTED material is the scene-level prior, so it is
    # computed from in-class materials only. Including the counterexample would
    # drag the fallback down towards metal (0.84 rather than 0.95) and hand an
    # unidentified patch of a wildfire scene an emissivity no part of that scene
    # has -- the prior would be contaminated by the thing it exists to exclude.
    in_class = {k: v for k, v in materials.items() if k != COUNTEREXAMPLE}
    table = EmissivityTable(
        materials,
        default=float(np.mean([m["emissivity"] for m in in_class.values()])),
        default_range=(
            min(m["range"][0] for m in in_class.values()),
            max(m["range"][1] for m in in_class.values()),
        ),
    )
    return table, provenance


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--path", default="data/thermal/flame3_cv_subset")
    p.add_argument("--negative", default="data/thermal/thermal_images_equip")
    p.add_argument("--out", default="results/thermal/flame3_retrieval.json")
    p.add_argument("--hot-threshold", type=float, default=100.0,
                   help="Degrees C above which a pixel is treated as an anomaly.")
    p.add_argument("--n-frames", type=int, default=5)
    args = p.parse_args(argv)

    cfg = PlanckConfig()
    table, provenance = load_emissivity_table()
    print("emissivity from ECOSTRESS via Kirchhoff (eps = 1 - R, hemispherical):")
    for material, n in sorted(provenance.items()):
        est = table.get(material)
        tag = "  <- out of class" if material == COUNTEREXAMPLE else ""
        label = DISPLAY_NAMES.get(material, material)
        print(f"  {label:<16} eps {est.value:.4f}  "
              f"[{est.low:.4f}, {est.high:.4f}]  n={n}{tag}")
    print()

    paths = list_frames(Path(args.path) / "Fire" / "Thermal" / "Celsius TIFF")[: args.n_frames]
    print(f"Eq. 5 retrieval at {cfg.wavelength_um:g} um on {len(paths)} fire frames\n")

    records = []
    for path in paths:
        frame = load_frame(path, label=True)
        kelvin = frame.to_kelvin()          # refuses non-radiometric input
        hot = frame.data > args.hot_threshold
        if not hot.any():
            continue

        brightness = float(kelvin[hot].mean())
        row = {"frame": frame.name, "n_hot_px": int(hot.sum()),
               "brightness_temperature_k": brightness}
        print(f"{frame.name}: {int(hot.sum()):,} px above {args.hot_threshold:g} C, "
              f"brightness {brightness - 273.15:.1f} C")

        for material in (*SCENE_MATERIALS, COUNTEREXAMPLE, "unlisted material"):
            estimate = table.get(material)
            out = retrieve_temperature(brightness, estimate, cfg.wavelength_um)
            row[material] = {
                "emissivity": estimate.value,
                "assumed": estimate.is_assumed,
                "retrieved_c": float(out.temperature_c),
                "low_c": float(np.asarray(out.low_k) - 273.15),
                "high_c": float(np.asarray(out.high_k) - 273.15),
                "spread_k": float(out.spread_k),
            }
            flag = " (assumed)" if estimate.is_assumed else ""
            print(
                f"    {DISPLAY_NAMES.get(material, material):<18} "
                f"e={estimate.value:.2f}{flag:<10} "
                f"-> {float(out.temperature_c):7.1f} C  "
                f"[{row[material]['low_c']:.1f}, {row[material]['high_c']:.1f}]  "
                f"spread {float(out.spread_k):.1f} K"
            )
        in_class = [row[m]["retrieved_c"] for m in SCENE_MATERIALS if m in row]
        out_class = in_class + [row[COUNTEREXAMPLE]["retrieved_c"]]
        row["within_class_spread_k"] = max(in_class) - min(in_class)
        row["across_class_spread_k"] = max(out_class) - min(out_class)
        print(f"    within-class spread {row['within_class_spread_k']:7.1f} K")
        print(f"    across-class spread {row['across_class_spread_k']:7.1f} K "
              f"({row['across_class_spread_k'] / max(row['within_class_spread_k'], 1e-9):.0f}x)")
        records.append(row)
        print()

    # The refusal, exercised on a real non-radiometric file.
    print("edge case 9 -- the negative fixture:")
    negative = Path(args.negative)
    refusal = None
    if negative.is_dir():
        bad = load_frame(list_frames(negative, "*.bmp")[0])
        try:
            retrieve_temperature(bad.to_kelvin(), table.get("steel"), cfg.wavelength_um)
            print("  FAILED: a display product reached Eq. 5")
            return 1
        except NonRadiometricInputError as exc:
            refusal = str(exc).splitlines()[0]
            print(f"  refused as required: {refusal}")
    else:
        print("  fixture absent, skipped")

    payload = {
        "wavelength_um": cfg.wavelength_um,
        "hot_threshold_c": args.hot_threshold,
        "mode": "assumed-emissivity (single-band)",
        "frames": records,
        "edge_case_9_refusal": refusal,
        "emissivity_provenance": provenance,
        "note": (
            "Emissivity is DERIVED from measured ECOSTRESS hemispherical reflectance "
            "via Kirchhoff's law (eps = 1 - R), valid for opaque surfaces at local "
            "thermodynamic equilibrium -- an inference from a published measurement, "
            "not a measurement. Single-band retrieval cannot separate temperature from "
            "emissivity; proper TES needs multiple bands, which FLAME 3 does not "
            "provide. LIMITATION: these are laboratory spectra at ambient temperature "
            "applied to fire pixels at 220-600 C. LWIR emissivity is reasonably stable "
            "with temperature for most of these materials, but char and soot during "
            "active combustion are the least certain case -- and char is the primary "
            "material in this scene."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
