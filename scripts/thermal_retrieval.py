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
from pipeline.errors import NonRadiometricInputError  # noqa: E402
from pipeline.thermal.loader import list_frames, load_frame  # noqa: E402
from pipeline.thermal.planck import (  # noqa: E402
    EmissivityTable,
    retrieve_temperature,
)

#: Materials a wildfire scene plausibly contains, with published LWIR
#: emissivities. Small and explicit rather than a stand-in for the ECOSTRESS
#: library -- these are the values the worked example needs, and each one is
#: sourced so the assumption is auditable.
WILDFIRE_MATERIALS = {
    "materials": {
        "vegetation": {"emissivity": 0.98, "range": [0.96, 0.99],
                       "source": "green canopy, LWIR 8-14 um"},
        "dry grass": {"emissivity": 0.95, "range": [0.92, 0.97],
                      "source": "senescent vegetation, LWIR"},
        "charred wood": {"emissivity": 0.96, "range": [0.94, 0.98],
                         "source": "carbonised surface, LWIR"},
        "soil": {"emissivity": 0.92, "range": [0.90, 0.95], "source": "dry bare soil, LWIR"},
        "water": {"emissivity": 0.99, "range": [0.98, 0.995], "source": "still water, LWIR"},
        "steel": {"emissivity": 0.28, "range": [0.10, 0.40],
                  "source": "oxidised steel -- low and highly variable, LWIR"},
    }
}


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
    table = EmissivityTable(
        WILDFIRE_MATERIALS["materials"],
        default=cfg.default_emissivity,
        default_range=tuple(cfg.default_emissivity_range),
    )

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

        for material in ("charred wood", "dry grass", "steel", "unlisted material"):
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
                f"    {material:<18} e={estimate.value:.2f}{flag:<10} "
                f"-> {float(out.temperature_c):7.1f} C  "
                f"[{row[material]['low_c']:.1f}, {row[material]['high_c']:.1f}]  "
                f"spread {float(out.spread_k):.1f} K"
            )
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
        "note": (
            "Single-band retrieval cannot separate temperature from emissivity; the "
            "emissivity is assumed and its range is propagated into the bounds. "
            "Proper TES needs multiple bands, which FLAME 3 does not provide."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
