"""M3 secondary: SAM on real HAD100 detections — a consistency check, not a demo.

    python scripts/sam_on_had100.py

**This runs in the regime §14 says is weakest, and that is the point.**

HAD100's anomalies are a median of 26 pixels in a 64×64 crop — 0.62 % of the
image — and individual anomalous pixels are mixtures of target and background at
sub-pixel to near-pixel abundance. §14's controlled experiment, at realistic
wavelength-dependent SNR against a 253-species library, measures species-level
top-1 accuracy of **0.068 at a = 0.10** and **0.024 at a = 0.05**.

So the expected outcome here is *not* a table of mineral names. Producing one and
presenting it as identification would contradict this project's own evidence.
What is reported instead:

* the **distribution of best-match spectral angles**, which says how close
  anything in the library comes to these spectra at all;
* the **low-confidence flag rate** at the configured threshold, which is the
  mechanism §11 edge case 7 requires for exactly this situation;
* the **top-1 stability** across the anomalous pixels of a single scene — if the
  match is real it should be consistent within one target, and if it is noise it
  should scatter.

A high flag rate and scattered matches would *confirm* §14's prediction on real
data, which is a stronger result than a plausible-looking mineral table. The two
experiments are cross-checking each other.

Note also that HAD100's targets are fabric panels, not minerals. The library
holds no fabric, so the correct answer is frequently "none of these" — a further
reason to report angles and flags rather than forced top-1 labels.
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

from pipeline.config import HyperspectralConfig, load_config  # noqa: E402
from pipeline.errors import DataUnavailableError  # noqa: E402
from pipeline.hyperspectral.loader import list_corpus, load_cube  # noqa: E402
from pipeline.hyperspectral.preprocess import mask_and_clip  # noqa: E402
from pipeline.hyperspectral.sam import (  # noqa: E402
    resample_to_grid,
    spectral_angle_batch,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam_mixing_experiment import sensor_wavelengths_nm  # noqa: E402

CURATED = Path("data/speclib/curated/reflectance_swir.npz")

#: What §14 predicts for this regime, at realistic SNR against 253 species.
PREDICTED = {"a=0.10": 0.068, "a=0.05": 0.024}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/had100.yaml")
    p.add_argument("--out", default="results/sam_on_had100.json")
    p.add_argument("--limit", type=int, default=100)
    args = p.parse_args(argv)

    cfg = load_config(args.config, HyperspectralConfig)
    if not CURATED.is_file():
        raise DataUnavailableError(
            reason="Curated SWIR reflectance subset not found",
            path=CURATED,
            remedy="python scripts/build_speclib.py",
        )
    blob = np.load(CURATED, allow_pickle=True)
    lib_nm = blob["wavelength_um"] * 1000.0
    materials = np.array([str(m) for m in blob["material"]])
    sensor_nm = sensor_wavelengths_nm()
    library = np.vstack([
        resample_to_grid(blob["spectra"][i], lib_nm, sensor_nm)
        for i in range(blob["spectra"].shape[0])
    ])

    threshold = cfg.sam.low_confidence_angle_rad
    print(f"library : {library.shape[0]} spectra, {len(set(materials))} materials")
    print(f"flag    : best-match angle above {threshold} rad is low confidence\n")

    paths = list_corpus(cfg.dataset.path)[: args.limit]
    rows, all_angles, all_flags, scene_stability = [], [], [], []

    for path in paths:
        cube = load_cube(cfg.dataset, path)
        if cube.ground_truth is None:
            continue
        clipped = mask_and_clip(cube, cfg.preprocess)
        mask = np.asarray(cube.ground_truth, dtype=bool)
        if not mask.any():
            continue

        # The anomaly relative to its own scene background -- the same
        # formulation §14 found necessary, and what RX gives the pipeline.
        background = clipped[~mask].reshape(-1, clipped.shape[-1]).mean(axis=0)
        pixels = clipped[mask].reshape(-1, clipped.shape[-1]) - background
        ref = library - background

        keep = np.linalg.norm(pixels, axis=1) > 0
        if not keep.any():
            continue
        angles = spectral_angle_batch(pixels[keep], ref)
        best = np.argmin(angles, axis=1)
        best_angle = angles[np.arange(angles.shape[0]), best]
        labels = materials[best]

        # Would a real match be consistent across one target's pixels?
        counts = np.unique(labels, return_counts=True)[1]
        stability = counts.max() / labels.size

        all_angles.append(best_angle)
        all_flags.append(best_angle > threshold)
        scene_stability.append(stability)
        rows.append({
            "scene": cube.name,
            "n_anomaly_px": int(mask.sum()),
            "anomaly_fraction": float(mask.mean()),
            "median_best_angle_rad": float(np.median(best_angle)),
            "low_confidence_fraction": float((best_angle > threshold).mean()),
            "top1_stability": float(stability),
            "modal_material": str(labels[np.argmax(np.bincount(
                np.unique(labels, return_inverse=True)[1]))]),
        })

    if not rows:
        print("no labelled scenes processed")
        return 1

    angles = np.concatenate(all_angles)
    flags = np.concatenate(all_flags)
    stab = np.array(scene_stability)

    print(f"{len(rows)} scenes, {angles.size:,} anomalous pixels\n")
    print("best-match spectral angle (rad), pooled over anomalous pixels:")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<3} {np.percentile(angles, q):.4f}")
    print(f"\nlow-confidence flag rate (angle > {threshold}): {flags.mean():.3f}")
    print(f"top-1 stability within a scene: median {np.median(stab):.3f}, "
          f"min {stab.min():.3f}, max {stab.max():.3f}")
    print(f"  (1.0 = every anomalous pixel in the scene matched the same material)")

    modal = {}
    for r in rows:
        modal[r["modal_material"]] = modal.get(r["modal_material"], 0) + 1
    print("\nmodal top-1 material per scene:")
    for m, n in sorted(modal.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<16} {n:>3} scenes")

    print("\n" + "-" * 70)
    print("READ THIS BEFORE THE TABLE ABOVE.")
    print(f"  HAD100 anomalies are a median 0.62% of a crop, so individual")
    print(f"  anomalous pixels sit at sub-pixel to near-pixel abundance. §14")
    print(f"  measures species-level accuracy of {PREDICTED['a=0.10']:.3f} at a=0.10 and")
    print(f"  {PREDICTED['a=0.05']:.3f} at a=0.05 against a 253-species library at realistic SNR.")
    print("  These labels are therefore NOT presented as identifications. The")
    print("  angle distribution and flag rate are the result; the material names")
    print("  are shown only so the scatter is visible.")
    print("  HAD100's targets are fabric panels. The library holds no fabric, so")
    print("  the correct answer is often 'none of these' and top-1 is forced.")
    print("-" * 70)

    payload = {
        "n_scenes": len(rows),
        "n_anomalous_pixels": int(angles.size),
        "low_confidence_threshold_rad": threshold,
        "low_confidence_fraction": float(flags.mean()),
        "best_angle_percentiles_rad": {
            f"p{q}": float(np.percentile(angles, q)) for q in (5, 25, 50, 75, 95)
        },
        "top1_stability": {
            "median": float(np.median(stab)),
            "min": float(stab.min()),
            "max": float(stab.max()),
        },
        "modal_material_scene_counts": modal,
        "per_scene": rows,
        "interpretation": (
            "Identification WITHOUT verification, reported as a consistency check "
            "against RESULTS.md §14 rather than as a result. HAD100's anomalies are "
            "a median 0.62% of a crop, placing them in the sub-pixel regime where §14 "
            "measures species-level top-1 accuracy of 0.068 (a=0.10) and 0.024 "
            "(a=0.05) at realistic SNR against 253 species. HAD100's targets are "
            "fabric panels and the library holds no fabric, so top-1 is forced and "
            "the correct answer is frequently 'none of these'. Angles and flag rates "
            "are the reportable output; material names are shown only to make the "
            "scatter visible."
        ),
        "predicted_by_section_14": PREDICTED,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
