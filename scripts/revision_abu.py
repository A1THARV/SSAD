"""C — generalisation to ABU, 13 scenes (INDICON revision).

    python scripts/revision_abu.py

Reviewer 1 notes ABU results are missing. They are, and producing them exposes a
practical asymmetry between the two branches that belongs in the paper:

* **RX transfers across sensors with no retraining.** It estimates its background
  statistics from whatever scene it is handed, so a per-scene PCA basis and a
  per-scene covariance is all it needs.
* **The autoencoder does not.** It was trained on HAD100's 276 AVIRIS-NG bands,
  and ABU's 13 scenes carry **seven different band counts** -- 102, 188, 191,
  193, 204, 205, 207 -- none of which is 276, and most of which differ from each
  other. A fixed-input network cannot consume them. So the autoencoder is
  retrained **per scene**, Auto-AD style: self-supervised on the scene under
  test, using no labels.

That is a legitimate published variant (Auto-AD, paper reference [8]) but it is
also a cost: 13 trainings instead of one transfer, and a background model fitted
on data containing the anomalies it must find.

Seeds 0-4 per scene, so ABU carries the same repeatability evidence as A1. Score
maps are written out so the cross-validated alpha from B can be applied without
retraining.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from pipeline.config import HyperspectralConfig, load_config, resolve_device, set_seeds  # noqa: E402
from pipeline.hyperspectral.autoencoder import (  # noqa: E402
    extract_patches,
    reconstruction_error_map,
    train_autoencoder,
)
from pipeline.hyperspectral.loader import load_cube  # noqa: E402
from pipeline.hyperspectral.preprocess import BandReducer, mask_and_clip  # noqa: E402
from pipeline.hyperspectral.rx_detector import rx_score  # noqa: E402
from pipeline.hyperspectral.score_fusion import fuse_scores  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
ALPHA_APRIORI = 0.5
ABU_DIR = Path("data/hyperspectral/abu")


def scene_auc(score_map, mask) -> float:
    g = np.asarray(mask, dtype=bool).ravel()
    if not (0 < g.sum() < g.size):
        return float("nan")
    return float(roc_auc_score(g, np.asarray(score_map, dtype=np.float64).ravel()))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/abu_urban.yaml")
    p.add_argument("--out", default="results/revision/c_abu.json")
    p.add_argument("--maps", default="results/revision/abu_maps")
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]

    base = load_config(args.config, HyperspectralConfig)
    map_dir = Path(args.maps)
    map_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(ABU_DIR.glob("*.mat"))[: args.limit]
    device = resolve_device(base.autoencoder.device)
    K = base.preprocess.pca_n_components

    print(f"ABU: {len(paths)} scenes, seeds {seeds}, K={K}, device={device}\n", flush=True)
    t0 = time.perf_counter()
    rows, band_counts = [], {}

    for path in paths:
        cfg = replace(
            base,
            dataset=replace(base.dataset, name=path.stem, path=str(path),
                            ground_truth_path=str(path)),
        )
        cube = load_cube(cfg.dataset)
        mask = np.asarray(cube.ground_truth, dtype=bool)
        clipped = mask_and_clip(cube, cfg.preprocess)
        band_counts[path.stem] = int(cube.n_bands)

        # RX: per-scene basis, because ABU's band counts differ from each other
        # as well as from the HAD100 corpus, so no shared basis exists.
        reducer = BandReducer(n_components=K,
                             variance=cfg.preprocess.pca_variance).fit(clipped)
        reduced = reducer.transform(clipped)
        rx_map = rx_score(reduced, ridge=cfg.rx.ridge, shrinkage=cfg.rx.shrinkage)
        rx_a = scene_auc(rx_map, mask)

        patches = extract_patches(reduced, cfg.autoencoder.patch_size)
        ae_aucs, fu_aucs, train_s = [], [], []

        for seed in seeds:
            blob = map_dir / f"{path.stem}_s{seed}.npz"
            if blob.is_file():
                d = np.load(blob)
                ae_map = d["ae"]
                train_s.append(float(d["train_seconds"]))
            else:
                # Auto-AD style: a fresh model per scene, self-supervised on the
                # scene itself. No labels are used -- `mask` is only ever read to
                # score, never to train.
                set_seeds(seed, cfg.deterministic)
                model, hist = train_autoencoder(
                    patches, cfg.autoencoder, reducer=reducer,
                    device=device, seed=seed,
                )
                ae_map = reconstruction_error_map(
                    model, reduced, patch_size=cfg.autoencoder.patch_size,
                    device=device)
                train_s.append(hist.seconds)
                np.savez_compressed(blob, ae=ae_map, rx=rx_map,
                                    train_seconds=hist.seconds)
            ae_aucs.append(scene_auc(ae_map, mask))
            fu_aucs.append(
                scene_auc(fuse_scores(rx_map, ae_map, mode=cfg.score_fusion.mode,
                                      alpha=ALPHA_APRIORI), mask))

        ae_aucs, fu_aucs = np.array(ae_aucs), np.array(fu_aucs)
        rows.append({
            "scene": path.stem, "n_bands": int(cube.n_bands),
            "shape": list(cube.shape[:2]),
            "anomaly_fraction": float(mask.mean()),
            "rx_auc": rx_a,
            "ae_auc_per_seed": [float(x) for x in ae_aucs],
            "ae_mean": float(ae_aucs.mean()), "ae_std": float(ae_aucs.std(ddof=1)),
            "fused_auc_per_seed": [float(x) for x in fu_aucs],
            "fused_mean": float(fu_aucs.mean()), "fused_std": float(fu_aucs.std(ddof=1)),
            "train_seconds_median": float(np.median(train_s)),
        })
        print(f"  {path.stem:<16} {cube.n_bands:>3}b  rx {rx_a:.4f}   "
              f"ae {ae_aucs.mean():.4f}+/-{ae_aucs.std(ddof=1):.4f}   "
              f"fused {fu_aucs.mean():.4f}+/-{fu_aucs.std(ddof=1):.4f}   "
              f"[{time.perf_counter() - t0:.0f}s]", flush=True)

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"alpha_apriori": ALPHA_APRIORI, "seeds": seeds,
             "band_counts": band_counts, "scenes": rows}, indent=2), encoding="utf-8")

    rx_all = np.array([r["rx_auc"] for r in rows])
    ae_all = np.array([r["ae_mean"] for r in rows])
    fu_all = np.array([r["fused_mean"] for r in rows])
    seed_stds = np.array([r["ae_std"] for r in rows])

    print(f"\n{'=' * 64}")
    print(f"  RX     mean over {len(rows)} scenes  {rx_all.mean():.4f} "
          f"+/- {rx_all.std(ddof=1):.4f}  (across-scene spread)")
    print(f"  AE     mean  {ae_all.mean():.4f} +/- {ae_all.std(ddof=1):.4f}   "
          f"median within-scene seed std {np.median(seed_stds):.4f}")
    print(f"  fused  mean  {fu_all.mean():.4f} +/- {fu_all.std(ddof=1):.4f}   "
          f"(alpha = {ALPHA_APRIORI})")
    print(f"  fused - RX   {fu_all.mean() - rx_all.mean():+.4f}")
    print(f"\n  band counts present: {sorted(set(band_counts.values()))}")
    print("  none is 276, so the HAD100-trained autoencoder cannot transfer;")
    print("  RX needs no retraining at any of them.")

    summary = {
        "alpha_apriori": ALPHA_APRIORI, "seeds": seeds,
        "band_counts": band_counts,
        "distinct_band_counts": sorted(set(band_counts.values())),
        "scenes": rows,
        "summary": {
            "rx_mean": float(rx_all.mean()), "rx_std": float(rx_all.std(ddof=1)),
            "ae_mean": float(ae_all.mean()), "ae_std": float(ae_all.std(ddof=1)),
            "ae_median_within_scene_seed_std": float(np.median(seed_stds)),
            "fused_mean": float(fu_all.mean()), "fused_std": float(fu_all.std(ddof=1)),
            "fused_minus_rx": float(fu_all.mean() - rx_all.mean()),
            "total_hours": (time.perf_counter() - t0) / 3600,
        },
        "note": (
            "The autoencoder is trained PER SCENE (Auto-AD style, self-supervised, "
            "no labels) because ABU's 13 scenes carry 7 different band counts, none "
            "of them 276, so the HAD100-trained network cannot consume them. RX "
            "transfers with no retraining. +/- on RX and on the means is the spread "
            "ACROSS scenes; the within-scene seed std is reported separately."
        ),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
