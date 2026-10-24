"""ABU paired tests, and B's cross-dataset alpha arm (INDICON revision).

    python scripts/revision_abu_stats.py

Two questions the ABU run leaves open.

**Does fusion beat RX on ABU?** The per-scene means answer it loosely; a paired
test over the 13 scenes answers it properly. Thirteen pairs is a small sample
and Wilcoxon's smallest attainable two-sided p at n=13 is about 2.4e-4, so the
test can reject but cannot resolve fine differences -- stated rather than left
for the reader to work out.

**Does an alpha chosen on ABU transfer to HAD100?** This is the cross-dataset
arm of B. Selecting alpha on one dataset and applying it to another is the
strongest form of the claim that the weight is not fitted to the test images:
the two differ in sensor, band count, scene content and anomaly type. If the
transferred alpha holds up against RX on HAD100, the weight is a property of the
method; if it only works when tuned in-domain, it is not.

The ABU autoencoder is trained per scene (Auto-AD style, no labels), so the
alpha selected there is selected on self-supervised models -- a further
difference from HAD100's corpus-trained one, and a harder test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from pipeline.hyperspectral.score_fusion import fuse_scores  # noqa: E402
from revision_stats import (  # noqa: E402
    ALPHA_GRID,
    by_scene,
    crop_names,
    load_maps,
    load_truths,
    paired_test,
    per_image_auc,
    source_scene,
)

ABU_JSON = Path("results/revision/c_abu.json")
ABU_MAPS = Path("results/revision/abu_maps")
HAD_MAPS = Path("results/revision/maps")
RX_K = 8
SEEDS = (0, 1, 2, 3, 4)


def spread(x) -> dict:
    a = np.asarray(x, dtype=float)
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)),
            "values": [float(v) for v in a]}


def abu_masks() -> dict:
    """ABU ground truth, read the same way revision_abu.py reads it.

    The saved maps carry no labels (correctly -- the per-scene training is
    self-supervised), so the masks are re-read from the .mat files here purely
    to score what was already computed.
    """
    from dataclasses import replace

    from pipeline.config import HyperspectralConfig, load_config
    from pipeline.hyperspectral.loader import load_cube

    from revision_abu import ABU_DIR

    cfg = load_config("configs/abu_urban.yaml", HyperspectralConfig)
    out = {}
    for path in sorted(ABU_DIR.glob("*.mat")):
        scene_cfg = replace(cfg, dataset=replace(
            cfg.dataset, path=str(path), ground_truth_path=str(path)))
        out[path.stem] = np.asarray(load_cube(scene_cfg.dataset).ground_truth,
                                    dtype=bool)
    return out


def abu_alpha_curve() -> tuple[dict, list[str]]:
    """Mean ABU AUC at each alpha, averaged over seeds then over scenes."""
    masks = abu_masks()
    blobs = sorted(ABU_MAPS.glob("*_s0.npz"))
    scenes = [p.name[: -len("_s0.npz")] for p in blobs]
    curve = {a: [] for a in ALPHA_GRID}
    for scene in scenes:
        gt = masks[scene].ravel()
        if not (0 < gt.sum() < gt.size):
            continue
        per_alpha = {a: [] for a in ALPHA_GRID}
        for seed in SEEDS:
            d = np.load(ABU_MAPS / f"{scene}_s{seed}.npz")
            for a in ALPHA_GRID:
                m = fuse_scores(d["rx"], d["ae"], alpha=a).ravel()
                per_alpha[a].append(roc_auc_score(gt, m.astype(np.float64)))
        for a in ALPHA_GRID:
            curve[a].append(float(np.mean(per_alpha[a])))
    return {a: float(np.mean(v)) for a, v in curve.items() if v}, scenes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/revision/abu_stats.json")
    args = p.parse_args(argv)

    # --- item 6: paired tests on the 13 ABU scenes ------------------------
    d = json.loads(ABU_JSON.read_text(encoding="utf-8"))
    sc = d["scenes"]
    rx = np.array([s["rx_auc"] for s in sc])
    ae = np.array([s["ae_auc_per_seed"] for s in sc])
    fu = np.array([s["fused_auc_per_seed"] for s in sc])
    names = [s["scene"] for s in sc]

    print(f"ABU: {len(sc)} scenes x {ae.shape[1]} seeds, alpha = 0.5, "
          f"AE trained per scene\n")
    print(f"  {'scene':<16}{'bands':>6}{'RX':>9}{'AE':>19}{'fused':>19}")
    for i, s in enumerate(sc):
        print(f"  {s['scene']:<16}{s['n_bands']:>6}{rx[i]:>9.4f}"
              f"{f'{ae[i].mean():.4f}+/-{ae[i].std(ddof=1):.4f}':>19}"
              f"{f'{fu[i].mean():.4f}+/-{fu[i].std(ddof=1):.4f}':>19}")

    tests = {}
    for label, vals in (("AE", ae.mean(1)), ("fused(a=0.5)", fu.mean(1))):
        rng = np.random.default_rng(11)
        t = paired_test(vals, rx, label, "RX", rng, n_boot=10_000)
        tests[label] = t
        print(f"\n  {label} vs RX over {len(sc)} scenes (seed-mean):")
        print(f"    mean diff {t['mean_difference']:+.5f}   95% CI "
              f"[{t['ci95_bootstrap'][0]:+.5f}, {t['ci95_bootstrap'][1]:+.5f}]")
        print(f"    win/tie/loss {t['wins']}/{t['ties']}/{t['losses']}   "
              f"W = {t['wilcoxon_statistic']:.1f},  p = {t['wilcoxon_p']:.4g}  "
              f"=> {'SIGNIFICANT' if t['significant_at_0.05'] else 'not significant'}")

    per_seed_tests = []
    print("\n  per seed, fused vs RX:")
    for j in range(fu.shape[1]):
        st, pv = wilcoxon(fu[:, j], rx, zero_method="wilcox")
        w = int((fu[:, j] > rx).sum())
        per_seed_tests.append({"seed": j, "wins": w, "losses": int(len(sc) - w),
                               "wilcoxon_statistic": float(st),
                               "wilcoxon_p": float(pv)})
        print(f"    seed {j}: fused {fu[:, j].mean():.4f}  W/L {w}/{len(sc) - w}"
              f"  W = {st:.1f}  p = {pv:.4g}")

    # --- B cross-dataset: pick alpha on ABU, apply to HAD100 --------------
    print("\n=== B cross-dataset: alpha selected on ABU, applied to HAD100 ===")
    curve, abu_scenes = abu_alpha_curve()
    best_abu = max(curve, key=curve.get)
    print(f"  ABU alpha curve ({len(abu_scenes)} scenes, {len(SEEDS)} seeds):")
    print("    " + "  ".join(f"{a:.1f}:{curve[a]:.4f}" for a in sorted(curve)))
    print(f"  => alpha selected on ABU: {best_abu}")

    hn = crop_names()
    groups = [source_scene(n, set(hn)) for n in hn]
    truths = load_truths(hn)
    rx_h = load_maps(HAD_MAPS / f"band50_rx_k{RX_K}_corpus.npz")
    rx_sc, _ = by_scene(per_image_auc(rx_h, truths), groups)

    rows, transferred = [], []
    for seed in SEEDS:
        ae_h = load_maps(HAD_MAPS / f"band50_ae_baseline_s{seed}.npz")
        fused = [fuse_scores(r, a, alpha=best_abu) for r, a in zip(rx_h, ae_h)]
        v, _ = by_scene(per_image_auc(fused, truths), groups)
        t = paired_test(v, rx_sc, f"fused(alpha={best_abu} from ABU)",
                        f"RX K={RX_K}", np.random.default_rng(3000 + seed))
        rows.append(t)
        transferred.append(v.mean())
        print(f"  seed {seed}: HAD100 scene mAUC {v.mean():.4f}  "
              f"diff {t['mean_difference']:+.5f}  W/T/L "
              f"{t['wins']}/{t['ties']}/{t['losses']}  p = {t['wilcoxon_p']:.3g}")

    tr = spread(transferred)
    n_sig = sum(r["significant_at_0.05"] and r["mean_difference"] > 0 for r in rows)
    print(f"\n  transferred alpha = {best_abu}: HAD100 {tr['mean']:.4f} +/- "
          f"{tr['std']:.4f}  (median {tr['median']:.4f})  vs RX {rx_sc.mean():.4f}")
    print(f"  seeds beating RX with the ABU-selected alpha: {n_sig} of {len(SEEDS)}")

    report = {
        "abu": {"n_scenes": len(sc), "scenes": names,
                "rx": [float(v) for v in rx],
                "ae_seed_mean": [float(v) for v in ae.mean(1)],
                "fused_seed_mean": [float(v) for v in fu.mean(1)],
                "paired_tests": tests,
                "fused_vs_rx_per_seed": per_seed_tests,
                "wilcoxon_floor_note": (
                    "n=13: the smallest attainable two-sided p is ~2.4e-4, so "
                    "this test rejects or does not, and cannot resolve finer.")},
        "cross_dataset_alpha": {
            "alpha_grid": list(ALPHA_GRID),
            "abu_alpha_curve": {str(k): v for k, v in curve.items()},
            "alpha_selected_on_abu": best_abu,
            "had100_rx_k": RX_K,
            "had100_rx_scene_mauc": float(rx_sc.mean()),
            "had100_transferred": tr,
            "had100_paired_tests": rows,
            "n_seeds_beating_rx": n_sig,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
