"""A2 + B — paired significance and cross-validated alpha (INDICON revision).

    python scripts/revision_stats.py

Both reviewers ask for something the project did not have. They are different
questions and are answered separately.

**A2, significance across test scenes (R3).** HAD100's 100 crops come from 94
source scenes: six scenes contribute two crops each, so the 100 per-image AUCs
are **not independent**. Crops are averaged within their source scene first,
giving 94 paired values, and the Wilcoxon signed-rank test and the bootstrap CI
both operate on those 94 -- the bootstrap resampling **source scenes**, not
crops, so the dependence is respected in the resampling too.

**B, alpha from a validation set (R3).** The a-priori alpha = 0.5 is defensible
but unvalidated, and the 0.9882 at alpha = 0.90 reported earlier was read off
the test set and is retired. This selects alpha by 5-fold cross-validation
**grouped by source scene**, so no scene appears in both a selection fold and
the fold it is evaluated on.

No retraining is needed for either: the autoencoder trains on the anomaly-free
corpus, so per-image score maps are reusable and fusion is a weighted sum.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import numpy as np  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from pipeline.hyperspectral.score_fusion import fuse_scores  # noqa: E402

TEST_DIR = Path("data/hyperspectral/had100/HAD100/HAD100Dataset/test/aviris_ng")
GT_DIR = Path("data/hyperspectral/had100/HAD100/HAD100Dataset/ground_truth/aviris_ng")
ALPHA_GRID = tuple(round(a, 1) for a in np.arange(0.0, 1.01, 0.1))
N_BOOTSTRAP = 10_000
N_FOLDS = 5


def crop_names() -> list[str]:
    return sorted(p.stem for p in TEST_DIR.glob("*.npy"))


def source_scene(name: str, all_names: set[str]) -> str:
    """Strip a `_1`/`_2` suffix only when its sibling crop also exists.

    Derived from the filenames rather than hard-coded: six source scenes
    contribute two crops each, and every other name ends in a digit too, so a
    blind regex would merge scenes that are genuinely distinct.
    """
    m = re.match(r"^(.*)_([12])$", name)
    if m and f"{m.group(1)}_1" in all_names and f"{m.group(1)}_2" in all_names:
        return m.group(1)
    return name


def load_truths(names: list[str]) -> list[np.ndarray]:
    from PIL import Image
    out = []
    for n in names:
        arr = np.array(Image.open(GT_DIR / f"{n}.png"))
        out.append((arr[:, :, 1] if arr.ndim == 3 else arr) > 0)
    return out


def load_maps(path: Path) -> list[np.ndarray]:
    d = np.load(path)
    keys = sorted((k for k in d.files if k.startswith("m")),
                  key=lambda k: int(k[1:]))
    return [d[k] for k in keys]


def per_image_auc(maps, truths) -> np.ndarray:
    out = np.full(len(maps), np.nan)
    for i, (m, g) in enumerate(zip(maps, truths)):
        g = np.asarray(g, bool).ravel()
        if 0 < g.sum() < g.size:
            out[i] = roc_auc_score(g, np.asarray(m, np.float64).ravel())
    return out


def by_scene(values: np.ndarray, groups: list[str]) -> tuple[np.ndarray, list[str]]:
    """Average crop-level values within each source scene."""
    acc = collections.defaultdict(list)
    for v, g in zip(values, groups):
        if not np.isnan(v):
            acc[g].append(v)
        else:
            acc.setdefault(g, [])
    keys = sorted(k for k, v in acc.items() if v)
    return np.array([np.mean(acc[k]) for k in keys]), keys


def paired_test(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str,
                rng, n_boot: int = N_BOOTSTRAP) -> dict:
    """Wilcoxon signed-rank plus a bootstrap CI that resamples scenes."""
    d = a - b
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    ties = int((d == 0).sum())

    nz = d[d != 0]
    if nz.size:
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        stat, p = float(stat), float(p)
    else:
        stat, p = float("nan"), 1.0

    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    return {
        "comparison": f"{label_a} vs {label_b}",
        "n_pairs": int(d.size),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_difference": float(d.mean()),
        "ci95_bootstrap": [float(lo), float(hi)],
        "n_bootstrap": n_boot,
        "wins": wins, "ties": ties, "losses": losses,
        "wilcoxon_statistic": stat, "wilcoxon_p": p,
        "significant_at_0.05": bool(p < 0.05),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rx", default="results/m2-band50/maps/rx_k15_corpus.npz")
    p.add_argument("--ae", default="results/m2-band50/maps/ae_baseline.npz")
    p.add_argument("--out", default="results/revision/a2b_stats.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="band50 / seed 0")
    args = p.parse_args(argv)

    names = crop_names()
    nameset = set(names)
    groups = [source_scene(n, nameset) for n in names]
    truths = load_truths(names)
    n_scenes = len(set(groups))
    print(f"{len(names)} crops -> {n_scenes} source scenes "
          f"({len(names) - n_scenes} crops are second crops)\n")

    rx_maps = load_maps(Path(args.rx))
    ae_maps = load_maps(Path(args.ae))
    assert len(rx_maps) == len(ae_maps) == len(names), "map count mismatch"

    rx_auc = per_image_auc(rx_maps, truths)
    ae_auc = per_image_auc(ae_maps, truths)
    fused_maps = [fuse_scores(r, a, alpha=0.5) for r, a in zip(rx_maps, ae_maps)]
    fu_auc = per_image_auc(fused_maps, truths)

    print("crop-level mAUC (100 crops, as previously reported):")
    print(f"  RX     {np.nanmean(rx_auc):.4f}")
    print(f"  AE     {np.nanmean(ae_auc):.4f}")
    print(f"  fused  {np.nanmean(fu_auc):.4f}   (alpha = 0.5)")

    rx_s, keys = by_scene(rx_auc, groups)
    ae_s, _ = by_scene(ae_auc, groups)
    fu_s, _ = by_scene(fu_auc, groups)
    print(f"\nscene-level mAUC ({len(keys)} source scenes):")
    print(f"  RX     {rx_s.mean():.4f}")
    print(f"  AE     {ae_s.mean():.4f}")
    print(f"  fused  {fu_s.mean():.4f}")

    rng = np.random.default_rng(args.seed)
    print(f"\n=== A2: paired tests on {len(keys)} source scenes ===")
    tests = [
        paired_test(fu_s, rx_s, "fused(a=0.5)", "RX", rng),
        paired_test(ae_s, rx_s, "autoencoder", "RX", rng),
    ]
    for t in tests:
        print(f"\n  {t['comparison']}")
        print(f"    mean diff {t['mean_difference']:+.5f}   "
              f"95% CI [{t['ci95_bootstrap'][0]:+.5f}, {t['ci95_bootstrap'][1]:+.5f}]")
        print(f"    win/tie/loss {t['wins']}/{t['ties']}/{t['losses']}")
        print(f"    Wilcoxon W = {t['wilcoxon_statistic']:.1f},  p = {t['wilcoxon_p']:.4g}")
        verdict = "SIGNIFICANT" if t["significant_at_0.05"] else "NOT significant"
        print(f"    => {verdict} at alpha=0.05;  CI "
              f"{'excludes' if t['ci_excludes_zero'] else 'INCLUDES'} zero")

    # --- B: alpha by grouped 5-fold cross-validation ----------------------
    print(f"\n=== B: alpha by {N_FOLDS}-fold CV, grouped by source scene ===")
    uniq = sorted(set(groups))
    fold_rng = np.random.default_rng(args.seed)
    order = fold_rng.permutation(len(uniq))
    fold_of = {uniq[j]: int(i % N_FOLDS) for i, j in enumerate(order)}
    crop_fold = np.array([fold_of[g] for g in groups])

    # per-crop AUC for every alpha on the grid, computed once
    auc_by_alpha = {}
    for a in ALPHA_GRID:
        maps = [fuse_scores(r, x, alpha=a) for r, x in zip(rx_maps, ae_maps)]
        auc_by_alpha[a] = per_image_auc(maps, truths)

    folds = []
    for f in range(N_FOLDS):
        tr = crop_fold != f
        te = crop_fold == f
        tr_groups = [g for g, m in zip(groups, tr) if m]
        te_groups = [g for g, m in zip(groups, te) if m]
        assert not (set(tr_groups) & set(te_groups)), "scene leaked across folds"

        scores = {}
        for a in ALPHA_GRID:
            v, _ = by_scene(auc_by_alpha[a][tr], tr_groups)
            scores[a] = float(v.mean())
        best = max(scores, key=scores.get)
        held, _ = by_scene(auc_by_alpha[best][te], te_groups)
        held_05, _ = by_scene(auc_by_alpha[0.5][te], te_groups)
        rx_held, _ = by_scene(rx_auc[te], te_groups)
        folds.append({
            "fold": f, "n_train_scenes": len(set(tr_groups)),
            "n_test_scenes": len(set(te_groups)),
            "alpha_selected": best,
            "train_mauc_at_selected": scores[best],
            "heldout_mauc": float(held.mean()),
            "heldout_mauc_at_alpha_0.5": float(held_05.mean()),
            "heldout_rx_mauc": float(rx_held.mean()),
            "train_curve": scores,
        })
        print(f"  fold {f}: alpha={best:.1f}  held-out {held.mean():.4f}   "
              f"(alpha=0.5 -> {held_05.mean():.4f},  RX -> {rx_held.mean():.4f})  "
              f"[{len(set(te_groups))} scenes]")

    hv = np.array([f["heldout_mauc"] for f in folds])
    h5 = np.array([f["heldout_mauc_at_alpha_0.5"] for f in folds])
    hr = np.array([f["heldout_rx_mauc"] for f in folds])
    sel = [f["alpha_selected"] for f in folds]
    print(f"\n  cross-validated alpha : {sel}")
    print(f"  held-out mAUC (CV a)  : {hv.mean():.4f} +/- {hv.std(ddof=1):.4f}")
    print(f"  held-out mAUC (a=0.5) : {h5.mean():.4f} +/- {h5.std(ddof=1):.4f}")
    print(f"  held-out mAUC (RX)    : {hr.mean():.4f} +/- {hr.std(ddof=1):.4f}")

    report = {
        "label": args.label,
        "rx_maps": args.rx, "ae_maps": args.ae,
        "n_crops": len(names), "n_source_scenes": n_scenes,
        "crop_level_mauc": {"rx": float(np.nanmean(rx_auc)),
                            "ae": float(np.nanmean(ae_auc)),
                            "fused_alpha0.5": float(np.nanmean(fu_auc))},
        "scene_level_mauc": {"rx": float(rx_s.mean()), "ae": float(ae_s.mean()),
                             "fused_alpha0.5": float(fu_s.mean())},
        "a2_paired_tests": tests,
        "b_alpha_cv": {
            "n_folds": N_FOLDS, "alpha_grid": list(ALPHA_GRID),
            "grouping": "source scene; no scene in both selection and held-out fold",
            "folds": folds,
            "alpha_selected_per_fold": sel,
            "heldout_mauc_cv_alpha_mean": float(hv.mean()),
            "heldout_mauc_cv_alpha_std": float(hv.std(ddof=1)),
            "heldout_mauc_alpha05_mean": float(h5.mean()),
            "heldout_mauc_alpha05_std": float(h5.std(ddof=1)),
            "heldout_mauc_rx_mean": float(hr.mean()),
            "heldout_mauc_rx_std": float(hr.std(ddof=1)),
        },
        "retired": ("The 0.9882 at alpha = 0.90 reported in an earlier draft is "
                    "retired: alpha was read off the test set it was evaluated on."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
