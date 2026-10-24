"""Step 3 — validate RX as the headline detector (INDICON revision).

    python scripts/revision_rx_validate.py

With fusion no longer carrying the result, RX is the headline, and two of its
choices were never justified.

**Where does the gain over published GRX come from?** HAD100 reports GRX at
0.9403; this work's RX reads 0.9835 on the same images. Two things differ:
the covariance is estimated in a PCA subspace rather than on raw bands, and that
subspace is fitted on a 1,040-crop background corpus rather than on each scene.
Only one of them may be the cause, so all three are measured and compared
pairwise:

    (a) raw 50 bands, no PCA      -- the GRX configuration
    (b) per-scene PCA-15          -- truncation without the corpus
    (c) corpus PCA-15             -- what the paper reports

(c) vs (a) is the total gain. (c) vs (b) isolates the corpus basis from
truncation. If (b) already recovers most of (a) -> (c), the corpus basis is not
the contribution and the paper should not claim it is.

**K = 15 was chosen with sight of HAD100's own curve.** R3's objection to
selecting alpha on the test set applies to K exactly as it does to alpha, so K
is re-selected here by the same scene-grouped 5-fold cross-validation, over
{5, 8, 10, 15, 20, 30, 50}.

The corpus basis is fitted ONCE at K = 50 and sliced. PCA components are nested,
so the top-5 of a 50-component fit are the 5-component fit; refitting per K
would cost seven streaming passes and give the same vectors.

Deterministic and CPU-only: no seed appears anywhere in RX.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import numpy as np  # noqa: E402

from pipeline.config import HyperspectralConfig, load_config  # noqa: E402
from pipeline.hyperspectral.preprocess import BandReducer  # noqa: E402
from pipeline.hyperspectral.rx_detector import rx_score  # noqa: E402
from pipeline.cli.run_hyperspectral import corpus_reducer  # noqa: E402
from revision_stats import (  # noqa: E402
    N_FOLDS,
    by_scene,
    crop_names,
    load_truths,
    paired_test,
    per_image_auc,
    source_scene,
)
from run_ablations import prepared_scenes  # noqa: E402

CONFIG = "configs/had100_band50.yaml"
K_GRID = (5, 8, 10, 15, 20, 30, 50)
K_REPORTED = 15             # what the paper reported
K_MODAL = 8                 # what cross-validation selects
FOLD_SEED = 0

#: Scored from the benchmark's own shipped result matrices by
#: scripts/had100_baselines.py. READ, never typed: an earlier revision of this
#: script hard-coded 0.9403 for published GRX from memory, which is wrong by
#: 0.04 and produced a decomposition claiming ~90% of the margin came from
#: preprocessing. The real figure is 0.97986 and the repo had it all along.
BASELINES_JSON = Path("results/had100_baselines.json")


def published_grx() -> float:
    if not BASELINES_JSON.is_file():
        raise SystemExit(
            f"{BASELINES_JSON} is missing; run scripts/had100_baselines.py first. "
            "This value is not hard-coded on purpose.")
    table = json.loads(BASELINES_JSON.read_text(encoding="utf-8"))
    table = table.get("table", table)
    return float(table["GRX"]["band50"])


def sliced(full: BandReducer, k: int) -> BandReducer:
    """The first k components of an already-fitted basis."""
    r = BandReducer(n_components=k)
    r.mean_ = full.mean_
    r.components_ = full.components_[:k]
    r.explained_variance_ = full.explained_variance_[:k]
    if full.explained_variance_ratio_ is not None:
        r.explained_variance_ratio_ = full.explained_variance_ratio_[:k]
    return r


def spread(x) -> dict:
    a = np.asarray(x, dtype=float)
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()),
            "values": [float(v) for v in a]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/revision/rx_validation.json")
    args = p.parse_args(argv)

    cfg = load_config(CONFIG, HyperspectralConfig)
    names = crop_names()
    groups = [source_scene(n, set(names)) for n in names]
    truths = load_truths(names)
    uniq = sorted(set(groups))
    print(f"{len(names)} crops -> {len(uniq)} source scenes\n")

    scenes = list(prepared_scenes(cfg, None))
    assert [n for n, _, _ in scenes] == names, "scene order mismatch"
    cubes = [c for _, c, _ in scenes]
    print(f"cube width after the band preset: {cubes[0].shape[-1]} bands\n")

    full = corpus_reducer(
        replace(cfg, preprocess=replace(cfg.preprocess, pca_n_components=max(K_GRID))),
        Path(cfg.preprocess.cache_dir))
    print(f"corpus basis fitted at K={max(K_GRID)}, sliced for the rest\n")

    def rx_of(reduced_cubes):
        return [rx_score(r, ridge=cfg.rx.ridge, shrinkage=cfg.rx.shrinkage)
                for r in reduced_cubes]

    # --- the three variants -------------------------------------------------
    variants = {}
    print("variant                       crop mAUC   scene mAUC")
    variants["raw50"] = rx_of(cubes)
    for k in (K_REPORTED, K_MODAL):
        variants[f"per_scene_pca{k}"] = rx_of([
            BandReducer(n_components=k, variance=cfg.preprocess.pca_variance)
            .fit(c).transform(c) for c in cubes])

    auc_by_k = {}
    for k in K_GRID:
        maps = rx_of([sliced(full, k).transform(c) for c in cubes])
        auc_by_k[k] = per_image_auc(maps, truths)
        if k in (K_REPORTED, K_MODAL):
            variants[f"corpus_pca{k}"] = maps

    labels = {"raw50": "(a) raw 50 bands = GRX",
              "per_scene_pca15": "(b) per-scene PCA-15",
              "corpus_pca15": "(c) corpus PCA-15",
              "per_scene_pca8": "(b8) per-scene PCA-8",
              "corpus_pca8": "(d) corpus PCA-8 = CV K"}
    aucs, scene_aucs = {}, {}
    for key, maps in variants.items():
        a = per_image_auc(maps, truths)
        s, scene_keys = by_scene(a, groups)
        aucs[key], scene_aucs[key] = a, s
        print(f"{labels[key]:<30} {np.nanmean(a):.4f}      {s.mean():.4f}")
    grx = published_grx()
    print(f"\nHAD100's published GRX, band50 column: {grx:.6f}  "
          f"(read from {BASELINES_JSON})")
    print(f"  (a) differs from it by {np.nanmean(aucs['raw50']) - grx:+.6f}; "
          "the band preset is equivalent to taking the first 50 bands, so the "
          "residual is the 99.9th-percentile clip that the reproduction does "
          "not apply.")

    # --- pairwise tests on the 94 scenes ------------------------------------
    print(f"\n=== paired tests on {len(scene_keys)} source scenes ===")
    tests = []
    for base, other in (("corpus_pca15", "raw50"),
                        ("corpus_pca15", "per_scene_pca15"),
                        ("corpus_pca8", "raw50"),
                        ("corpus_pca8", "per_scene_pca8")):
        rng = np.random.default_rng(7)
        t = paired_test(scene_aucs[base], scene_aucs[other],
                        labels[base], labels[other], rng)
        tests.append(t)
        print(f"\n  {t['comparison']}")
        print(f"    mean diff {t['mean_difference']:+.5f}   95% CI "
              f"[{t['ci95_bootstrap'][0]:+.5f}, {t['ci95_bootstrap'][1]:+.5f}]")
        print(f"    win/tie/loss {t['wins']}/{t['ties']}/{t['losses']}   "
              f"W = {t['wilcoxon_statistic']:.1f},  p = {t['wilcoxon_p']:.4g}")
        print(f"    => {'SIGNIFICANT' if t['significant_at_0.05'] else 'NOT significant'}")

    decomp = {}
    for k, bl, cl in ((K_REPORTED, "per_scene_pca15", "corpus_pca15"),
                      (K_MODAL, "per_scene_pca8", "corpus_pca8")):
        total = scene_aucs[cl].mean() - scene_aucs["raw50"].mean()
        trunc = scene_aucs[bl].mean() - scene_aucs["raw50"].mean()
        share = trunc / total if total else float("nan")
        decomp[k] = {"total_raw_to_corpus": float(total),
                     "truncation_only": float(trunc),
                     "corpus_basis_only": float(total - trunc),
                     "truncation_share": float(share)}
        print(f"\n  decomposition at K={k} (scene level):")
        print(f"    total, raw -> corpus       {total:+.5f}")
        print(f"    truncation alone           {trunc:+.5f}  = {share * 100:.0f}%")
        print(f"    corpus basis alone         {total - trunc:+.5f}")
    print(f"\n  gap to PUBLISHED GRX at K={K_MODAL} (crop level): "
          f"{np.nanmean(aucs['corpus_pca8']) - grx:+.5f}  -- all of it from PCA, "
          "since (a) already reproduces published GRX.")

    # --- K by grouped CV ----------------------------------------------------
    print(f"\n=== K by {N_FOLDS}-fold CV grouped by source scene ===")
    order = np.random.default_rng(FOLD_SEED).permutation(len(uniq))
    fold_of = {uniq[j]: int(i % N_FOLDS) for i, j in enumerate(order)}
    crop_fold = np.array([fold_of[g] for g in groups])

    folds = []
    for f in range(N_FOLDS):
        tr, te = crop_fold != f, crop_fold == f
        tr_g = [g for g, m in zip(groups, tr) if m]
        te_g = [g for g, m in zip(groups, te) if m]
        assert not (set(tr_g) & set(te_g)), "scene leaked across folds"
        curve = {k: float(by_scene(auc_by_k[k][tr], tr_g)[0].mean()) for k in K_GRID}
        best = max(curve, key=curve.get)
        held, _ = by_scene(auc_by_k[best][te], te_g)
        held15, _ = by_scene(auc_by_k[K_REPORTED][te], te_g)
        folds.append({"fold": f, "k_selected": best,
                      "heldout_mauc": float(held.mean()),
                      "heldout_mauc_at_k15": float(held15.mean()),
                      "train_curve": curve})
        print(f"  fold {f}: K={best:<3} held-out {held.mean():.4f}   "
              f"(K=15 -> {held15.mean():.4f})")

    hv = spread([f["heldout_mauc"] for f in folds])
    h15 = spread([f["heldout_mauc_at_k15"] for f in folds])
    ks = [f["k_selected"] for f in folds]
    print(f"\n  K selected per fold  : {ks}")
    print(f"  held-out at CV K     : {hv['mean']:.4f} +/- {hv['std']:.4f}  "
          f"(median {hv['median']:.4f})")
    print(f"  held-out at K=15     : {h15['mean']:.4f} +/- {h15['std']:.4f}  "
          f"(median {h15['median']:.4f})")
    verdict = ("K=15 is confirmed by cross-validation"
               if all(k == K_REPORTED for k in ks)
               else f"CV does NOT confirm K=15; it selects {sorted(set(ks))}")
    print(f"  => {verdict}")

    print("\n  full-data mAUC by K (diagnostic, not a selection basis):")
    for k in K_GRID:
        s, _ = by_scene(auc_by_k[k], groups)
        mark = "  <- reported" if k == K_REPORTED else ""
        print(f"    K={k:<3} crop {np.nanmean(auc_by_k[k]):.4f}   scene {s.mean():.4f}{mark}")

    report = {
        "config": CONFIG,
        "n_crops": len(names), "n_source_scenes": len(scene_keys),
        "published_grx_band50": grx,
        "variants": {k: {"crop_mauc": float(np.nanmean(v)),
                         "scene_mauc": float(scene_aucs[k].mean()),
                         "label": labels[k]}
                     for k, v in aucs.items()},
        "paired_tests": tests,
        "gain_decomposition_by_k": {str(k): v for k, v in decomp.items()},
        "gap_to_published_grx_at_k8_crop": float(np.nanmean(aucs["corpus_pca8"]) - grx),
        "k_selection": {
            "grid": list(K_GRID), "n_folds": N_FOLDS, "fold_seed": FOLD_SEED,
            "k_selected_per_fold": ks,
            "heldout_at_cv_k": hv, "heldout_at_k15": h15,
            "verdict": verdict,
            "folds": folds,
            "full_data_scene_mauc_by_k": {
                str(k): float(by_scene(auc_by_k[k], groups)[0].mean()) for k in K_GRID},
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
