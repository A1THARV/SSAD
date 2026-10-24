"""Step 2 — B repeated across seeds 0-4 (INDICON revision).

    python scripts/revision_b_seeds.py

The first run of B gave held-out 0.9874 +/- 0.0051, and that number is not yet
defensible: it came from seed 0's score maps alone, so the +/- is **fold**
variance with the autoencoder held fixed. A1 then measured the fused seed std at
0.0158 -- three times the fold spread -- which means the quoted interval was
dominated by the source of variation it excluded.

This repeats the scene-grouped cross-validation for every seed, reusing the score
maps A1 already wrote. No retraining.

Two choices worth stating:

**The fold assignment is identical across seeds.** Folds are drawn once from a
fixed generator, so the spread between seeds is the autoencoder changing and
nothing else. Re-drawing folds per seed would confound the two.

**The per-seed significance test uses OUT-OF-FOLD alpha.** A single alpha fitted
on all 94 scenes and then tested on the same 94 would be the test-set tuning R3
objected to. Instead each scene is fused with the alpha selected by the four
folds that excluded it, so every scene's score is out-of-sample, and the Wilcoxon
then runs over all 94.

If fusion beats RX on some seeds and not others, it does not beat RX.
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

from pipeline.hyperspectral.score_fusion import fuse_scores  # noqa: E402
from revision_stats import (  # noqa: E402
    ALPHA_GRID,
    N_FOLDS,
    by_scene,
    crop_names,
    load_maps,
    load_truths,
    paired_test,
    per_image_auc,
    source_scene,
)

SEEDS = (0, 1, 2, 3, 4)
MAPS = Path("results/revision/maps")
RX_MAPS = MAPS / "band50_rx_k15_corpus.npz"
FOLD_SEED = 0          # fixed, so folds are identical across seeds


def spread(x: np.ndarray) -> dict:
    """Mean, std, median and the raw values.

    One collapsed seed dominates a standard deviation, so the median is
    reported beside it everywhere and the per-seed values are never summarised
    away.
    """
    a = np.asarray(x, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(a.min()), "max": float(a.max()),
        "values": [float(v) for v in a],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/revision/b_seeds.json")
    args = p.parse_args(argv)

    names = crop_names()
    nameset = set(names)
    groups = [source_scene(n, nameset) for n in names]
    truths = load_truths(names)
    uniq = sorted(set(groups))

    rx_maps = load_maps(RX_MAPS)
    rx_auc = per_image_auc(rx_maps, truths)
    rx_scene, scene_keys = by_scene(rx_auc, groups)
    print(f"{len(names)} crops -> {len(uniq)} source scenes")
    print(f"RX (deterministic, no seed): scene-level mAUC {rx_scene.mean():.4f}\n")

    # One fold assignment, shared by every seed.
    order = np.random.default_rng(FOLD_SEED).permutation(len(uniq))
    fold_of = {uniq[j]: int(i % N_FOLDS) for i, j in enumerate(order)}
    crop_fold = np.array([fold_of[g] for g in groups])
    scene_fold = np.array([fold_of[k] for k in scene_keys])

    per_seed, all_heldout, all_alpha = [], [], []
    for seed in SEEDS:
        ae_path = MAPS / f"band50_ae_baseline_s{seed}.npz"
        if not ae_path.is_file():
            print(f"seed {seed}: MISSING {ae_path} -- skipped")
            continue
        ae_maps = load_maps(ae_path)
        assert len(ae_maps) == len(names), f"map count mismatch at seed {seed}"

        auc_by_alpha = {}
        for a in ALPHA_GRID:
            maps = [fuse_scores(r, x, alpha=a) for r, x in zip(rx_maps, ae_maps)]
            auc_by_alpha[a] = per_image_auc(maps, truths)

        folds, oof_alpha = [], {}
        for f in range(N_FOLDS):
            tr, te = crop_fold != f, crop_fold == f
            tr_groups = [g for g, m in zip(groups, tr) if m]
            te_groups = [g for g, m in zip(groups, te) if m]
            assert not (set(tr_groups) & set(te_groups)), "scene leaked across folds"

            scores = {a: float(by_scene(auc_by_alpha[a][tr], tr_groups)[0].mean())
                      for a in ALPHA_GRID}
            best = max(scores, key=scores.get)
            held, _ = by_scene(auc_by_alpha[best][te], te_groups)
            held05, _ = by_scene(auc_by_alpha[0.5][te], te_groups)
            rxh, _ = by_scene(rx_auc[te], te_groups)
            for g in set(te_groups):
                oof_alpha[g] = best
            folds.append({
                "fold": f, "alpha_selected": best,
                "n_test_scenes": len(set(te_groups)),
                "heldout_mauc": float(held.mean()),
                "heldout_mauc_at_alpha_0.5": float(held05.mean()),
                "heldout_rx_mauc": float(rxh.mean()),
            })

        hv = np.array([f["heldout_mauc"] for f in folds])
        sel = [f["alpha_selected"] for f in folds]
        all_heldout.extend(hv.tolist())
        all_alpha.extend(sel)

        # Out-of-fold fused AUC, scene by scene: each scene uses the alpha its
        # own fold never saw.
        oof = np.array([auc_by_alpha[oof_alpha[k]][[i for i, g in enumerate(groups)
                                                    if g == k]].mean()
                        for k in scene_keys])
        rng = np.random.default_rng(1000 + seed)
        test = paired_test(oof, rx_scene, f"fused(oof-alpha, seed {seed})", "RX", rng)

        beat = test["significant_at_0.05"] and test["mean_difference"] > 0
        per_seed.append({
            "seed": seed, "folds": folds,
            "alpha_selected_per_fold": sel,
            "heldout": spread(hv),
            "heldout_alpha05": spread(np.array([f["heldout_mauc_at_alpha_0.5"] for f in folds])),
            "oof_fused_scene_mauc": float(oof.mean()),
            "paired_vs_rx": test,
            "beats_rx_significantly": bool(beat),
        })
        print(f"seed {seed}:  alpha {sel}   held-out {hv.mean():.4f} +/- "
              f"{hv.std(ddof=1):.4f}  (median {np.median(hv):.4f})")
        print(f"          out-of-fold fused {oof.mean():.4f} vs RX {rx_scene.mean():.4f}   "
              f"diff {test['mean_difference']:+.5f}  p = {test['wilcoxon_p']:.4g}  "
              f"W/T/L {test['wins']}/{test['ties']}/{test['losses']}  "
              f"=> {'BEATS RX' if beat else 'does NOT beat RX'}")

    pooled = spread(np.array(all_heldout))
    n_beat = sum(s["beats_rx_significantly"] for s in per_seed)
    alpha_counts = {str(a): all_alpha.count(a) for a in sorted(set(all_alpha))}

    print("\n" + "=" * 70)
    print(f"held-out mAUC over {len(per_seed)} seeds x {N_FOLDS} folds "
          f"= {len(all_heldout)} values")
    print(f"  mean   {pooled['mean']:.4f} +/- {pooled['std']:.4f}")
    print(f"  median {pooled['median']:.4f}   range [{pooled['min']:.4f}, {pooled['max']:.4f}]")
    print(f"alpha selected, pooled over seeds x folds: {alpha_counts}")
    print(f"RX (no seed): {rx_scene.mean():.4f}")
    print(f"\nseeds beating RX significantly: {n_beat} of {len(per_seed)}")
    if n_beat < len(per_seed):
        print("  => fusion does not survive. It wins on some draws of the "
              "autoencoder and not others.")
    print("=" * 70)

    report = {
        "seeds": [s["seed"] for s in per_seed],
        "n_folds": N_FOLDS,
        "fold_seed": FOLD_SEED,
        "fold_assignment": "identical across seeds, so spread is seed variance only",
        "alpha_grid": list(ALPHA_GRID),
        "rx_scene_mauc": float(rx_scene.mean()),
        "heldout_pooled_seeds_x_folds": pooled,
        "heldout_per_seed_mean": spread(
            np.array([s["heldout"]["mean"] for s in per_seed])),
        "alpha_counts_pooled": alpha_counts,
        "n_seeds_beating_rx_significantly": n_beat,
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "note": ("The per-seed test fuses each scene with the alpha chosen by the "
                 "four folds that excluded it, so no scene contributes to the "
                 "selection of its own alpha."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
