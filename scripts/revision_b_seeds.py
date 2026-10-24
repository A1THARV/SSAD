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
FOLD_SEED = 0          # fixed, so folds are identical across seeds

#: RX's PCA width for the fused branch AND the baseline it is judged against.
#:
#: K = 8 is what grouped cross-validation selects (unanimously, all five folds);
#: K = 15 was the reported value and was chosen with sight of HAD100's own
#: curve. The autoencoder's own input stays at K = 15 -- its architecture and
#: its corpus patch cache are built for that width, and re-deriving them would
#: be a different experiment rather than the same one measured correctly. So a
#: fused score here combines an RX map at K = 8 with an AE map at K = 15; they
#: are independent score maps and nothing requires them to share a width.
RX_K = 8
AE_INPUT_K = 15


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
    p.add_argument("--rx-k", type=int, default=RX_K,
                   help="PCA width of the RX branch and of the RX baseline")
    args = p.parse_args(argv)
    rx_path = MAPS / f"band50_rx_k{args.rx_k}_corpus.npz"

    names = crop_names()
    nameset = set(names)
    groups = [source_scene(n, nameset) for n in names]
    truths = load_truths(names)
    uniq = sorted(set(groups))

    rx_maps = load_maps(rx_path)
    rx_auc = per_image_auc(rx_maps, truths)
    rx_scene, scene_keys = by_scene(rx_auc, groups)
    print(f"{len(names)} crops -> {len(uniq)} source scenes")
    print(f"RX branch and baseline: corpus PCA K={args.rx_k}  ({rx_path.name})")
    print(f"  autoencoder input stays at K={AE_INPUT_K}; the two score maps are "
          "independent and need not share a width")
    print(f"RX (deterministic, no seed): crop {np.nanmean(rx_auc):.4f}   "
          f"scene {rx_scene.mean():.4f}\n")

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
        crop_oof = np.array([auc_by_alpha[oof_alpha[g]][i]
                             for i, g in enumerate(groups)])
        oof, _ = by_scene(crop_oof, groups)
        rng = np.random.default_rng(1000 + seed)
        test = paired_test(oof, rx_scene, f"fused(oof-alpha, seed {seed})", "RX", rng)
        beat = test["significant_at_0.05"] and test["mean_difference"] > 0

        # The three branches the final table reports, at both levels, each
        # tested against the SAME RX baseline the fused score uses.
        ae_crop = per_image_auc(ae_maps, truths)
        branches = {}
        for bname, crop_v in (("ae_alone", ae_crop),
                              ("fused_alpha05", auc_by_alpha[0.5]),
                              ("fused_cv_alpha", crop_oof)):
            scene_v, _ = by_scene(crop_v, groups)
            branches[bname] = {
                "crop_mauc": float(np.nanmean(crop_v)),
                "scene_mauc": float(scene_v.mean()),
                "vs_rx": paired_test(scene_v, rx_scene, bname,
                                     f"RX K={args.rx_k}",
                                     np.random.default_rng(2000 + seed)),
            }

        per_seed.append({
            "seed": seed, "folds": folds,
            "alpha_selected_per_fold": sel,
            "heldout": spread(hv),
            "heldout_alpha05": spread(np.array([f["heldout_mauc_at_alpha_0.5"] for f in folds])),
            "oof_fused_scene_mauc": float(oof.mean()),
            "oof_fused_crop_mauc": float(np.nanmean(crop_oof)),
            "branches": branches,
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

    # --- the final HAD100 table ------------------------------------------
    BR = (("ae_alone", "AE alone"),
          ("fused_alpha05", "fused, alpha = 0.5"),
          ("fused_cv_alpha", "fused, CV alpha"))
    print(f"\nFINAL HAD100 TABLE  (every fused number uses RX at corpus "
          f"PCA K={args.rx_k}; AE input K={AE_INPUT_K})")
    table = {}
    for level in ("crop", "scene"):
        key = f"{level}_mauc"
        n = len(names) if level == "crop" else len(scene_keys)
        print(f"\n  --- {level} level ({n} {'crops' if level == 'crop' else 'source scenes'})")
        print(f"    {'branch':<20}{'mean +/- std':>18}{'median':>10}   per-seed")
        rxv = float(np.nanmean(rx_auc)) if level == "crop" else float(rx_scene.mean())
        print(f"    {'RX (no seed)':<20}{rxv:>12.4f}{'':>6}{rxv:>10.4f}   deterministic")
        for bname, lab in BR:
            v = np.array([s["branches"][bname][key] for s in per_seed])
            sp = spread(v)
            print(f"    {lab:<20}{sp['mean']:>12.4f} +/-{sp['std']:>5.4f}"
                  f"{sp['median']:>10.4f}   "
                  + " ".join(f"{x:.4f}" for x in sp["values"]))
            table.setdefault(bname, {})[level] = sp
        table.setdefault("rx", {})[level] = rxv

    print(f"\n  --- Wilcoxon vs RX K={args.rx_k}, {len(scene_keys)} scenes, per seed")
    print(f"    {'branch':<20}{'seed':>5}{'diff':>11}{'W/T/L':>12}{'p':>12}")
    for bname, lab in BR:
        for s in per_seed:
            t = s["branches"][bname]["vs_rx"]
            wtl = "{}/{}/{}".format(t["wins"], t["ties"], t["losses"])
            print(f"    {lab:<20}{s['seed']:>5}{t['mean_difference']:>+11.5f}"
                  f"{wtl:>12}{t['wilcoxon_p']:>12.3g}")

    report = {
        "seeds": [s["seed"] for s in per_seed],
        "rx_k": args.rx_k,
        "ae_input_k": AE_INPUT_K,
        "rx_maps": str(rx_path),
        "final_table": table,
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
