"""A1 — seed variance: retrain the autoencoder with seeds 0-4 (INDICON revision).

    python scripts/revision_seed_variance.py

Reviewer 1 asks for repeatability across training runs. **Same-seed determinism
is not that** -- the project already pins bit-identical reruns under a fixed
seed, which says nothing about how much the result moves when the seed changes.
This retrains from scratch at seeds 0-4 and reports mean +/- std.

The seed varies **both** the patch draw and the network initialisation, which is
the honest reading of "repeatability across training runs": a practitioner
rerunning the pipeline gets a new sample of the corpus as well as new weights.
Holding the patches fixed would report init variance alone and understate it.

Per-image AUC arrays and score maps are written out so A2 (paired significance)
and B (alpha cross-validation) can reuse them without retraining.

**If the seed std approaches an ablation delta, that delta is not
distinguishable** -- the summary marks those explicitly rather than leaving the
reader to compare a mean against a spread.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from pipeline.config import HyperspectralConfig, load_config, resolve_device, set_seeds  # noqa: E402
from pipeline.hyperspectral.autoencoder import (  # noqa: E402
    reconstruction_error_map,
    train_autoencoder,
)
from pipeline.hyperspectral.loader import list_corpus  # noqa: E402
from pipeline.hyperspectral.preprocess import BandReducer  # noqa: E402
from pipeline.hyperspectral.rx_detector import rx_score  # noqa: E402
from pipeline.hyperspectral.score_fusion import fuse_scores  # noqa: E402
from pipeline.cli.run_hyperspectral import cached_patches, corpus_reducer  # noqa: E402
from run_ablations import AE_VARIANTS, prepared_scenes  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
CONFIGS = (
    ("band50", "configs/had100_band50.yaml"),
    ("band276", "configs/had100.yaml"),
)
ALPHA_APRIORI = 0.5


class _Args:
    """cached_patches() wants an args object; only refit_pca is read."""
    refit_pca = False


def per_image_auc(maps, truths) -> np.ndarray:
    """AUC per image, NaN where the image holds a single class."""
    out = np.full(len(maps), np.nan)
    for i, (m, g) in enumerate(zip(maps, truths)):
        g = np.asarray(g, dtype=bool).ravel()
        n = int(g.sum())
        if 0 < n < g.size:
            out[i] = roc_auc_score(g, np.asarray(m, dtype=np.float64).ravel())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/revision/a1_seed_variance.json")
    p.add_argument("--maps", default="results/revision/maps")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = p.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]

    map_dir = Path(args.maps)
    map_dir.mkdir(parents=True, exist_ok=True)
    report = {"seeds": seeds, "alpha_apriori": ALPHA_APRIORI, "configs": {}}
    t_start = time.perf_counter()

    for tag, cfg_path in CONFIGS:
        cfg = load_config(cfg_path, HyperspectralConfig)
        print(f"\n{'=' * 72}\n{tag}  ({cfg_path})  rx.pca_basis={cfg.rx.pca_basis}\n{'=' * 72}",
              flush=True)

        scenes = list(prepared_scenes(cfg, args.limit))
        names = [n for n, _, _ in scenes]
        truths = [g for _, _, g in scenes]
        print(f"  {len(scenes)} scenes", flush=True)

        widths = sorted({v[1] for v in AE_VARIANTS})
        pcas = {
            k: corpus_reducer(
                replace(cfg, preprocess=replace(cfg.preprocess, pca_n_components=k)),
                Path(cfg.preprocess.cache_dir),
            )
            for k in widths
        }

        # --- RX once per width; it has no seed ----------------------------
        rx_maps, rx_auc = {}, {}
        for k in widths:
            blob = map_dir / f"{tag}_rx_k{k}_{cfg.rx.pca_basis}.npz"
            if blob.is_file():
                d = np.load(blob)
                maps = [d[f"m{i}"] for i in range(len(d.files))]
            else:
                maps = []
                for _, clipped, _ in scenes:
                    if cfg.rx.pca_basis == "corpus":
                        reduced = pcas[k].transform(clipped)
                    else:
                        reduced = (
                            BandReducer(n_components=k, variance=cfg.preprocess.pca_variance)
                            .fit(clipped).transform(clipped)
                        )
                    maps.append(rx_score(reduced, ridge=cfg.rx.ridge,
                                         shrinkage=cfg.rx.shrinkage))
                np.savez_compressed(blob, **{f"m{i}": m for i, m in enumerate(maps)})
            rx_maps[k] = maps
            rx_auc[k] = per_image_auc(maps, truths)
            print(f"  RX K={k:<3} mAUC {np.nanmean(rx_auc[k]):.4f}", flush=True)

        cfg_block = {
            "config": cfg_path,
            "rx_pca_basis": cfg.rx.pca_basis,
            "n_scenes": len(scenes),
            "scene_names": names,
            "rx": {str(k): {"mauc": float(np.nanmean(v)),
                            "per_image_auc": [None if np.isnan(x) else float(x) for x in v]}
                   for k, v in rx_auc.items()},
            "variants": {},
        }

        for vname, k, latent, whiten, lam, sam_domain in AE_VARIANTS:
            rows = {"ae": [], "fused": [], "train_seconds": [], "peak_vram_gb": []}
            print(f"\n  --- {vname}  (K={k}, latent={latent}, whiten={whiten}, "
                  f"lam={lam}, sam={sam_domain})", flush=True)

            for seed in seeds:
                blob = map_dir / f"{tag}_ae_{vname}_s{seed}.npz"
                variant = replace(
                    cfg,
                    seed=seed,
                    preprocess=replace(cfg.preprocess, pca_n_components=k),
                    autoencoder=replace(cfg.autoencoder, latent_dim=latent,
                                        whiten_input=whiten, lam=lam,
                                        sam_domain=sam_domain),
                )
                if blob.is_file():
                    d = np.load(blob)
                    ae_maps = [d[f"m{i}"] for i in range(len(d.files) - 2)]
                    meta = {"train_seconds": float(d["train_seconds"]),
                            "peak_vram_gb": float(d["peak_vram_gb"])}
                else:
                    reducer = pcas[k]
                    device = resolve_device(variant.autoencoder.device)
                    set_seeds(seed, variant.deterministic)
                    patches = cached_patches(
                        variant, reducer,
                        list_corpus(variant.autoencoder.training_corpus),
                        Path(variant.preprocess.cache_dir), _Args(),
                    )
                    model, hist = train_autoencoder(
                        patches, variant.autoencoder, reducer=reducer,
                        device=device, seed=seed,
                    )
                    ae_maps = [
                        reconstruction_error_map(
                            model, reducer.transform(clipped, whiten=whiten),
                            patch_size=variant.autoencoder.patch_size, device=device)
                        for _, clipped, _ in scenes
                    ]
                    meta = {"train_seconds": hist.seconds,
                            "peak_vram_gb": hist.peak_vram_gb or 0.0}
                    np.savez_compressed(
                        blob,
                        train_seconds=meta["train_seconds"],
                        peak_vram_gb=meta["peak_vram_gb"],
                        **{f"m{i}": m for i, m in enumerate(ae_maps)},
                    )

                ae_a = per_image_auc(ae_maps, truths)
                fused = [fuse_scores(r, a, mode=cfg.score_fusion.mode, alpha=ALPHA_APRIORI)
                         for r, a in zip(rx_maps[k], ae_maps)]
                fu_a = per_image_auc(fused, truths)

                rows["ae"].append(ae_a)
                rows["fused"].append(fu_a)
                rows["train_seconds"].append(meta["train_seconds"])
                rows["peak_vram_gb"].append(meta["peak_vram_gb"])
                print(f"    seed {seed}  ae {np.nanmean(ae_a):.4f}   "
                      f"fused {np.nanmean(fu_a):.4f}   "
                      f"[{time.perf_counter() - t_start:.0f}s elapsed]", flush=True)

            ae_m = np.array([np.nanmean(a) for a in rows["ae"]])
            fu_m = np.array([np.nanmean(a) for a in rows["fused"]])
            print(f"    => ae    {ae_m.mean():.4f} +/- {ae_m.std(ddof=1):.4f}")
            print(f"       fused {fu_m.mean():.4f} +/- {fu_m.std(ddof=1):.4f}")

            cfg_block["variants"][vname] = {
                "K": k, "latent": latent, "whiten": whiten, "lam": lam,
                "sam_domain": sam_domain,
                "ae_mauc_per_seed": [float(x) for x in ae_m],
                "ae_mean": float(ae_m.mean()), "ae_std": float(ae_m.std(ddof=1)),
                "fused_mauc_per_seed": [float(x) for x in fu_m],
                "fused_mean": float(fu_m.mean()), "fused_std": float(fu_m.std(ddof=1)),
                "train_seconds_median": float(np.median(rows["train_seconds"])),
                "peak_vram_gb_max": float(np.max(rows["peak_vram_gb"])),
            }
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            report["configs"][tag] = cfg_block
            Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

        report["configs"][tag] = cfg_block

    # --- Are the ablation deltas distinguishable from seed noise? ---------
    print(f"\n{'=' * 72}\nAblation deltas against seed noise\n{'=' * 72}")
    for tag, block in report["configs"].items():
        base = block["variants"].get("baseline")
        if not base:
            continue
        print(f"\n{tag}: baseline ae = {base['ae_mean']:.4f} +/- {base['ae_std']:.4f}")
        for vname, v in block["variants"].items():
            if vname == "baseline":
                continue
            delta = v["ae_mean"] - base["ae_mean"]
            pooled = float(np.sqrt(v["ae_std"] ** 2 + base["ae_std"] ** 2))
            distinguishable = abs(delta) > 2 * pooled
            v["delta_vs_baseline"] = delta
            v["pooled_seed_std"] = pooled
            v["distinguishable"] = bool(distinguishable)
            flag = "" if distinguishable else "   <-- NOT DISTINGUISHABLE from seed noise"
            print(f"  {vname:<12} delta {delta:+.4f}   pooled seed std {pooled:.4f}{flag}")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\ntotal {(time.perf_counter() - t_start) / 3600:.2f} h")
    print(f"written to {args.out}  (maps in {map_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
