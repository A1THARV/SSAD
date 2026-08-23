"""M2 ablation battery: train each distinct model once, evaluate every combination.

    python scripts/run_ablations.py --out results/m2

Structured around what each result actually depends on, so nothing is recomputed
that cannot change:

* An **RX map** depends on ``(K, pca_basis, ridge, shrinkage)`` -- seconds.
* An **autoencoder map** depends on ``(K, whiten, lambda, sam_domain, seed)`` --
  minutes, dominated by training.
* A **fused map** depends on those two plus ``(alpha, mode)`` and costs nothing.

So the alpha sweep is free once the branches exist. Running it as separate
end-to-end invocations would retrain an identical network per alpha, and -- worse
-- each retrain would land on slightly different weights, so the alpha curve
would carry training noise it has no business carrying.

Every score map is cached to disk, so a re-run after adding one variant costs
only that variant.
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

from pipeline.cli.run_hyperspectral import (  # noqa: E402
    cached_patches,
    corpus_reducer,
)
from pipeline.config import (  # noqa: E402
    HyperspectralConfig,
    load_config,
    resolve_device,
    set_seeds,
)
from pipeline.evaluation.metrics import mean_per_image_auc  # noqa: E402
from pipeline.hyperspectral.autoencoder import (  # noqa: E402
    reconstruction_error_map,
    train_autoencoder,
)
from pipeline.hyperspectral.loader import list_corpus, load_cube  # noqa: E402
from pipeline.hyperspectral.preprocess import BandReducer, mask_and_clip  # noqa: E402
from pipeline.hyperspectral.rx_detector import rx_score  # noqa: E402
from pipeline.hyperspectral.score_fusion import fuse_scores  # noqa: E402


# --------------------------------------------------------------------------
# The variants
# --------------------------------------------------------------------------

# One-at-a-time from the baseline, not a full factorial. Each row isolates a
# single decision that was previously assumed rather than measured; a factorial
# over six factors would cost eight hours to answer questions nobody asked.
AE_VARIANTS = [
    # name                K   latent  whiten  lam   sam_domain   what it tests
    ("baseline",          15, 8,      False,  0.1,  "spectral"),
    ("whiten",            15, 8,      True,   0.1,  "spectral"),
    ("no_sam",            15, 8,      False,  0.0,  "spectral"),
    ("sam_latent",        15, 8,      False,  0.1,  "latent"),
    ("k8",                8,  4,      False,  0.1,  "spectral"),
    ("k8_whiten",         8,  4,      True,   0.1,  "spectral"),
]

RX_VARIANTS = [
    # name             K   pca_basis
    ("rx_k15_scene",   15, "per_scene"),
    ("rx_k15_corpus",  15, "corpus"),
    ("rx_k8_scene",    8,  "per_scene"),
    ("rx_k8_corpus",   8,  "corpus"),
]

ALPHAS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]


# --------------------------------------------------------------------------
# Scene iteration
# --------------------------------------------------------------------------


def prepared_scenes(cfg: HyperspectralConfig, limit: int | None):
    """Yield ``(name, clipped_cube, ground_truth)`` once, band-masked and clipped.

    Band masking and clipping are identical across every variant, so they happen
    here rather than inside each one.
    """
    paths = list_corpus(cfg.dataset.path)
    if limit:
        paths = paths[:limit]
    for path in paths:
        cube = load_cube(cfg.dataset, path)
        yield cube.name, mask_and_clip(cube, cfg.preprocess), cube.ground_truth


# --------------------------------------------------------------------------
# Branch maps, cached
# --------------------------------------------------------------------------


def rx_maps(cfg, name, k, basis, scenes, cache: Path, corpus_pcas) -> list[np.ndarray]:
    blob = cache / f"{name}.npz"
    if blob.is_file():
        data = np.load(blob)
        return [data[f"m{i}"] for i in range(len(data.files))]

    print(f"  RX  {name} (K={k}, {basis}) ...", flush=True)
    t0 = time.perf_counter()
    pre = replace(cfg.preprocess, pca_n_components=k)
    maps = []
    for _, clipped, _ in scenes:
        if basis == "corpus":
            reduced = corpus_pcas[k].transform(clipped)
        else:
            reduced = (
                BandReducer(n_components=k, variance=pre.pca_variance)
                .fit(clipped)
                .transform(clipped)
            )
        maps.append(rx_score(reduced, ridge=cfg.rx.ridge, shrinkage=cfg.rx.shrinkage))
    blob.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(blob, **{f"m{i}": m for i, m in enumerate(maps)})
    print(f"      {time.perf_counter() - t0:.1f}s")
    return maps


def ae_maps(cfg, name, k, latent, whiten, lam, sam_domain, scenes, cache, corpus_pcas, args):
    blob = cache / f"ae_{name}.npz"
    meta_path = cache / f"ae_{name}.json"
    if blob.is_file():
        data = np.load(blob)
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        return [data[f"m{i}"] for i in range(len(data.files))], meta

    print(
        f"  AE  {name} (K={k}, latent={latent}, whiten={whiten}, lam={lam}, "
        f"sam={sam_domain}) ...",
        flush=True,
    )
    variant = replace(
        cfg,
        preprocess=replace(cfg.preprocess, pca_n_components=k),
        autoencoder=replace(
            cfg.autoencoder,
            latent_dim=latent,
            whiten_input=whiten,
            lam=lam,
            sam_domain=sam_domain,
            epochs=args.epochs or cfg.autoencoder.epochs,
        ),
    )
    reducer = corpus_pcas[k]
    device = resolve_device(variant.autoencoder.device)
    set_seeds(variant.seed, variant.deterministic)

    patches = cached_patches(
        variant,
        reducer,
        list_corpus(variant.autoencoder.training_corpus),
        Path(variant.preprocess.cache_dir),
        args,
    )
    model, history = train_autoencoder(
        patches, variant.autoencoder, reducer=reducer, device=device, seed=variant.seed
    )
    print(
        f"      trained {history.seconds:.0f}s  "
        f"loss {history.loss[0]:.4f} -> {history.loss[-1]:.4f}  "
        f"sam {history.sam[0]:.4f} -> {history.sam[-1]:.4f} rad"
    )

    t0 = time.perf_counter()
    maps = [
        reconstruction_error_map(
            model,
            reducer.transform(clipped, whiten=whiten),
            patch_size=variant.autoencoder.patch_size,
            device=device,
        )
        for _, clipped, _ in scenes
    ]
    print(f"      scored {len(maps)} scenes in {time.perf_counter() - t0:.1f}s")

    blob.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(blob, **{f"m{i}": m for i, m in enumerate(maps)})
    meta = history.as_dict()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return maps, meta


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/had100.yaml")
    p.add_argument("--out", default="results/m2")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--refit-pca", action="store_true")
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated variant names to run, e.g. --only baseline. Restricts "
             "both tables; the fusion sweep then covers just those. Use when a "
             "re-run only needs to answer one question -- the alpha sweep on a "
             "different config needs one AE, not six.",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config, HyperspectralConfig)
    wanted = {n.strip() for n in args.only.split(",")} if args.only else None
    ae_variants = [v for v in AE_VARIANTS if wanted is None or v[0] in wanted]
    rx_variants = [v for v in RX_VARIANTS if wanted is None or v[0] in wanted]
    if not ae_variants and not rx_variants:
        raise SystemExit(
            f"--only {args.only!r} matched no variants. Known: "
            + ", ".join([v[0] for v in AE_VARIANTS] + [v[0] for v in RX_VARIANTS])
        )
    out = Path(args.out)
    cache = out / "maps"
    out.mkdir(parents=True, exist_ok=True)

    print("loading test scenes ...", flush=True)
    scenes = list(prepared_scenes(cfg, args.limit))
    truths = [gt for _, _, gt in scenes]
    print(f"  {len(scenes)} scenes")

    widths = sorted({k for _, k, *_ in rx_variants} | {v[1] for v in ae_variants})
    corpus_pcas = {}
    for k in widths:
        corpus_pcas[k] = corpus_reducer(
            replace(cfg, preprocess=replace(cfg.preprocess, pca_n_components=k)),
            Path(cfg.preprocess.cache_dir),
            refit=args.refit_pca,
        )

    results: dict[str, dict] = {}

    print("\nRX branch")
    rx_by_name = {}
    for name, k, basis in rx_variants:
        maps = rx_maps(cfg, name, k, basis, scenes, cache, corpus_pcas)
        rx_by_name[name] = maps
        mauc, n = mean_per_image_auc(maps, truths)
        results[name] = {"mauc": mauc, "n": n, "kind": "rx", "K": k, "basis": basis}
        print(f"      mAUC {mauc:.4f}")

    print("\nAutoencoder branch")
    ae_by_name = {}
    for name, k, latent, whiten, lam, sam_domain in ae_variants:
        maps, meta = ae_maps(
            cfg, name, k, latent, whiten, lam, sam_domain, scenes, cache, corpus_pcas, args
        )
        ae_by_name[name] = (maps, k)
        mauc, n = mean_per_image_auc(maps, truths)
        results[f"ae_{name}"] = {
            "mauc": mauc, "n": n, "kind": "ae", "K": k, "latent": latent,
            "whiten": whiten, "lam": lam, "sam_domain": sam_domain,
            "final_loss": meta.get("loss", [None])[-1],
            "final_sam_rad": meta.get("sam_rad", [None])[-1],
            "train_seconds": meta.get("seconds"),
            "peak_vram_gb": meta.get("peak_vram_gb"),
        }
        print(f"      mAUC {mauc:.4f}")

    print("\nFusion sweep (free -- both branches already scored)")
    fusion: dict[str, dict] = {}
    for ae_name, (ae_map, k) in ae_by_name.items():
        # Pair each AE with the RX branch the CONFIG actually uses. Hardcoding
        # per_scene here is what put the first alpha sweep on a weaker RX than
        # the headline run's, which shifts the whole curve.
        suffix = "corpus" if cfg.rx.pca_basis == "corpus" else "scene"
        rx_name = f"rx_k{k}_{suffix}"
        if rx_name not in rx_by_name:
            print(f"  {ae_name:<12} skipped -- {rx_name} not among the RX variants run")
            continue
        rx_map = rx_by_name[rx_name]
        row = {}
        for alpha in ALPHAS:
            fused = [
                fuse_scores(r, a, mode=cfg.score_fusion.mode, alpha=alpha)
                for r, a in zip(rx_map, ae_map)
            ]
            mauc, _ = mean_per_image_auc(fused, truths)
            row[f"{alpha:.2f}"] = mauc
        fusion[ae_name] = row
        best = max(row, key=row.get)
        print(
            f"  {ae_name:<12} best alpha {best} -> {row[best]:.4f}   "
            f"(alpha=0.50 -> {row['0.50']:.4f}, RX alone -> {row['1.00']:.4f})"
        )

    payload = {
        "variants": results,
        "fusion_alpha_sweep": fusion,
        "alphas": ALPHAS,
        "n_scenes": len(scenes),
        "config": args.config,
        "rx_pca_basis": cfg.rx.pca_basis,
        "good_bands_preset": cfg.preprocess.good_bands_preset,
    }
    (out / "ablations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {out / 'ablations.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
