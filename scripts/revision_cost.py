"""D — computational cost on the stated hardware (INDICON revision).

    python scripts/revision_cost.py

Reviewer 1 asks for cost. Per-image wall clock and peak memory for each stage,
medians over images with warm-up excluded, plus autoencoder training time for
both the corpus-trained and per-scene-trained regimes.

Warm-up matters more than it sounds: the first CUDA call in a process pays
context creation and cuDNN autotuning, which on this card is seconds. Including
it would put a one-off initialisation cost into a per-image median.
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

from pipeline.config import HyperspectralConfig, load_config, resolve_device, set_seeds  # noqa: E402
from pipeline.hyperspectral.autoencoder import (  # noqa: E402
    extract_patches,
    reconstruction_error_map,
    train_autoencoder,
)
from pipeline.hyperspectral.loader import list_corpus, load_cube  # noqa: E402
from pipeline.hyperspectral.preprocess import BandReducer, mask_and_clip  # noqa: E402
from pipeline.hyperspectral.rx_detector import rx_score  # noqa: E402
from pipeline.hyperspectral.score_fusion import fuse_scores  # noqa: E402
from pipeline.cli.run_hyperspectral import cached_patches, corpus_reducer  # noqa: E402

WARMUP = 3

#: Inference batch for Eq. 1, confirmed by the sweep below rather than assumed.
#:
#: The obvious choice is 4,096 -- a 64x64 crop in one call -- and it is the
#: WORST of the four: 13,780 ms against 1,136 ms at 1,024, because 4,096 patches
#: of 15x7x7 through 32/64/128 channels peaks at 4.35 GB on a 4 GB card and
#: spills. The library default of 1,024 is also the fastest, so the cost
#: reported here is not an artefact of an unlucky batch size.
AE_BATCH = 1024

#: Batches to compare, so the reported cost reflects a reasonable
#: implementation rather than whatever the default happened to be.
AE_BATCH_SWEEP = (256, 1024, 2048, 4096)


def sync(device: str) -> None:
    """Make an async CUDA queue finish before the clock is read.

    Without this a GPU timing measures how fast work was *enqueued*. The error
    map happens to copy to host each batch, which forces a sync anyway, but
    relying on that would make the number depend on an implementation detail of
    the function being timed.
    """
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


class _Args:
    refit_pca = False


def gpu_peak_gb(reset: bool = False) -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return 0.0
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        return 0.0


def rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 3
    except Exception:
        return float("nan")


def summarise(name, times, extra=None) -> dict:
    t = np.array(times[WARMUP:]) if len(times) > WARMUP else np.array(times)
    row = {"stage": name, "n_timed": int(t.size),
           "median_s": float(np.median(t)), "mean_s": float(t.mean()),
           "p95_s": float(np.percentile(t, 95)), "min_s": float(t.min()),
           "max_s": float(t.max()), "warmup_excluded": WARMUP}
    row.update(extra or {})
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/had100_band50.yaml")
    p.add_argument("--abu-config", default="configs/abu_urban.yaml")
    p.add_argument("--out", default="results/revision/d_cost.json")
    p.add_argument("--limit", type=int, default=30)
    args = p.parse_args(argv)

    cfg = load_config(args.config, HyperspectralConfig)
    device = resolve_device(cfg.autoencoder.device)
    K = cfg.preprocess.pca_n_components
    reducer = corpus_reducer(cfg, Path(cfg.preprocess.cache_dir))
    paths = list_corpus(cfg.dataset.path)[: args.limit]
    print(f"device={device}  K={K}  {len(paths)} images  warm-up={WARMUP} excluded\n")

    # One model, reused: inference cost is what is being measured here, not
    # training, which is timed separately below.
    set_seeds(cfg.seed, cfg.deterministic)
    patches = cached_patches(cfg, reducer,
                             list_corpus(cfg.autoencoder.training_corpus),
                             Path(cfg.preprocess.cache_dir), _Args())
    gpu_peak_gb(reset=True)
    t0 = time.perf_counter()
    model, hist = train_autoencoder(patches, cfg.autoencoder, reducer=reducer,
                                    device=device, seed=cfg.seed)
    corpus_train = {"seconds": time.perf_counter() - t0,
                    "n_patches": int(patches.shape[0]),
                    "epochs": cfg.autoencoder.epochs,
                    "peak_vram_gb": gpu_peak_gb(),
                    "reported_by_history_s": hist.seconds}
    print(f"AE training, corpus ({patches.shape[0]:,} patches, "
          f"{cfg.autoencoder.epochs} epochs): {corpus_train['seconds']:.1f}s, "
          f"peak {corpus_train['peak_vram_gb']:.2f} GB")

    pre_t, rx_t, ae_t, fu_t, shapes = [], [], [], [], []
    gpu_peak_gb(reset=True)
    rss0 = rss_gb()
    for path in paths:
        cube = load_cube(cfg.dataset, path)

        t = time.perf_counter(); clipped = mask_and_clip(cube, cfg.preprocess)
        reduced = reducer.transform(clipped); pre_t.append(time.perf_counter() - t)

        t = time.perf_counter()
        rx_map = rx_score(reduced, ridge=cfg.rx.ridge, shrinkage=cfg.rx.shrinkage)
        rx_t.append(time.perf_counter() - t)

        t = time.perf_counter()
        ae_map = reconstruction_error_map(model, reduced,
                                          patch_size=cfg.autoencoder.patch_size,
                                          batch_size=AE_BATCH, device=device)
        sync(device)
        ae_t.append(time.perf_counter() - t)

        t = time.perf_counter()
        fuse_scores(rx_map, ae_map, mode=cfg.score_fusion.mode,
                    alpha=cfg.score_fusion.alpha)
        fu_t.append(time.perf_counter() - t)
        shapes.append(list(cube.shape))

    inference_peak = gpu_peak_gb()
    rows = [
        summarise("preprocess + PCA transform", pre_t),
        summarise("RX detector (Eq. 2)", rx_t),
        summarise("autoencoder inference (Eq. 1)", ae_t),
        summarise("fusion", fu_t),
    ]
    total = np.median(np.array(pre_t[WARMUP:]) + np.array(rx_t[WARMUP:])
                      + np.array(ae_t[WARMUP:]) + np.array(fu_t[WARMUP:]))
    rows.append({"stage": "TOTAL per image (pipeline)", "median_s": float(total),
                 "n_timed": len(pre_t) - WARMUP, "warmup_excluded": WARMUP})

    print(f"\nper-image cost on {shapes[0]} images, median over "
          f"{len(paths) - WARMUP}:")
    for r in rows:
        print(f"  {r['stage']:<32} {r['median_s'] * 1000:8.1f} ms")
    print(f"\n  inference peak VRAM {inference_peak:.3f} GB   "
          f"host RSS {rss_gb():.2f} GB (delta {rss_gb() - rss0:+.2f})")

    # --- Is the autoencoder's cost an artefact of how it was called? -------
    #
    # R2 is entitled to a cost ratio measured against a reasonable
    # implementation, not against an unlucky default. Three things are checked:
    # the batch size, where the time actually goes, and which device each
    # branch ran on.
    import torch

    from pipeline.hyperspectral.autoencoder import extract_patches

    probe = reducer.transform(mask_and_clip(load_cube(cfg.dataset, paths[0]),
                                            cfg.preprocess))
    sweep = {}
    for bs in AE_BATCH_SWEEP:
        ts = []
        for _ in range(WARMUP + 5):
            t = time.perf_counter()
            reconstruction_error_map(model, probe,
                                     patch_size=cfg.autoencoder.patch_size,
                                     batch_size=bs, device=device)
            sync(device)
            ts.append(time.perf_counter() - t)
        sweep[bs] = float(np.median(ts[WARMUP:]))
    best_bs = min(sweep, key=sweep.get)

    # Split the winning configuration into its host and device halves.
    ex_t, fw_t = [], []
    for _ in range(WARMUP + 5):
        t = time.perf_counter()
        patches = extract_patches(probe, cfg.autoencoder.patch_size)
        ex_t.append(time.perf_counter() - t)

        data = torch.as_tensor(patches, dtype=torch.float32)
        sync(device)
        t = time.perf_counter()
        model.eval()
        with torch.no_grad():
            for lo in range(0, data.shape[0], best_bs):
                model(data[lo:lo + best_bs].to(device))
        sync(device)
        fw_t.append(time.perf_counter() - t)

    extract_ms = float(np.median(ex_t[WARMUP:])) * 1000
    forward_ms = float(np.median(fw_t[WARMUP:])) * 1000
    n_patches = int(probe.shape[0] * probe.shape[1])

    print(f"\n  autoencoder inference, batch-size sweep "
          f"({n_patches} patches per image):")
    for bs, s in sweep.items():
        mark = "  <- best" if bs == best_bs else ""
        used = "  (used above)" if bs == AE_BATCH else ""
        print(f"    batch {bs:<5} {s * 1000:8.1f} ms{mark}{used}")
    print(f"\n  where the time goes at batch {best_bs}:")
    print(f"    patch extraction (host, numpy)   {extract_ms:8.1f} ms")
    print(f"    network forward ({device:<4})            {forward_ms:8.1f} ms")
    print(f"\n  device per branch:  RX = cpu (numpy/scipy),  "
          f"autoencoder = {device}")
    if device == "cuda":
        print("    RX is CPU-only by implementation, so the ratio below already "
              "favours the autoencoder.")

    ae_cost = {
        "batch_sweep_seconds": {str(k): v for k, v in sweep.items()},
        "best_batch": best_bs,
        "batch_used_in_pipeline": AE_BATCH,
        "n_patches_per_image": n_patches,
        "patch_extraction_host_ms": extract_ms,
        "network_forward_ms": forward_ms,
        "forward_device": device,
        "rx_device": "cpu",
        "note": ("RX is implemented in numpy and runs on the CPU; the "
                 "autoencoder runs on the GPU where one is present. The "
                 "comparison therefore favours the autoencoder, and the "
                 "measured gap is a lower bound on the gap."),
    }

    # --- Per-scene (Auto-AD style) training, as ABU requires --------------
    abu = load_config(args.abu_config, HyperspectralConfig)
    abu_paths = sorted(Path("data/hyperspectral/abu").glob("*.mat"))[:3]
    per_scene = []
    for ap in abu_paths:
        acfg = replace(abu, dataset=replace(abu.dataset, path=str(ap),
                                            ground_truth_path=str(ap)))
        cube = load_cube(acfg.dataset)
        clipped = mask_and_clip(cube, acfg.preprocess)
        red = BandReducer(n_components=acfg.preprocess.pca_n_components,
                          variance=acfg.preprocess.pca_variance).fit(clipped)
        reduced = red.transform(clipped)
        pts = extract_patches(reduced, acfg.autoencoder.patch_size)
        gpu_peak_gb(reset=True)
        set_seeds(0, True)
        t = time.perf_counter()
        train_autoencoder(pts, acfg.autoencoder, reducer=red, device=device, seed=0)
        per_scene.append({"scene": ap.stem, "n_patches": int(pts.shape[0]),
                          "seconds": time.perf_counter() - t,
                          "peak_vram_gb": gpu_peak_gb(),
                          "n_bands": int(cube.n_bands)})
        print(f"  AE training, per scene {ap.stem:<16} "
              f"{per_scene[-1]['n_patches']:>6,} patches  "
              f"{per_scene[-1]['seconds']:6.1f}s  "
              f"peak {per_scene[-1]['peak_vram_gb']:.2f} GB")

    import platform
    hw = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch
        hw["torch"] = torch.__version__
        if torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
            hw["gpu_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    except Exception:
        pass

    report = {
        "hardware": hw, "device": device,
        "image_shape": shapes[0], "n_images_timed": len(paths) - WARMUP,
        "warmup_excluded": WARMUP,
        "per_image_stages": rows,
        "inference_peak_vram_gb": inference_peak,
        "autoencoder_inference_fairness": ae_cost,
        "training": {
            "corpus": corpus_train,
            "per_scene_autoad_style": per_scene,
            "per_scene_median_s": float(np.median([x["seconds"] for x in per_scene])),
        },
        "note": ("Medians over images with the first 3 excluded as warm-up: the "
                 "first CUDA call in a process pays context creation and cuDNN "
                 "autotuning, which would otherwise land in a per-image median. "
                 "Corpus training happens once for all 100 images; per-scene "
                 "training happens once PER image, which is the cost ABU pays."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
