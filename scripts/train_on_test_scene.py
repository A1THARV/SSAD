"""Task 4.6: is change C2's premise actually true?

    python scripts/train_on_test_scene.py

Change C2 replaced PRD v1's self-supervised single-scene training with training
on a separate anomaly-free corpus, on the argument that Eq. 1's premise --
"anomalies reconstruct poorly" -- cannot hold for a background model fitted on
data containing the anomalies it must detect.

That argument is not self-evidently right. Training on the scene under test is a
published, competitive approach (Auto-AD, paper reference [8], which scores
0.9688 on HAD100's band50 column), and it has a real advantage: the model sees
this scene's exact background rather than a corpus average of 1,040 others. So
the comparison is a result, not a formality.

Everything except the training data is held fixed -- same architecture, same
epochs, same lambda, same corpus-fitted PCA basis. A per-scene PCA would confound
the training-data question with the basis question, which was settled separately.

**The confound that has to be controlled.** One 64x64 scene yields 4,096 patches;
the corpus run trains on 40,000. A raw comparison would credit the corpus with an
advantage that is really just ten times the data. So a matched control is also
run: the corpus model retrained on 4,096 patches, the same count a single scene
provides.
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

from pipeline.cli.run_hyperspectral import cached_patches, corpus_reducer  # noqa: E402
from pipeline.config import (  # noqa: E402
    HyperspectralConfig,
    load_config,
    resolve_device,
    set_seeds,
)
from pipeline.evaluation.metrics import mean_per_image_auc  # noqa: E402
from pipeline.hyperspectral.autoencoder import (  # noqa: E402
    extract_patches,
    reconstruction_error_map,
    train_autoencoder,
)
from pipeline.hyperspectral.loader import list_corpus, load_cube  # noqa: E402
from pipeline.hyperspectral.preprocess import mask_and_clip  # noqa: E402


class _Args:
    refit_pca = False


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/had100.yaml")
    p.add_argument("--out", default="results/m2/train_on_test_scene.json")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config, HyperspectralConfig)
    ae = cfg.autoencoder
    if args.epochs:
        ae = replace(ae, epochs=args.epochs)
    device = resolve_device(ae.device)
    reducer = corpus_reducer(cfg, Path(cfg.preprocess.cache_dir))

    paths = list_corpus(cfg.dataset.path)
    if args.limit:
        paths = paths[: args.limit]

    print(f"\nscenes: {len(paths)}   device: {device}   epochs: {ae.epochs}")

    # -- Pass 1: a fresh model per scene, trained on that scene's own patches.
    print("\n[A] train_on_test_scene = true  (Auto-AD-style, one model per scene)")
    maps: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    patch_counts: list[int] = []
    t0 = time.perf_counter()
    for i, path in enumerate(paths, 1):
        cube = load_cube(cfg.dataset, path)
        clipped = mask_and_clip(cube, cfg.preprocess)
        reduced = reducer.transform(clipped, whiten=ae.whiten_input)
        patches = extract_patches(reduced, ae.patch_size)
        patch_counts.append(patches.shape[0])

        # A fresh model each time. Carrying weights between scenes would make
        # this a corpus model with extra steps.
        set_seeds(cfg.seed, cfg.deterministic)
        model, _ = train_autoencoder(
            patches, ae, reducer=reducer, device=device, seed=cfg.seed
        )
        maps.append(
            reconstruction_error_map(
                model, reduced, patch_size=ae.patch_size, device=device
            )
        )
        truths.append(cube.ground_truth)
        if i % 10 == 0:
            rate = (time.perf_counter() - t0) / i
            print(
                f"    {i}/{len(paths)}  {rate:.1f}s/scene  "
                f"eta {rate * (len(paths) - i) / 60:.1f} min",
                flush=True,
            )

    per_scene_mauc, n = mean_per_image_auc(maps, truths)
    per_scene_seconds = time.perf_counter() - t0
    print(
        f"  mAUC {per_scene_mauc:.4f}  (n={n})  "
        f"{per_scene_seconds / 60:.1f} min total, "
        f"{int(np.mean(patch_counts)):,} patches/scene"
    )

    # -- Pass 2: matched control -- corpus training, same patch budget.
    matched = int(np.mean(patch_counts))
    print(f"\n[B] corpus training, patch budget matched to one scene ({matched:,})")
    control_cfg = replace(
        cfg, autoencoder=replace(ae, max_patches=matched)
    )
    set_seeds(cfg.seed, cfg.deterministic)
    patches = cached_patches(
        control_cfg,
        reducer,
        list_corpus(ae.training_corpus),
        Path(cfg.preprocess.cache_dir),
        _Args(),
    )
    model, history = train_autoencoder(
        patches, control_cfg.autoencoder, reducer=reducer, device=device, seed=cfg.seed
    )
    print(f"  trained {history.seconds:.0f}s on {patches.shape[0]:,} patches")

    control_maps = []
    for path in paths:
        cube = load_cube(cfg.dataset, path)
        clipped = mask_and_clip(cube, cfg.preprocess)
        control_maps.append(
            reconstruction_error_map(
                model,
                reducer.transform(clipped, whiten=ae.whiten_input),
                patch_size=ae.patch_size,
                device=device,
            )
        )
    control_mauc, _ = mean_per_image_auc(control_maps, truths)
    print(f"  mAUC {control_mauc:.4f}")

    payload = {
        "train_on_test_scene": {
            "mauc": per_scene_mauc,
            "n_images": n,
            "patches_per_scene": matched,
            "total_minutes": per_scene_seconds / 60,
            "models_trained": len(paths),
        },
        "corpus_matched_budget": {
            "mauc": control_mauc,
            "patches": int(patches.shape[0]),
            "train_seconds": history.seconds,
            "models_trained": 1,
        },
        "epochs": ae.epochs,
        "note": (
            "Both use the same corpus-fitted PCA basis, architecture, epochs and "
            "lambda; the only difference is which pixels the network saw. The "
            "corpus arm is capped at one scene's patch count so the comparison is "
            "not a data-volume comparison in disguise."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    delta = control_mauc - per_scene_mauc
    verdict = (
        "corpus training wins" if delta > 0 else "training on the test scene wins"
    )
    print(f"\n  delta {delta:+.4f} -> {verdict} (change C2's premise)")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
