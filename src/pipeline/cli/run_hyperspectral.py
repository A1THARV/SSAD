"""End-to-end hyperspectral run: RX, autoencoder, and their fusion (Fig. 1).

    python -m pipeline.cli.run_hyperspectral --config configs/had100.yaml

Reports all three branches from one invocation, because the number that matters
is the *delta* the autoencoder buys over RX, and computing the two in separate
runs invites them to drift onto different preprocessing.

Two PCA bases are fitted, deliberately:

* **RX** uses a per-scene basis. Eq. 2's ``Sigma^-1`` is estimated from the
  scene anyway, so a scene-local basis is consistent with it.
* **The autoencoder** uses one basis fitted across the whole background corpus.
  It has to: train and inference must present the encoder with the same axes,
  and a per-scene basis would hand it a different input space at test time.

Fig. 1 draws a single PCA block feeding both. That is the figure's simplification
and is on the paper edit list; ``rx.pca_basis: corpus`` runs the figure's version
so the cost of the difference is measured rather than argued about.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..config import (
    HyperspectralConfig,
    config_hash,
    load_config,
    resolve_device,
    set_seeds,
)
from ..errors import DegenerateInputError, PipelineError
from ..evaluation.metrics import compute_detection_metrics, mean_per_image_auc
from ..evaluation.report import format_auc, metrics_table
from ..evaluation.thresholding import compute_threshold
from ..hyperspectral.autoencoder import (
    reconstruction_error_map,
    sample_corpus_patches,
    train_autoencoder,
)
from ..hyperspectral.band_presets import good_band_indices
from ..hyperspectral.loader import list_corpus, load_cube
from ..hyperspectral.preprocess import (
    BandReducer,
    fit_corpus_reducer,
    mask_and_clip,
)
from ..hyperspectral.rx_detector import rx_score
from ..hyperspectral.score_fusion import fuse_scores
from ..results_io import RunRecord, current_command, new_run_id, write_run

BRANCHES = ("rx", "autoencoder", "fused")


# --------------------------------------------------------------------------
# Corpus basis, cached
# --------------------------------------------------------------------------


def corpus_reducer(cfg: HyperspectralConfig, cache_dir: Path, refit: bool = False) -> BandReducer:
    """Fit (or reload) the background PCA basis shared by training and inference.

    Cached because the exact streaming fit over 1,040 crops takes ~85 s, and
    every ablation in this file reuses the same basis. The cache key includes
    the corpus path, the component count *and the band preset*, so changing any
    of them refits rather than silently reusing a mismatched basis.

    The band preset has to reach the corpus fit as well as the scenes. Under
    ``had100_band50`` the test cubes arrive with 50 bands, so a basis fitted on
    the corpus's full 276 would be unusable -- ``transform`` refuses the width
    mismatch, which is the right failure but only if the preset is threaded
    through rather than forgotten.
    """
    ae = cfg.autoencoder
    pre = cfg.preprocess
    corpus = ae.training_corpus
    if not corpus:
        raise PipelineError(
            "autoencoder.training_corpus is unset, so there is no background corpus "
            "to fit the shared PCA basis on (PRD v2 change C2)."
        )

    paths = list_corpus(corpus)
    good_bands = None
    if pre.good_bands_preset != "none":
        n_bands = np.load(paths[0], mmap_mode="r").shape[-1]
        good_bands = good_band_indices(pre.good_bands_preset, n_bands)

    key = (
        f"corpuspca_k{pre.pca_n_components}_{pre.good_bands_preset}"
        f"_n{len(paths)}_{Path(corpus).name}.npz"
    )
    cached = cache_dir / key

    if cached.is_file() and not refit:
        blob = np.load(cached)
        reducer = BandReducer(n_components=pre.pca_n_components)
        reducer.mean_ = blob["mean"]
        reducer.components_ = blob["components"]
        reducer.explained_variance_ = blob["variance"]
        reducer.explained_variance_ratio_ = blob["ratio"]
        reducer.n_samples_seen_ = int(blob["n_samples"])
        print(f"  corpus PCA: reused {cached.name}")
        return reducer

    print(f"  corpus PCA: streaming fit over {len(paths)} crops ...", flush=True)
    start = time.perf_counter()
    reducer = fit_corpus_reducer(paths, pre, good_bands=good_bands)
    cached.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cached,
        mean=reducer.mean_,
        components=reducer.components_,
        variance=reducer.explained_variance_,
        ratio=reducer.explained_variance_ratio_,
        n_samples=reducer.n_samples_seen_,
    )
    print(
        f"  corpus PCA: {time.perf_counter() - start:.1f}s, "
        f"{reducer.n_samples_seen_:,} pixels, "
        f"{reducer.explained_variance():.4f} variance retained"
    )
    return reducer


def cached_patches(
    cfg: HyperspectralConfig,
    corpus_pca: BandReducer,
    corpus_paths: list[Path],
    cache_dir: Path,
    args,
) -> np.ndarray:
    """Sample the training patches once and reuse them across ablations.

    Sampling costs ~64 s whatever the cap, because it opens and projects all
    1,040 crops regardless of how few patches it keeps from each. The lambda,
    epoch and fusion sweeps all train on the *same* patches, so re-drawing them
    per run would spend an hour of I/O to produce identical arrays -- and, worse,
    would leave each ablation trained on a different sample, confounding the
    comparison the sweep exists to make.

    The key carries every input that changes the patches. Whitening is in it
    because it is applied at projection time, not in the network.
    """
    ae = cfg.autoencoder
    key = (
        f"patches_k{cfg.preprocess.pca_n_components}_p{ae.patch_size}"
        f"_n{ae.max_patches}_w{int(ae.whiten_input)}_s{cfg.seed}"
        f"_{cfg.preprocess.good_bands_preset}"
        f"_{len(corpus_paths)}.npy"
    )
    cached = cache_dir / key
    if cached.is_file() and not args.refit_pca:
        return np.load(cached)

    good_bands = None
    if cfg.preprocess.good_bands_preset != "none":
        n_bands = np.load(corpus_paths[0], mmap_mode="r").shape[-1]
        good_bands = good_band_indices(cfg.preprocess.good_bands_preset, n_bands)

    patches = sample_corpus_patches(
        corpus_paths,
        corpus_pca,
        cfg.preprocess,
        patch_size=ae.patch_size,
        max_patches=ae.max_patches,
        whiten=ae.whiten_input,
        good_bands=good_bands,
        seed=cfg.seed,
    )
    cached.parent.mkdir(parents=True, exist_ok=True)
    np.save(cached, patches)
    return patches


# --------------------------------------------------------------------------
# Per-scene scoring
# --------------------------------------------------------------------------


def score_scene(
    cube,
    cfg: HyperspectralConfig,
    model,
    corpus_pca: BandReducer,
    device: str,
) -> dict[str, np.ndarray]:
    """Produce the RX, autoencoder and fused score maps for one scene."""
    hs = cfg
    pre, rx_cfg, ae, fusion = hs.preprocess, hs.rx, hs.autoencoder, hs.score_fusion

    clipped = mask_and_clip(cube, pre)

    # -- RX branch, on its own basis unless the figure's version was asked for.
    if rx_cfg.pca_basis == "corpus":
        rx_reduced = corpus_pca.transform(clipped)
    else:
        rx_reduced = (
            BandReducer(n_components=pre.pca_n_components, variance=pre.pca_variance)
            .fit(clipped)
            .transform(clipped)
        )
    rx_map = rx_score(rx_reduced, ridge=rx_cfg.ridge, shrinkage=rx_cfg.shrinkage)

    out = {"rx": rx_map}
    if model is None:
        return out

    # -- Autoencoder branch, always on the shared corpus basis.
    ae_reduced = corpus_pca.transform(clipped, whiten=ae.whiten_input)
    recon_map = reconstruction_error_map(
        model, ae_reduced, patch_size=ae.patch_size, device=device
    )
    out["autoencoder"] = recon_map
    out["fused"] = fuse_scores(
        rx_map, recon_map, mode=fusion.mode, alpha=fusion.alpha
    )
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run(cfg: HyperspectralConfig, args) -> RunRecord:
    hs = cfg
    ae = hs.autoencoder
    set_seeds(cfg.seed, cfg.deterministic)
    cache_dir = Path(hs.preprocess.cache_dir)

    # A dataset is either a directory of scenes (HAD100) or one scene file
    # (Salinas, ABU). Both are legitimate; only the first can be listed.
    dataset_path = Path(hs.dataset.path)
    test_paths = [dataset_path] if dataset_path.is_file() else list_corpus(dataset_path)
    if args.limit:
        test_paths = test_paths[: args.limit]
    print(f"test set: {len(test_paths)} scene(s) from {hs.dataset.path}")

    model = None
    history = None
    device = "cpu"

    # Only fitted when something actually consumes it. Salinas and ABU have no
    # background corpus, and demanding one there would block the zero-anomaly
    # edge-case run (change C5) over a basis it never uses.
    wants_ae = ae.enabled and not args.rx_only
    corpus_pca = BandReducer()
    if wants_ae or hs.rx.pca_basis == "corpus":
        corpus_pca = corpus_reducer(cfg, cache_dir, refit=args.refit_pca)

    if wants_ae:
        device = resolve_device(ae.device)
        print(f"  device: {device}")
        corpus_paths = list_corpus(ae.training_corpus)
        print(
            f"  sampling <= {ae.max_patches:,} patches from {len(corpus_paths)} "
            "background crops ...",
            flush=True,
        )
        t0 = time.perf_counter()
        patches = cached_patches(cfg, corpus_pca, corpus_paths, cache_dir, args)
        print(
            f"  patches: {patches.shape[0]:,} x {patches.shape[2]} components "
            f"({patches.nbytes / 1024**2:.0f} MB, {time.perf_counter() - t0:.1f}s)"
        )
        print(f"  training {ae.epochs} epochs, lambda={ae.lam}, "
              f"sam_domain={ae.sam_domain} ...", flush=True)
        model, history = train_autoencoder(
            patches, ae, reducer=corpus_pca, device=device, seed=cfg.seed,
            progress=args.progress,
        )
        vram = "" if history.peak_vram_gb is None else f", peak VRAM {history.peak_vram_gb:.2f} GB"
        print(
            f"  trained in {history.seconds:.1f}s  "
            f"loss {history.loss[0]:.6f} -> {history.loss[-1]:.6f}{vram}"
        )

    # -- Score every test scene.
    maps: dict[str, list[np.ndarray]] = {b: [] for b in BRANCHES}
    truths: list[np.ndarray] = []
    t0 = time.perf_counter()
    for i, path in enumerate(test_paths, 1):
        cube = load_cube(hs.dataset, path)
        if cube.ground_truth is None:
            raise PipelineError(f"{cube.name}: no ground truth, so it cannot be scored.")
        scores = score_scene(cube, cfg, model, corpus_pca, device)
        for name, m in scores.items():
            maps[name].append(m)
        truths.append(cube.ground_truth)
        if args.progress and i % 20 == 0:
            print(f"    scored {i}/{len(test_paths)}", flush=True)
    print(f"  scored {len(test_paths)} scenes in {time.perf_counter() - t0:.1f}s")

    # -- Metrics. mAUC is HAD100's headline; the operating-point numbers come
    #    from pooling, and carry a note saying so.
    record = RunRecord(
        run_id=new_run_id(args.tag or hs.dataset.name),
        command=current_command(),
        config_hash=config_hash(cfg),
        labels={
            "pca_components": str(hs.preprocess.pca_n_components),
            "rx_pca_basis": hs.rx.pca_basis,
            "rx_shrinkage": hs.rx.shrinkage,
            "whiten_input": str(ae.whiten_input).lower(),
            "sam_domain": ae.sam_domain,
            "lambda": str(ae.lam),
            "fusion_alpha": str(hs.score_fusion.alpha),
            "trained_on": "corpus" if not ae.train_on_test_scene else "test_scene",
        },
    )

    print()
    for branch in BRANCHES:
        if not maps[branch]:
            continue
        # On a zero-anomaly scene (change C5's Salinas fixture) no image has
        # both classes, so mAUC is genuinely undefined. That is the expected
        # result there, not a failure: the number that run reports is the
        # false-positive rate, and `None` is what keeps a stand-in value out of
        # the record.
        try:
            mauc, n_used = mean_per_image_auc(maps[branch], truths)
        except DegenerateInputError:
            mauc, n_used = None, 0
        pooled = np.concatenate([m.ravel() for m in maps[branch]])
        pooled_gt = np.concatenate([g.ravel() for g in truths])
        threshold = compute_threshold(
            pooled,
            method=hs.score_fusion.threshold_method,
            percentile=hs.score_fusion.threshold_percentile,
            gamma=hs.score_fusion.threshold_gamma,
        )
        pooled_metrics = compute_detection_metrics(
            pooled, pooled_gt, threshold=threshold
        )
        # Replace the pooled AUC with HAD100's per-image mean -- the only figure
        # comparable to its published baseline table. `replace` rather than
        # assignment because DetectionMetrics is frozen, which is the right
        # default for a record that ends up in metrics.json.
        metrics = replace(
            pooled_metrics,
            auc_roc=mauc,
            notes=tuple(pooled_metrics.notes)
            + (
                (
                    f"auc_roc is HAD100 mAUC (per-image AUC averaged over {n_used} "
                    "images), not a pooled AUC; "
                    if mauc is not None
                    else "auc_roc is undefined: no image held both classes. "
                )
                + "precision/recall/F1 are pooled at a single "
                f"{hs.score_fusion.threshold_method} threshold.",
            ),
        )
        record.metrics[branch] = metrics
        print(f"  {branch:<12} mAUC {format_auc(mauc)}  (n={n_used})")

    if history is not None:
        record.notes.append(json.dumps({"training": history.as_dict()}))

    run_dir = write_run(cfg.output_root, record, cfg)
    print(f"\n{metrics_table(record.metrics)}")
    print(f"written to {run_dir}")
    return record


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="YAML config, e.g. configs/had100.yaml")
    p.add_argument("--tag", default=None, help="Run-id prefix; defaults to the dataset name.")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N scenes.")
    p.add_argument("--rx-only", action="store_true", help="Skip the autoencoder branch.")
    p.add_argument("--refit-pca", action="store_true", help="Ignore the cached corpus basis.")
    p.add_argument("--progress", action="store_true", help="Per-epoch and per-scene progress.")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override one config value, e.g. --set hyperspectral.autoencoder.lam=0.5. "
             "Overrides are applied before the config hash, so an ablation is "
             "distinguishable from the baseline in its run record.",
    )
    return p


def apply_overrides(cfg: HyperspectralConfig, assignments: list[str]) -> Config:
    """Apply ``--set a.b.c=value`` against the typed config.

    Values are coerced to the target field's existing type, so a typo in a
    numeric override fails here rather than surfacing as a string comparison
    deep inside training.
    """
    for item in assignments:
        if "=" not in item:
            raise PipelineError(f"--set expects PATH=VALUE, got {item!r}.")
        path, raw = item.split("=", 1)
        target = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise PipelineError(f"--set {path}: no config section {part!r}.")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise PipelineError(f"--set {path}: no config field {leaf!r}.")

        current = getattr(target, leaf)
        if isinstance(current, bool):
            value = raw.strip().lower() in ("1", "true", "yes")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        elif current is None or isinstance(current, str):
            value = None if raw == "null" else raw
        else:
            raise PipelineError(
                f"--set {path}: cannot override a {type(current).__name__} from the "
                "command line; edit the YAML instead."
            )
        setattr(target, leaf, value)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = apply_overrides(load_config(args.config, HyperspectralConfig), args.set)
        run(cfg, args)
    except PipelineError as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
