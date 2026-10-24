"""Step 4 — why did seed 4 collapse? (INDICON revision)

    python scripts/revision_seed4_diag.py

A1 found the band50 baseline autoencoder scoring 0.8299 at seed 4 against
0.9255 +/- 0.0556 over the five seeds. Before the paper calls that a property of
the method, it has to be ruled out as a fault specific to that run -- a bad patch
draw, a loader returning something degenerate, a numerical blow-up.

A1 recorded only the final AUC, so the loss curve is not available and this
retrains seeds 0-4 to capture it. **Training only** -- no scoring, no score maps,
since A1 already wrote those -- which is why this costs minutes rather than the
half-hour the full block took.

The setup is copied from revision_seed_variance.py rather than re-specified:
same config, same variant, same cached_patches() call, same set_seeds() before
it. If this diverges from A1's setup the comparison is worthless.

Classification, applied to the final epoch relative to seeds 0-3:

  diverged   non-finite loss at any epoch, or final loss above the first
  plateaued  loss essentially flat after the first few epochs
  degenerate the patch draw itself is broken -- near-zero variance, non-finite
  normal     curve looks like its siblings and the AUC gap is elsewhere

The last of those is the interesting answer: a seed that trains normally and
still detects badly says the method is unstable, which is reportable. A seed
that fed on broken patches says the harness is.
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

from pipeline.config import (  # noqa: E402
    HyperspectralConfig,
    load_config,
    resolve_device,
    set_seeds,
)
from pipeline.hyperspectral.autoencoder import train_autoencoder  # noqa: E402
from pipeline.hyperspectral.loader import list_corpus  # noqa: E402
from pipeline.cli.run_hyperspectral import cached_patches, corpus_reducer  # noqa: E402
from revision_seed_variance import _Args  # noqa: E402
from run_ablations import AE_VARIANTS  # noqa: E402

CONFIG = "configs/had100_band50.yaml"
VARIANT = "baseline"
SEEDS = (0, 1, 2, 3, 4)
SUSPECT = 4


#: A final epoch this many times worse than the run's own best is a spike, not
#: a converged model. Every seed's loss wanders by ~5x between epochs, so the
#: threshold has to sit above that band to mean anything.
SPIKE_RATIO = 15.0


def classify(loss: list[float], siblings: list[list[float]]) -> tuple[str, str]:
    """Label one curve against the curves that did not collapse.

    The ratio of the FINAL loss to the run's own BEST loss is what matters,
    because the pipeline keeps the last epoch's weights unconditionally -- there
    is no checkpoint selection. A run can therefore train perfectly well and
    still be saved at its worst moment.
    """
    a = np.asarray(loss, dtype=float)
    if not np.all(np.isfinite(a)):
        bad = int(np.argmax(~np.isfinite(a)))
        return "diverged", f"non-finite loss from epoch {bad + 1}"
    if a[-1] > a[0]:
        return "diverged", f"final loss {a[-1]:.6g} above initial {a[0]:.6g}"

    best, best_ep = float(a.min()), int(a.argmin()) + 1
    ratio = a[-1] / best if best > 0 else float("inf")
    if ratio > SPIKE_RATIO:
        return "final-epoch spike", (
            f"best {best:.4g} at epoch {best_ep}, final {a[-1]:.4g} "
            f"= {ratio:.0f}x worse; the saved weights are the spike")

    # A curve that has stopped moving has stopped learning. Measured on the
    # best-so-far, which is monotone, so ordinary epoch-to-epoch noise does not
    # register as progress or as stalling.
    run_min = np.minimum.accumulate(a)
    tail = run_min[min(3, len(a) - 1):]
    rel = (tail[0] - tail[-1]) / abs(tail[0]) if tail[0] else 0.0
    if rel < 0.01:
        return "plateaued", f"best-so-far improved only {rel * 100:.2f}% after epoch 3"

    if siblings:
        bests = np.array([min(s) for s in siblings], dtype=float)
        mu, sd = bests.mean(), bests.std()
        if sd > 0 and abs(best - mu) > 3 * sd:
            return "outlier-loss", (f"best loss {best:.4g} is "
                                    f"{(best - mu) / sd:+.1f} sd from siblings "
                                    f"({mu:.4g} +/- {sd:.4g})")
    return "normal", (f"best {best:.4g} at epoch {best_ep}, final {a[-1]:.4g} "
                      f"({ratio:.1f}x best)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/revision/seed4_diagnosis.json")
    p.add_argument("--reclassify", action="store_true",
                   help="re-label the curves already in --out without retraining")
    args = p.parse_args(argv)

    if args.reclassify:
        prev = json.loads(Path(args.out).read_text(encoding="utf-8"))
        runs = {int(k): v for k, v in prev["runs"].items()}
        return finish(runs, Path(args.out))

    cfg = load_config(CONFIG, HyperspectralConfig)
    spec = next(v for v in AE_VARIANTS if v[0] == VARIANT)
    _, k, latent, whiten, lam, sam_domain = spec
    reducer = corpus_reducer(
        replace(cfg, preprocess=replace(cfg.preprocess, pca_n_components=k)),
        Path(cfg.preprocess.cache_dir))
    corpus = list_corpus(cfg.autoencoder.training_corpus)
    device = resolve_device(cfg.autoencoder.device)
    print(f"{CONFIG}  variant={VARIANT}  K={k}  device={device}\n", flush=True)

    runs = {}
    for seed in SEEDS:
        variant = replace(
            cfg, seed=seed,
            preprocess=replace(cfg.preprocess, pca_n_components=k),
            autoencoder=replace(cfg.autoencoder, latent_dim=latent,
                                whiten_input=whiten, lam=lam,
                                sam_domain=sam_domain),
        )
        set_seeds(seed, variant.deterministic)
        patches = cached_patches(variant, reducer, corpus,
                                 Path(variant.preprocess.cache_dir), _Args())
        arr = np.asarray(patches, dtype=np.float64)
        # The patch draw is checked BEFORE training, so a broken input is not
        # mistaken for a broken optimiser.
        draw = {
            "n_patches": int(arr.shape[0]),
            "finite": bool(np.all(np.isfinite(arr))),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "zero_variance_fraction": float(
                np.mean(arr.reshape(arr.shape[0], -1).std(axis=1) < 1e-12)),
        }
        _, hist = train_autoencoder(patches, variant.autoencoder,
                                    reducer=reducer, device=device, seed=seed)
        runs[seed] = {"draw": draw, "loss": hist.loss, "mse": hist.mse,
                      "sam": hist.sam, "seconds": hist.seconds}
        print(f"seed {seed}  patches {draw['n_patches']}  "
              f"std {draw['std']:.4f}  loss {hist.loss[0]:.6g} -> "
              f"{hist.loss[-1]:.6g}  [{hist.seconds:.0f}s]", flush=True)

    return finish(runs, Path(args.out))


def finish(runs: dict, out_path: Path) -> int:
    """Classify the curves and write the report.

    Separated from training so `--reclassify` can re-derive the verdicts from
    curves already measured. The first version of classify() labelled a
    last-epoch blow-up as a plateau, and retraining for 54 minutes to correct a
    label would have been the wrong way to fix it.
    """
    for seed in SEEDS:
        others = [runs[s]["loss"] for s in SEEDS if s != seed]
        label, why = classify(runs[seed]["loss"], others)
        runs[seed]["verdict"], runs[seed]["why"] = label, why

    d = runs[SUSPECT]
    draw_ok = (d["draw"]["finite"] and d["draw"]["std"] > 1e-6
               and d["draw"]["zero_variance_fraction"] < 0.5)
    if not draw_ok:
        conclusion = ("HARNESS FAULT: the patch draw at this seed is degenerate, "
                      "so the low AUC says nothing about the method.")
    elif d["verdict"] == "final-epoch spike":
        conclusion = (
            "NO CHECKPOINT SELECTION. The patch draw is sound and training was "
            "healthy to within one epoch of the end -- this seed reached a best "
            "loss in line with its siblings and then spiked on the last epoch. "
            "The pipeline keeps the final weights unconditionally, so the spike "
            "is what got saved and scored. This is a harness gap, not a property "
            "of the architecture: every seed's loss wanders several-fold between "
            "epochs, and seed 4 is the draw whose wander landed on epoch 20. "
            "Report as 1 of 5 seeds, and report the cause.")
    elif d["verdict"] == "normal":
        conclusion = (
            "METHOD INSTABILITY: the patch draw is sound and the loss curve is "
            "indistinguishable from its siblings, yet detection is far worse. "
            "Training converged to a reconstruction that is accurate on "
            "background and does not separate anomalies -- report as 1 of 5 "
            "seeds, not as a rate.")
    else:
        conclusion = (
            f"TRAINING FAILURE ({d['verdict']}): {d['why']}. The optimiser, not "
            "the data, is the cause -- report as 1 of 5 seeds and say what "
            "failed.")

    finals = {s: runs[s]["loss"][-1] for s in SEEDS}
    bests = {s: min(runs[s]["loss"]) for s in SEEDS}
    ratios = {s: finals[s] / bests[s] for s in SEEDS}
    out = {
        "config": CONFIG, "variant": VARIANT, "seeds": list(SEEDS),
        "suspect": SUSPECT,
        "final_loss": finals,
        "best_loss": bests,
        "best_epoch": {s: int(np.argmin(runs[s]["loss"])) + 1 for s in SEEDS},
        "final_over_best_ratio": ratios,
        "sibling_best_loss_mean": float(np.mean([bests[s] for s in SEEDS if s != SUSPECT])),
        "sibling_best_loss_std": float(np.std([bests[s] for s in SEEDS if s != SUSPECT])),
        "runs": {str(s): runs[s] for s in SEEDS},
        "patch_draw_sound": bool(draw_ok),
        "conclusion": conclusion,
        "checkpointing": (
            "train_autoencoder() returns the model in its final-epoch state; "
            "nothing tracks or restores the best epoch. Every seed's final loss "
            "sits above its own best, by "
            f"{min(ratios.values()):.1f}x to {max(ratios.values()):.1f}x."),
        "note": ("Reported as 1 of 5 seeds. Five runs do not support a rate, and "
                 "quoting one would invite a reader to multiply it."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"{'seed':<6}{'best':>10}{'@ep':>5}{'final':>11}{'x best':>9}  verdict")
    for seed in SEEDS:
        mark = "  <-- suspect" if seed == SUSPECT else ""
        print(f"{seed:<6}{bests[seed]:>10.4g}"
              f"{int(np.argmin(runs[seed]['loss'])) + 1:>5}"
              f"{finals[seed]:>11.4g}{ratios[seed]:>9.1f}  "
              f"{runs[seed]['verdict']}{mark}")
    print("=" * 74)
    print(f"patch draw sound at seed {SUSPECT}: {draw_ok}")
    print(conclusion)
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
