"""Export real scene data so Eq. 2 can be RE-COMPUTED live in a browser.

    python scripts/export_demo_data.py

A reviewer watching a slideshow has no way to tell a measured result from an
authored one. Static figures cannot answer "did you actually implement this, or
just draw the outcome?" -- the honest answer to that is to let them run it.

So this ships what the detector needs rather than what it produced:

  * the raw band50 cube, quantised to uint16 (64x64x50 per scene)
  * the CORPUS PCA basis -- components and mean, fitted over 1,040 anomaly-free
    crops, exactly the one RESULTS.md uses
  * the ground-truth mask
  * the autoencoder's score map, which CANNOT be recomputed in a browser and is
    therefore shipped and labelled as precomputed

Given those, a browser can carry out the whole RX path itself: project onto the
first K components, estimate mean and covariance, invert, and score every pixel
by squared Mahalanobis distance. Change K and the covariance changes, the
inverse changes, the map changes and the AUC changes -- which is the thing a
canned image cannot do, and the reason this file exists.

`reference` holds the AUCs computed HERE, in Python, by the same code that
produced the paper. The page recomputes from the cube and compares. If the two
disagree the page says so, because a demo that cannot fail proves nothing.
"""

from __future__ import annotations

import argparse
import base64
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
from pipeline.hyperspectral.rx_detector import rx_score  # noqa: E402
from pipeline.hyperspectral.score_fusion import fuse_scores  # noqa: E402
from pipeline.cli.run_hyperspectral import corpus_reducer  # noqa: E402
from demo_detection_panel import SCENES  # noqa: E402
from revision_stats import (  # noqa: E402
    crop_names,
    load_maps,
    load_truths,
    per_image_auc,
)
from run_ablations import prepared_scenes  # noqa: E402

K_MAX = 16          # the page lets K run from 2 to this
K_DEFAULT = 8       # what cross-validation selected
MAPS = Path("results/revision/maps")


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def quantise(a: np.ndarray) -> tuple[str, float, float]:
    """uint16 with a linear scale -- 2 bytes a band instead of 8.

    The quantisation error is ~1/65535 of the scene's range, four orders below
    anything the detector responds to, and the page's agreement check against
    the Python reference AUC is what confirms that rather than assuming it.
    """
    lo, hi = float(a.min()), float(a.max())
    q = np.round((a - lo) / (hi - lo) * 65535.0).astype(np.uint16)
    return b64(q), lo, (hi - lo) / 65535.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="figures/demo/scenes.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = load_config("configs/had100_band50.yaml", HyperspectralConfig)
    names = crop_names()
    truths = load_truths(names)
    prepared = {n: c for n, c, _ in prepared_scenes(cfg, None)}

    reducer = corpus_reducer(
        replace(cfg, preprocess=replace(cfg.preprocess, pca_n_components=K_MAX)),
        Path(cfg.preprocess.cache_dir))
    ae_maps = load_maps(MAPS / f"band50_ae_baseline_s{args.seed}.npz")
    rx8 = load_maps(MAPS / "band50_rx_k8_corpus.npz")
    rx8_auc = per_image_auc(rx8, truths)

    out = {
        "generated_by": "scripts/export_demo_data.py",
        "config": "configs/had100_band50.yaml",
        "k_max": K_MAX, "k_default": K_DEFAULT, "ae_seed": args.seed,
        "basis": {
            "components": b64(reducer.components_[:K_MAX].astype(np.float32)),
            "mean": b64(reducer.mean_.astype(np.float32)),
            "n_components": K_MAX, "n_bands": int(reducer.mean_.size),
            "fitted_on": "1,040 anomaly-free HAD100 training crops (corpus basis)",
        },
        "rx": {"ridge": cfg.rx.ridge, "shrinkage": cfg.rx.shrinkage},
        "scenes": [],
    }

    for scene, caption in SCENES:
        i = names.index(scene)
        cube = np.asarray(prepared[scene], dtype=np.float64)
        gt = truths[i]
        h, w, b = cube.shape

        # Reference AUCs at every K the page offers, from THIS code -- the page
        # recomputes them and flags any disagreement.
        ref = {}
        for k in range(2, K_MAX + 1):
            red = reducer.components_[:k] @ (cube.reshape(-1, b) - reducer.mean_).T
            m = rx_score(red.T.reshape(h, w, k).copy(),
                         ridge=cfg.rx.ridge, shrinkage=cfg.rx.shrinkage)
            ref[str(k)] = float(per_image_auc([m], [gt])[0])

        alpha_curve = {}
        for a10 in range(11):
            a = a10 / 10.0
            m = fuse_scores(rx8[i], ae_maps[i], alpha=a)
            alpha_curve[f"{a:.1f}"] = float(per_image_auc([m], [gt])[0])

        cube_b64, lo, scale = quantise(cube.astype(np.float32))
        out["scenes"].append({
            "name": scene, "caption": caption,
            "height": h, "width": w, "bands": b,
            "anomaly_fraction": float(gt.mean()),
            "cube": cube_b64, "cube_offset": lo, "cube_scale": scale,
            "gt": b64(np.packbits(gt.ravel().astype(np.uint8))),
            "ae": b64(ae_maps[i].astype(np.float32)),
            "reference": {
                "rx_auc_by_k": ref,
                "rx_k8_auc": float(rx8_auc[i]),
                "ae_auc": float(per_image_auc([ae_maps[i]], [gt])[0]),
                "fused_auc_by_alpha": alpha_curve,
            },
        })
        print(f"  {scene:<26} {caption:<14} RX(K=8) {ref['8']:.4f}  "
              f"AE {out['scenes'][-1]['reference']['ae_auc']:.4f}")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out), encoding="utf-8")
    print(f"\n  wrote {dest}  ({dest.stat().st_size / 1e6:.2f} MB, "
          f"{len(out['scenes'])} scenes, K = 2..{K_MAX})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
