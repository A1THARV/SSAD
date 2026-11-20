"""Presentation panel: what the detectors actually see (INDICON demo).

    python scripts/demo_detection_panel.py

RESULTS.md reports mAUCs. A number is the right thing for a paper and the wrong
thing for a room: nobody can picture 0.9890. This renders the score maps behind
it -- the real scene, the real ground truth, and what each branch produced --
so an audience can see the detection rather than take it on faith.

The three scenes are chosen to tell the honest version of the story, not the
flattering one:

  ang20171012t194435_29   fusion helps a lot   (RX 0.932 -> 0.984)
  ang20191004t215336_12   the hardest scene    (RX 0.834, everything struggles)
  ang20191004t185054_26   fusion HURTS         (AE 0.742 drags 0.993 -> 0.967)

The third panel is the point. A demo that only shows wins is a sales pitch; the
reviewers' question was whether the claims match the evidence, and the evidence
includes the scenes where the autoencoder makes things worse.

Reuses the score maps A1 already wrote, so this needs no GPU and no retraining.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pipeline.hyperspectral.score_fusion import fuse_scores  # noqa: E402
from revision_stats import (  # noqa: E402
    TEST_DIR,
    crop_names,
    load_maps,
    load_truths,
    per_image_auc,
)

MAPS = Path("results/revision/maps")
ALPHA = 0.9          # the cross-validated weight, not the a-priori 0.5

#: (scene, what it demonstrates)
SCENES = [
    ("ang20171012t194435_29", "fusion helps"),
    ("ang20191004t215336_12", "hardest scene"),
    ("ang20191004t185054_26", "fusion hurts"),
]

#: Visible-range band indices inside the 50-band band50 cube (455-700 nm at
#: ~5 nm), picked near 640 / 550 / 470 nm for an approximately true-colour
#: composite. Illustrative only -- no result depends on it.
RGB_BANDS = (37, 19, 3)


def stretch(a: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Percentile stretch to [0, 1]; AVIRIS radiance is not display-ranged."""
    p1, p2 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-12), 0, 1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="figures/demo_detection_panel")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.size": 9,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })

    names = crop_names()
    truths = load_truths(names)
    rx_maps = load_maps(MAPS / "band50_rx_k8_corpus.npz")
    ae_maps = load_maps(MAPS / f"band50_ae_baseline_s{args.seed}.npz")
    fu_maps = [fuse_scores(r, a, alpha=ALPHA) for r, a in zip(rx_maps, ae_maps)]
    rx_a = per_image_auc(rx_maps, truths)
    ae_a = per_image_auc(ae_maps, truths)
    fu_a = per_image_auc(fu_maps, truths)

    cols = ["scene (visible)", "ground truth", "RX (Eq. 2)",
            "autoencoder (Eq. 1)", f"fused, $\\alpha$ = {ALPHA}"]
    fig, axes = plt.subplots(len(SCENES), 5, figsize=(13.5, 2.95 * len(SCENES)))

    for row, (scene, caption) in enumerate(SCENES):
        i = names.index(scene)
        cube = np.load(TEST_DIR / f"{scene}.npy")
        gt = truths[i]

        rgb = np.dstack([stretch(cube[:, :, b]) for b in RGB_BANDS])
        panels = [
            (rgb, None, None),
            (gt.astype(float), "gray", None),
            (rx_maps[i], "inferno", rx_a[i]),
            (ae_maps[i], "inferno", ae_a[i]),
            (fu_maps[i], "inferno", fu_a[i]),
        ]
        for col, (img, cmap, auc) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(stretch(img, 1, 99) if cmap == "inferno" else img,
                      cmap=cmap, interpolation="nearest")
            # The anomaly outline is drawn over every score map, so a viewer can
            # check WHERE each detector fired instead of trusting the AUC.
            if col >= 2:
                ax.contour(gt, levels=[0.5], colors="#00E5FF", linewidths=0.9)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.6); s.set_color("#999999")
            if row == 0:
                ax.set_title(cols[col], fontsize=10, pad=7)
            if auc is not None:
                best = auc >= max(v for v in (rx_a[i], ae_a[i], fu_a[i])) - 1e-9
                ax.set_xlabel(f"AUC {auc:.4f}", fontsize=9.5,
                              color="#0B7A3B" if best else "#333333",
                              fontweight="bold" if best else "normal")
            if col == 0:
                ax.set_ylabel(f"{scene}\n{caption}  ·  {gt.mean() * 100:.2f}% anomalous",
                              fontsize=8.5, labelpad=6)

    fig.suptitle(
        "Spectral–spatial anomaly detection on HAD100 — cyan outline is ground truth",
        fontsize=12.5, y=1.003)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        fig.savefig(f"{out}.{ext}", **kw)
        print(f"  wrote {out}.{ext}")
    plt.close(fig)

    print(f"\n  alpha = {ALPHA} (cross-validated); autoencoder seed {args.seed}")
    for scene, caption in SCENES:
        i = names.index(scene)
        print(f"  {scene:<26} {caption:<14} rx {rx_a[i]:.4f}  "
              f"ae {ae_a[i]:.4f}  fused {fu_a[i]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
