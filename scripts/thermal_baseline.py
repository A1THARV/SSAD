"""What does image-level fire detection on FLAME 3 cost, before any model?

    python scripts/thermal_baseline.py

Run *before* building the M4 CNN, not after, and that ordering is the point. A
model's number means nothing without the trivial baseline beside it, and the
cheapest way to guarantee the baseline is honest is to measure it while there is
still no model to be disappointed by.

The answer here is that the task is saturated: maximum pixel temperature gives
AUC 1.0000 over all 738 frames, with a 61 C margin between the coldest fire and
the hottest no-fire frame. No subset of the data is harder. See RESULTS.md §11.

No train/test split is involved because nothing is fitted -- a threshold on
absolute temperature has no parameters learned from these labels. The reported
F1 uses the F1-maximising threshold, which *is* chosen on this data and is
labelled as such; the AUC, which needs no threshold, is the honest headline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from pipeline.config import DatasetConfig  # noqa: E402
from pipeline.evaluation.metrics import compute_detection_metrics  # noqa: E402
from pipeline.evaluation.thresholding import f1_optimal_threshold  # noqa: E402
from pipeline.thermal.loader import load_dataset  # noqa: E402

#: Single-frame summaries, each one a candidate "detector" with no parameters.
FEATURES = {
    "max_T": lambda a: float(a.max()),
    "p99.9_T": lambda a: float(np.percentile(a, 99.9)),
    "p99_T": lambda a: float(np.percentile(a, 99.0)),
    "frac_over_100C": lambda a: float((a > 100.0).mean()),
    "mean_T": lambda a: float(a.mean()),
    "std_T": lambda a: float(a.std()),
}

#: Fractions of the fire class, kept from the smallest fires by burning area, to
#: test whether a harder subset exists inside the dataset.
HARD_SUBSET_QUANTILES = (100, 50, 25, 10, 5)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--path", default="data/thermal/flame3_cv_subset")
    p.add_argument("--out", default="results/thermal/flame3_baseline.json")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    frames = load_dataset(
        DatasetConfig(name="flame3", kind="thermal", path=args.path), limit=args.limit
    )
    y = np.array([f.require_label() for f in frames], dtype=bool)
    print(
        f"{len(frames)} frames ({y.sum()} fire / {(~y).sum()} no-fire) "
        f"loaded in {time.perf_counter() - t0:.0f}s\n"
    )

    values = {name: np.array([fn(f.data) for f in frames]) for name, fn in FEATURES.items()}

    print(f"{'single feature':<16} {'AUC':>8} {'best F1':>9}")
    table = {}
    for name, v in values.items():
        auc = float(roc_auc_score(y, v))
        metrics = compute_detection_metrics(v, y, threshold=f1_optimal_threshold(v, y))
        table[name] = {"auc": auc, "f1_at_optimal_threshold": metrics.f1}
        print(f"  {name:<14} {auc:>8.4f} {metrics.f1:>9.4f}")

    mx = values["max_T"]
    margin = {
        "coldest_fire_peak_c": float(mx[y].min()),
        "hottest_no_fire_peak_c": float(mx[~y].max()),
    }
    margin["separation_c"] = margin["coldest_fire_peak_c"] - margin["hottest_no_fire_peak_c"]
    print(
        f"\nmargin: coldest fire peaks at {margin['coldest_fire_peak_c']:.2f} C, "
        f"hottest no-fire at {margin['hottest_no_fire_peak_c']:.2f} C "
        f"-> {margin['separation_c']:+.2f} C gap"
    )

    # Is there a harder problem inside this one? Shrink the fire class to the
    # smallest fires by burning area and re-test; every no-fire frame is kept,
    # so the negative class never gets easier.
    print("\nhard-subset probe (smallest fires by burning area, all no-fire kept):")
    area = values["frac_over_100C"]
    subsets = {}
    for q in HARD_SUBSET_QUANTILES:
        cutoff = np.percentile(area[y], q)
        keep = (~y) | (y & (area <= cutoff))
        auc = float(roc_auc_score(y[keep], mx[keep]))
        subsets[f"smallest_{q}pct"] = {
            "n_fire": int((y & keep).sum()),
            "auc_max_T": auc,
            "coldest_fire_peak_c": float(mx[y & keep].min()),
        }
        label = "all fires" if q == 100 else f"smallest {q}% by area"
        print(
            f"  {label:<24} n_fire={subsets[f'smallest_{q}pct']['n_fire']:>4}  "
            f"AUC={auc:.4f}  coldest peak="
            f"{subsets[f'smallest_{q}pct']['coldest_fire_peak_c']:.2f}C"
        )

    payload = {
        "n_frames": len(frames),
        "n_fire": int(y.sum()),
        "n_no_fire": int((~y).sum()),
        "features": table,
        "margin": margin,
        "hard_subsets": subsets,
        "note": (
            "Image-level labels; FLAME 3's CV subset ships no pixel masks. AUC needs "
            "no threshold and nothing is fitted, so no split is required. The F1 "
            "column uses the F1-maximising threshold chosen on this same data and is "
            "reported as an upper bound, not a held-out score."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
