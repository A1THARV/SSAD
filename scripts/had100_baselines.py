"""Compute HAD100's 19 published baselines' mAUC with *our* metric code.

    python scripts/had100_baselines.py

Change C3 requires positioning this pipeline inside HAD100's published ranking
and explicitly forbids re-implementing the baselines. It does not require
trusting a transcribed table: the benchmark ships each method's raw per-pixel
result matrices (``band50/``, ``band100/``, ``band200/``, 100x64x64 each), so
the honest thing is to run them through the same
:func:`mean_per_image_auc` this project reports its own numbers with.

That removes the two ways a comparison like this usually goes wrong -- a typo in
transcription, and a silent metric mismatch (pooled AUC against per-image mAUC
reads as a ~0.01 difference that is really a definition difference).

Cite: Li et al., "Hyperspectral Anomaly Detection: A Survey and Benchmark",
IEEE TGRS 61, 2023, doi:10.1109/TGRS.2023.3258067.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import scipy.io as scio  # noqa: E402
from PIL import Image  # noqa: E402

from pipeline.errors import DataUnavailableError  # noqa: E402
from pipeline.evaluation.metrics import mean_per_image_auc  # noqa: E402

DEFAULT_ROOT = Path("data/hyperspectral/had100/HAD100_resluts_and_eval")
DEFAULT_GT = Path("data/hyperspectral/had100/HAD100/HAD100Dataset/ground_truth/aviris_ng")
BAND_SETTINGS = ("band50", "band100", "band200")


def load_ground_truth(gt_dir: Path) -> list[np.ndarray]:
    """Read the benchmark's binary PNG masks, in the order its own scripts use."""
    paths = sorted(gt_dir.glob("*.png"))
    if not paths:
        raise DataUnavailableError(
            reason="No ground-truth PNGs found",
            path=gt_dir,
            remedy="Run HAD100's main.py: cd data/hyperspectral/had100/HAD100 && python main.py",
        )
    masks = []
    for path in paths:
        arr = np.array(Image.open(path))
        # Their eval_results.py takes channel 1 of the RGB PNG and maps 255 -> 1.
        channel = arr[:, :, 1] if arr.ndim == 3 else arr
        masks.append(channel > 0)
    return masks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--out", type=Path, default=Path("results/had100_baselines.json"))
    args = p.parse_args(argv)

    truths = load_ground_truth(args.gt)
    print(f"ground truth: {len(truths)} masks, "
          f"{sum(int(m.any()) for m in truths)} with anomalies present\n")

    table: dict[str, dict[str, float]] = {}
    for setting in BAND_SETTINGS:
        folder = args.root / setting
        if not folder.is_dir():
            print(f"  {setting}: not present, skipped")
            continue
        for mat in sorted(folder.glob("*.mat")):
            method = mat.stem
            result = scio.loadmat(mat)["result"]
            # PTA emits NaN on some scenes; their eval script zeroes them, and
            # matching that is what makes this comparable to their table.
            n_nan = int(np.count_nonzero(~np.isfinite(result)))
            if n_nan:
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            maps = [result[i] for i in range(result.shape[0])]
            mauc, n_used = mean_per_image_auc(maps, truths)
            table.setdefault(method, {})[setting] = mauc
            if n_nan:
                table[method][f"{setting}_nan_pixels"] = n_nan
            table[method]["n_images"] = n_used

    # Report ordered by the band setting with the most entries.
    best_setting = max(
        BAND_SETTINGS,
        key=lambda s: sum(1 for v in table.values() if s in v),
    )
    ordered = sorted(
        table.items(), key=lambda kv: kv[1].get(best_setting, -1), reverse=True
    )

    header = f"| Rank | Method | {' | '.join(BAND_SETTINGS)} |"
    print(header)
    print("|---|---|" + "---|" * len(BAND_SETTINGS))
    for rank, (method, row) in enumerate(ordered, 1):
        cells = " | ".join(
            f"{row[s]:.4f}" if s in row else "--" for s in BAND_SETTINGS
        )
        print(f"| {rank} | {method} | {cells} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"ranked_by": best_setting, "table": table, "n_masks": len(truths)},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
