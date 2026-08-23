"""Detection metrics: AUC-ROC, precision, recall, F1 (FR-9.11).

Built in Phase 0 rather than alongside the reporting work, because both the
hyperspectral and the thermal branch depend on it: without it, the milestone-1
baseline has no way to state a headline number.

Degenerate ground truth is handled explicitly rather than by exception. A scene
with no true anomalies (§11 edge case 3) must still produce a report, so
metrics that are genuinely undefined there are reported as ``None`` (AUC-ROC)
or as zero *with an accompanying note*, never as a plausible-looking number
that hides the fact nothing could be measured (PRD §10).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from ..errors import DegenerateInputError
from .thresholding import compute_threshold

__all__ = [
    "DetectionMetrics",
    "compute_detection_metrics",
    "roc_points",
    "mean_per_image_auc",
]


@dataclass(frozen=True)
class DetectionMetrics:
    """Detection quality at one operating point.

    Extends the PRD §9 contract (``auc_roc``, ``precision``, ``recall``, ``f1``,
    ``threshold_used``) with the counts needed to interpret those numbers and a
    ``notes`` field carrying any caveat that applies to this particular result.

    ``auc_roc`` is ``float | None`` rather than ``float``: it is undefined when
    the ground truth contains only one class, and ``None`` forces the caller to
    handle that rather than letting a stand-in value flow into a report.
    """

    auc_roc: float | None
    precision: float
    recall: float
    f1: float
    threshold_used: float
    false_positive_rate: float
    n_pixels: int
    n_anomaly_pixels: int
    n_flagged_pixels: int
    true_positives: int
    false_positives: int
    false_negatives: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping, as written into ``metrics.json``."""
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionMetrics":
        payload = dict(data)
        payload["notes"] = tuple(payload.get("notes", ()))
        return cls(**payload)


def _validate(scores: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    gt = np.asarray(ground_truth)

    if scores.shape != gt.shape:
        raise DegenerateInputError(
            f"Score map shape {scores.shape} does not match ground truth shape {gt.shape}. "
            "These must be the same grid; check for a transposed or cropped scene."
        )
    if scores.size == 0:
        raise DegenerateInputError("Cannot compute metrics on an empty score map.")
    if not np.all(np.isfinite(scores)):
        n_bad = int(np.count_nonzero(~np.isfinite(scores)))
        raise DegenerateInputError(
            f"Score map contains {n_bad} non-finite value(s). Metrics computed over "
            "NaN or infinite scores would be meaningless, so this is refused rather "
            "than silently masked (PRD §10)."
        )

    # A class map (e.g. Salinas' 17 land-cover ids) must be reduced to a binary
    # mask by the caller under a stated convention -- casting it to bool here
    # would silently treat every non-background class as an anomaly.
    if gt.dtype != bool:
        unique = np.unique(gt)
        if unique.size > 2 or not np.all(np.isin(unique, (0, 1))):
            raise DegenerateInputError(
                f"Ground truth holds values {unique[:8].tolist()}"
                f"{'...' if unique.size > 8 else ''}, so it is a class map rather than a "
                "binary anomaly mask. Reduce it under an explicit label convention first "
                "(see DatasetConfig.label_convention)."
            )
    return scores.ravel(), gt.astype(bool).ravel()


def compute_detection_metrics(
    scores: np.ndarray,
    ground_truth: np.ndarray,
    threshold_method: str = "percentile",
    threshold_percentile: float = 99.0,
    threshold: float | None = None,
) -> DetectionMetrics:
    """Score an anomaly map against ground truth.

    Args:
        scores: Continuous anomaly scores, any shape, higher = more anomalous.
        ground_truth: Binary mask of the same shape; truthy = anomaly.
        threshold_method: ``percentile`` or ``otsu``, used when ``threshold``
            is not given.
        threshold_percentile: Percentile for the ``percentile`` method.
        threshold: Explicit threshold, overriding the method. Used when
            comparing two detectors at an identical operating point.

    Returns:
        A :class:`DetectionMetrics`. AUC-ROC is ``None`` when the ground truth
        holds only one class; precision/recall/F1 are ``0.0`` with an
        explanatory note when their denominators are empty.

    Raises:
        DegenerateInputError: shapes disagree, the map is empty, the scores
            contain non-finite values, or the ground truth is a class map
            rather than a binary mask.
    """
    flat_scores, flat_gt = _validate(scores, ground_truth)
    notes: list[str] = []

    n_pixels = int(flat_gt.size)
    n_anomaly = int(np.count_nonzero(flat_gt))

    # --- AUC-ROC: threshold-independent, undefined for single-class truth ---
    if n_anomaly == 0:
        auc: float | None = None
        notes.append(
            "Ground truth contains no anomaly pixels, so AUC-ROC and recall are "
            "undefined; the false-positive rate is the meaningful number here "
            "(§11 edge case 3)."
        )
    elif n_anomaly == n_pixels:
        auc = None
        notes.append(
            "Ground truth contains only anomaly pixels, so AUC-ROC is undefined."
        )
    else:
        auc = float(roc_auc_score(flat_gt, flat_scores))

    # --- Operating point ---
    if threshold is None:
        thr = compute_threshold(flat_scores, threshold_method, threshold_percentile)
        if threshold_method == "percentile" and n_anomaly == 0:
            notes.append(
                f"A {threshold_percentile:g}th-percentile threshold flags a fixed "
                "fraction of pixels by construction, so the false-positive rate "
                "below reflects that choice rather than detector quality. Use "
                "threshold_method='otsu' for an adaptive comparison."
            )
    else:
        thr = float(threshold)

    predicted = flat_scores >= thr

    tp = int(np.count_nonzero(predicted & flat_gt))
    fp = int(np.count_nonzero(predicted & ~flat_gt))
    fn = int(np.count_nonzero(~predicted & flat_gt))
    tn = int(np.count_nonzero(~predicted & ~flat_gt))

    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0
        notes.append(
            "No pixels exceeded the threshold, so precision is undefined and "
            "reported as 0.0."
        )

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return DetectionMetrics(
        auc_roc=auc,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        threshold_used=float(thr),
        false_positive_rate=float(fpr),
        n_pixels=n_pixels,
        n_anomaly_pixels=n_anomaly,
        n_flagged_pixels=int(tp + fp),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        notes=tuple(notes),
    )


def mean_per_image_auc(
    scores: list[np.ndarray] | np.ndarray,
    ground_truths: list[np.ndarray] | np.ndarray,
) -> tuple[float, int]:
    """HAD100's headline metric: AUC-ROC per image, then averaged.

    This is **not** the same number as pooling every pixel from every image
    into one AUC. Pooling lets a scene whose scores happen to sit on a higher
    absolute scale dominate the ranking, whereas HAD100 scores each 64x64 image
    on its own and takes the mean -- which is what its published table of 19
    baseline results reports (``eval_results.py``: ``mAUC``). Comparing a
    pooled AUC against their mAUC would be comparing two different quantities.

    Images whose ground truth holds a single class are skipped rather than
    scored, and the count of images actually averaged is returned alongside so
    a report can state it.

    Args:
        scores: One anomaly map per image.
        ground_truths: Matching binary masks.

    Returns:
        ``(mean_auc, n_images_used)``.

    Raises:
        DegenerateInputError: the two sequences differ in length, are empty, or
            no image had both classes present.
    """
    scores = list(scores)
    ground_truths = list(ground_truths)

    if len(scores) != len(ground_truths):
        raise DegenerateInputError(
            f"Got {len(scores)} score maps but {len(ground_truths)} ground truths."
        )
    if not scores:
        raise DegenerateInputError("Cannot average AUC over an empty image set.")

    aucs: list[float] = []
    for score, gt in zip(scores, ground_truths):
        flat_scores, flat_gt = _validate(score, gt)
        n_pos = int(np.count_nonzero(flat_gt))
        if n_pos == 0 or n_pos == flat_gt.size:
            continue
        aucs.append(float(roc_auc_score(flat_gt, flat_scores)))

    if not aucs:
        raise DegenerateInputError(
            "No image had both anomaly and background pixels, so no AUC could be "
            "computed for any of them."
        )
    return float(np.mean(aucs)), len(aucs)


def roc_points(scores: np.ndarray, ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False-positive and true-positive rates for plotting an ROC curve.

    Raises:
        DegenerateInputError: the ground truth holds only one class, so no ROC
            curve exists.
    """
    flat_scores, flat_gt = _validate(scores, ground_truth)
    n_anomaly = int(np.count_nonzero(flat_gt))
    if n_anomaly == 0 or n_anomaly == flat_gt.size:
        raise DegenerateInputError(
            "An ROC curve requires both classes to be present in the ground truth."
        )
    fpr, tpr, _ = roc_curve(flat_gt, flat_scores)
    return fpr, tpr
