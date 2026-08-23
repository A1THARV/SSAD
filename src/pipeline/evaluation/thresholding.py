"""Turning a continuous anomaly score into a binary mask (FR-9.5).

Lives in ``evaluation`` rather than ``hyperspectral.score_fusion`` because both
branches and the metrics module need the same thresholds; duplicating Otsu in
two places would let the two drift apart.
"""

from __future__ import annotations

import numpy as np

from ..errors import DegenerateInputError

__all__ = [
    "percentile_threshold",
    "otsu_threshold",
    "mean_std_threshold",
    "f1_optimal_threshold",
    "compute_threshold",
]


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    """Return the finite score values, or raise if there are none.

    PRD §10 forbids returning a value derived from NaNs as if it were valid, so
    an all-NaN score map is an error rather than a silently empty mask.
    """
    arr = np.asarray(scores, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise DegenerateInputError(
            "Score map contains no finite values; cannot derive a threshold. "
            "This usually means an upstream module produced all-NaN scores."
        )
    return finite


def percentile_threshold(scores: np.ndarray, percentile: float = 99.0) -> float:
    """Threshold at a percentile of the score distribution.

    Note that this flags a *fixed fraction* of pixels by construction: at the
    99th percentile roughly 1% of pixels are flagged whether or not the scene
    contains any anomalies. That is intended for the anomaly-rate assumption
    standard in HAD work, but it means a zero-anomaly scene will still show a
    ~1% false-positive rate under this method (see §11 edge case 3).
    """
    if not 0.0 <= percentile <= 100.0:
        raise DegenerateInputError(f"percentile must be within [0, 100], got {percentile}")
    return float(np.percentile(_finite_scores(scores), percentile))


def otsu_threshold(scores: np.ndarray, n_bins: int = 256) -> float:
    """Otsu's method: the threshold maximising between-class variance.

    Unlike :func:`percentile_threshold` this adapts to the score distribution's
    shape, so a scene with no anomalies need not flag a fixed fraction.
    """
    finite = _finite_scores(scores)
    lo, hi = float(finite.min()), float(finite.max())
    if not hi > lo:
        raise DegenerateInputError(
            f"All scores are identical ({lo}); Otsu's method has no split to find. "
            "A constant score map usually indicates a degenerate scene or an "
            "upstream detector that failed to discriminate."
        )

    hist, edges = np.histogram(finite, bins=n_bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    weights = hist.astype(np.float64) / finite.size

    omega = np.cumsum(weights)                 # class-0 probability at each split
    mu = np.cumsum(weights * centers)          # class-0 cumulative mean
    mu_total = mu[-1]

    denom = omega * (1.0 - omega)
    valid = denom > 0
    if not np.any(valid):
        raise DegenerateInputError("Otsu's method found no valid split point.")

    between = np.zeros_like(denom)
    between[valid] = (mu_total * omega[valid] - mu[valid]) ** 2 / denom[valid]

    # A well-separated distribution leaves an empty gap between its modes, and
    # every split inside that gap scores identically. Taking ``argmax`` alone
    # would return the first such bin -- hard against the lower mode, where a
    # slightly longer tail would flip pixels across the threshold. Averaging
    # the plateau puts the threshold in the middle of the gap instead.
    best = float(between.max())
    plateau = np.flatnonzero(between >= best - 1e-12 * max(best, 1.0))
    return float(np.mean(centers[plateau]))


def mean_std_threshold(scores: np.ndarray, gamma: float = 3.0) -> float:
    """``T = mu_S + gamma * sigma_S`` -- the rule drawn in the paper's Fig. 1.

    Fig. 1 stage 6 ("Adaptive Thresholding") specifies this and nothing else,
    so it is implemented here to keep the code matching the figure. It is
    genuinely adaptive in a way a percentile is not: on a scene with no anomaly
    mode the score distribution is tight, so the threshold sits close to the
    bulk and flags few pixels, rather than flagging a fixed fraction whatever
    the scene contains.

    Args:
        scores: Fused anomaly scores.
        gamma: Standard deviations above the mean. Fig. 1 leaves it as a free
            parameter; 3.0 is the conventional starting point.
    """
    finite = _finite_scores(scores)
    sigma = float(finite.std())
    if sigma <= 0:
        raise DegenerateInputError(
            "Score map has zero variance, so a mean-plus-sigma threshold is undefined. "
            "A constant score map means the detector did not discriminate at all."
        )
    return float(finite.mean() + gamma * sigma)


def f1_optimal_threshold(scores: np.ndarray, ground_truth: np.ndarray) -> float:
    """The threshold maximising F1, fitted on labelled data.

    The paper's §V protocol reports precision, recall and F1 "at the operating
    threshold that maximizes F1 on a held-out validation split". This function
    implements the maximisation; **enforcing the held-out part is the caller's
    job** -- see :func:`compute_threshold`, which refuses to fit this on the
    same images a run reports against.

    Fitting a threshold on the data you then score at it inflates every number
    downstream, so this is the one thresholding method that cannot be used
    without a split.

    Args:
        scores: Anomaly scores from the validation images.
        ground_truth: Matching binary labels.

    Returns:
        The score value maximising F1. Ties resolve to the lowest such value,
        which is the more inclusive operating point.

    Raises:
        DegenerateInputError: shapes disagree, no finite scores, or the
            validation labels contain no anomalies (F1 is then zero
            everywhere and the maximum is meaningless).
    """
    flat = np.asarray(scores, dtype=np.float64).ravel()
    gt = np.asarray(ground_truth).astype(bool).ravel()

    if flat.shape != gt.shape:
        raise DegenerateInputError(
            f"Score/label shape mismatch when fitting an F1-optimal threshold: "
            f"{flat.shape} vs {gt.shape}."
        )
    finite = _finite_scores(flat)
    if finite.size != flat.size:
        raise DegenerateInputError(
            "Validation scores contain non-finite values; an F1-optimal threshold "
            "fitted over them would be meaningless."
        )
    n_pos = int(np.count_nonzero(gt))
    if n_pos == 0:
        raise DegenerateInputError(
            "The validation split contains no anomaly pixels, so F1 is zero at every "
            "threshold and has no maximum. Use a split that includes labelled "
            "anomalies, or choose threshold_method 'percentile' or 'otsu'."
        )

    # Sort once descending and sweep: at each cut the prefix is predicted
    # positive, so TP/FP accumulate and precision/recall follow directly. This
    # is O(n log n) rather than the O(n^2) of testing every candidate cut
    # against the whole array -- which matters on a 100-image split.
    order = np.argsort(-flat, kind="mergesort")
    sorted_scores = flat[order]
    sorted_gt = gt[order]

    tp = np.cumsum(sorted_gt)
    flagged = np.arange(1, flat.size + 1)
    precision = tp / flagged
    recall = tp / n_pos
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)

    # Only cuts at a value boundary are real thresholds; a cut inside a run of
    # equal scores cannot be realised by `scores >= tau`.
    is_boundary = np.empty(flat.size, dtype=bool)
    is_boundary[:-1] = sorted_scores[:-1] > sorted_scores[1:]
    is_boundary[-1] = True
    f1 = np.where(is_boundary, f1, -1.0)

    best = int(np.argmax(f1))
    return float(sorted_scores[best])


def compute_threshold(
    scores: np.ndarray,
    method: str = "percentile",
    percentile: float = 99.0,
    gamma: float = 3.0,
    validation_scores: np.ndarray | None = None,
    validation_ground_truth: np.ndarray | None = None,
) -> float:
    """Dispatch to the configured thresholding method (FR-9.5).

    Args:
        scores: The score map being thresholded.
        method: ``percentile``, ``otsu``, or ``f1_optimal``.
        percentile: Used by the ``percentile`` method.
        validation_scores / validation_ground_truth: Held-out data for
            ``f1_optimal``. Required for that method -- it is not fitted on
            ``scores``, because a threshold chosen on the reported data is not
            a held-out threshold.

    Raises:
        DegenerateInputError: unknown method, or ``f1_optimal`` requested
            without a validation split.
    """
    if method == "percentile":
        return percentile_threshold(scores, percentile)
    if method == "otsu":
        return otsu_threshold(scores)
    if method == "mean_std":
        return mean_std_threshold(scores, gamma)
    if method == "f1_optimal":
        if validation_scores is None or validation_ground_truth is None:
            raise DegenerateInputError(
                "threshold_method 'f1_optimal' requires a held-out validation split, "
                "which was not supplied. Fitting the threshold on the images the run "
                "reports against would inflate precision, recall and F1. Pass "
                "validation_scores and validation_ground_truth, or use 'percentile'."
            )
        return f1_optimal_threshold(validation_scores, validation_ground_truth)
    raise DegenerateInputError(
        f"Unknown threshold method {method!r} "
        "(expected 'percentile', 'otsu', 'mean_std' or 'f1_optimal')."
    )
