"""Detection metrics (Roadmap 0.5, FR-9.11).

The primary case is hand-computed: a 10-pixel score map whose confusion matrix,
AUC-ROC, precision, recall and F1 are all worked out below by hand, so the test
checks the arithmetic rather than merely that the function returns something.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.errors import DegenerateInputError
from pipeline.evaluation.metrics import (
    DetectionMetrics,
    compute_detection_metrics,
    roc_points,
)

# Ten pixels, descending scores. Anomalies sit at 0.9 and 0.7, so one negative
# (0.8) outranks a positive -- deliberately imperfect, to avoid an AUC of 1.0
# that would pass even with a broken implementation.
SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
GT = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)

# By hand, at threshold 0.65 -> flagged = {0.9, 0.8, 0.7}:
#   TP = {0.9, 0.7} = 2   FP = {0.8} = 1   FN = 0   TN = 7
#   precision = 2/3, recall = 2/2 = 1.0, F1 = 2(2/3)(1)/((2/3)+1) = 0.8
#   FPR = 1/8 = 0.125
# AUC-ROC = concordant (positive, negative) pairs / (2 x 8):
#   0.9 beats all 8 negatives; 0.7 beats 7 of them (0.8 outranks it) -> 15/16
THRESHOLD = 0.65
EXPECTED_AUC = 15 / 16


def test_hand_computed_confusion_matrix() -> None:
    m = compute_detection_metrics(SCORES, GT, threshold=THRESHOLD)

    assert m.true_positives == 2
    assert m.false_positives == 1
    assert m.false_negatives == 0
    assert m.n_flagged_pixels == 3
    assert m.n_pixels == 10
    assert m.n_anomaly_pixels == 2


def test_hand_computed_rates() -> None:
    m = compute_detection_metrics(SCORES, GT, threshold=THRESHOLD)

    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(0.8)
    assert m.false_positive_rate == pytest.approx(0.125)
    assert m.threshold_used == pytest.approx(THRESHOLD)


def test_hand_computed_auc() -> None:
    m = compute_detection_metrics(SCORES, GT, threshold=THRESHOLD)
    assert m.auc_roc == pytest.approx(EXPECTED_AUC)


def test_auc_is_threshold_independent() -> None:
    """AUC-ROC must not move when only the operating point changes."""
    low = compute_detection_metrics(SCORES, GT, threshold=0.05)
    high = compute_detection_metrics(SCORES, GT, threshold=0.95)
    assert low.auc_roc == pytest.approx(high.auc_roc)
    assert low.n_flagged_pixels != high.n_flagged_pixels


def test_perfect_detector_scores_one() -> None:
    scores = np.array([0.9, 0.8, 0.1, 0.05])
    gt = np.array([1, 1, 0, 0], dtype=bool)
    m = compute_detection_metrics(scores, gt, threshold=0.5)
    assert m.auc_roc == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)


def test_works_on_2d_maps() -> None:
    """Real anomaly maps arrive as (H, W), not flattened."""
    flat = compute_detection_metrics(SCORES, GT, threshold=THRESHOLD)
    square = compute_detection_metrics(
        SCORES.reshape(2, 5), GT.reshape(2, 5), threshold=THRESHOLD
    )
    assert square.auc_roc == pytest.approx(flat.auc_roc)
    assert square.f1 == pytest.approx(flat.f1)


def test_percentile_threshold_flags_expected_count() -> None:
    scores = np.arange(1000, dtype=float)
    gt = np.zeros(1000, dtype=bool)
    gt[-5:] = True
    m = compute_detection_metrics(scores, gt, threshold_percentile=99.0)
    # p99 of 0..999 is 989.1, so 990..999 clear it: 10 pixels.
    assert m.n_flagged_pixels == 10
    assert m.recall == pytest.approx(1.0)


# --------------------------------------------------------------------------
# §11 edge case 3 -- a scene with zero true anomalies must still report
# --------------------------------------------------------------------------


def test_zero_anomaly_ground_truth_does_not_crash() -> None:
    scores = np.linspace(0, 1, 100)
    gt = np.zeros(100, dtype=bool)

    m = compute_detection_metrics(scores, gt, threshold_percentile=99.0)

    assert m.auc_roc is None, "AUC-ROC is undefined with only one class present"
    assert m.n_anomaly_pixels == 0
    assert m.true_positives == 0
    assert 0.0 <= m.false_positive_rate <= 1.0
    assert m.notes, "the undefined metrics must be explained, not left bare"


def test_zero_anomaly_notes_explain_the_percentile_artefact() -> None:
    """A percentile threshold flags a fixed fraction whether or not anomalies exist."""
    scores = np.linspace(0, 1, 100)
    gt = np.zeros(100, dtype=bool)
    m = compute_detection_metrics(scores, gt, threshold_percentile=99.0)
    joined = " ".join(m.notes).lower()
    assert "percentile" in joined


def test_all_anomaly_ground_truth_is_also_handled() -> None:
    scores = np.linspace(0, 1, 20)
    gt = np.ones(20, dtype=bool)
    m = compute_detection_metrics(scores, gt, threshold=0.5)
    assert m.auc_roc is None
    assert m.false_positive_rate == 0.0
    assert m.notes


def test_nothing_flagged_gives_defined_precision() -> None:
    scores = np.zeros(50)
    gt = np.zeros(50, dtype=bool)
    gt[0] = True
    m = compute_detection_metrics(scores, gt, threshold=10.0)
    assert m.n_flagged_pixels == 0
    assert m.precision == 0.0
    assert m.f1 == 0.0
    assert any("precision" in n.lower() for n in m.notes)


# --------------------------------------------------------------------------
# Refusals -- PRD §10, no silently meaningless numbers
# --------------------------------------------------------------------------


def test_nan_scores_are_refused() -> None:
    scores = np.array([0.1, np.nan, 0.3, 0.4])
    gt = np.array([0, 0, 1, 0], dtype=bool)
    with pytest.raises(DegenerateInputError, match="non-finite"):
        compute_detection_metrics(scores, gt, threshold=0.2)


def test_infinite_scores_are_refused() -> None:
    scores = np.array([0.1, np.inf, 0.3, 0.4])
    gt = np.array([0, 0, 1, 0], dtype=bool)
    with pytest.raises(DegenerateInputError):
        compute_detection_metrics(scores, gt, threshold=0.2)


def test_shape_mismatch_is_refused() -> None:
    with pytest.raises(DegenerateInputError, match="shape"):
        compute_detection_metrics(np.zeros((4, 5)), np.zeros((5, 4), dtype=bool))


def test_empty_input_is_refused() -> None:
    with pytest.raises(DegenerateInputError):
        compute_detection_metrics(np.array([]), np.array([], dtype=bool))


def test_class_map_ground_truth_is_refused() -> None:
    """A Salinas-style land-cover map must be reduced to binary first."""
    scores = np.linspace(0, 1, 12)
    class_map = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
    with pytest.raises(DegenerateInputError, match="class map"):
        compute_detection_metrics(scores, class_map, threshold=0.5)


def test_zero_one_integer_ground_truth_is_accepted() -> None:
    """An int mask holding only 0/1 is a legitimate binary mask."""
    m = compute_detection_metrics(SCORES, GT.astype(np.uint8), threshold=THRESHOLD)
    assert m.auc_roc == pytest.approx(EXPECTED_AUC)


# --------------------------------------------------------------------------
# Serialization -- metrics.json round-trip
# --------------------------------------------------------------------------


def test_metrics_round_trip_through_dict() -> None:
    m = compute_detection_metrics(SCORES, GT, threshold=THRESHOLD)
    restored = DetectionMetrics.from_dict(m.to_dict())
    assert restored == m


def test_none_auc_survives_serialization() -> None:
    scores = np.linspace(0, 1, 10)
    gt = np.zeros(10, dtype=bool)
    m = compute_detection_metrics(scores, gt, threshold=0.5)
    assert DetectionMetrics.from_dict(m.to_dict()).auc_roc is None


# --------------------------------------------------------------------------
# ROC curve points
# --------------------------------------------------------------------------


def test_roc_points_are_monotonic() -> None:
    fpr, tpr = roc_points(SCORES, GT)
    assert np.all(np.diff(fpr) >= 0)
    assert np.all(np.diff(tpr) >= 0)
    assert fpr[0] == 0.0 and tpr[-1] == 1.0


def test_roc_points_refuse_single_class() -> None:
    with pytest.raises(DegenerateInputError, match="both classes"):
        roc_points(np.linspace(0, 1, 10), np.zeros(10, dtype=bool))
