"""Thresholding helpers (FR-9.5)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.errors import DegenerateInputError
from pipeline.evaluation.thresholding import (
    compute_threshold,
    f1_optimal_threshold,
    mean_std_threshold,
    otsu_threshold,
    percentile_threshold,
)


def test_percentile_matches_numpy() -> None:
    scores = np.arange(101, dtype=float)
    assert percentile_threshold(scores, 99.0) == pytest.approx(99.0)
    assert percentile_threshold(scores, 50.0) == pytest.approx(50.0)


def test_percentile_ignores_shape() -> None:
    scores = np.arange(100, dtype=float).reshape(10, 10)
    assert percentile_threshold(scores, 90.0) == pytest.approx(np.percentile(scores, 90.0))


def test_percentile_rejects_out_of_range() -> None:
    with pytest.raises(DegenerateInputError):
        percentile_threshold(np.arange(10, dtype=float), 101.0)


def test_otsu_splits_a_bimodal_distribution() -> None:
    """The threshold must land in the gap between the two modes."""
    rng = np.random.default_rng(0)
    low = rng.normal(1.0, 0.1, 500)
    high = rng.normal(5.0, 0.1, 500)
    thr = otsu_threshold(np.concatenate([low, high]))
    assert 1.5 < thr < 4.5


def test_otsu_separates_the_classes_it_found() -> None:
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(0.0, 0.05, 400), rng.normal(2.0, 0.05, 100)])
    thr = otsu_threshold(scores)
    flagged = scores >= thr
    assert 90 <= int(flagged.sum()) <= 110


def test_otsu_refuses_a_constant_map() -> None:
    """A detector that produced one value everywhere has found nothing."""
    with pytest.raises(DegenerateInputError, match="identical"):
        otsu_threshold(np.full(100, 3.0))


def test_all_nan_scores_are_refused() -> None:
    with pytest.raises(DegenerateInputError, match="no finite values"):
        percentile_threshold(np.full(10, np.nan))


def test_compute_threshold_dispatches() -> None:
    scores = np.arange(101, dtype=float)
    assert compute_threshold(scores, "percentile", 99.0) == pytest.approx(99.0)
    assert compute_threshold(scores, "otsu") == pytest.approx(otsu_threshold(scores))


def test_compute_threshold_rejects_unknown_method() -> None:
    with pytest.raises(DegenerateInputError, match="Unknown threshold method"):
        compute_threshold(np.arange(10, dtype=float), "kmeans")


# --------------------------------------------------------------------------
# mean_std -- the rule drawn in the paper's Fig. 1 stage 6
# --------------------------------------------------------------------------


def test_mean_std_matches_its_formula() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(5.0, 2.0, 10000)
    thr = mean_std_threshold(scores, gamma=3.0)
    assert thr == pytest.approx(scores.mean() + 3.0 * scores.std())


def test_mean_std_adapts_where_percentile_cannot() -> None:
    """The distinction that matters for edge case 3.

    A percentile flags a fixed fraction whatever the scene holds. A
    mean-plus-sigma rule keys off the distribution's spread, so a tight
    anomaly-free distribution flags far fewer pixels than a scene with a real
    anomaly tail.
    """
    rng = np.random.default_rng(1)
    quiet = rng.normal(0.0, 1.0, 20000)
    with_anomalies = np.concatenate([rng.normal(0.0, 1.0, 19800), rng.normal(12.0, 1.0, 200)])

    quiet_flagged = (quiet >= mean_std_threshold(quiet, 3.0)).mean()
    anom_flagged = (with_anomalies >= mean_std_threshold(with_anomalies, 3.0)).mean()

    assert quiet_flagged < 0.01
    assert anom_flagged > quiet_flagged

    # A percentile, by contrast, flags exactly 1% of both.
    assert (quiet >= percentile_threshold(quiet, 99.0)).mean() == pytest.approx(0.01, abs=1e-3)
    assert (with_anomalies >= percentile_threshold(with_anomalies, 99.0)).mean() == pytest.approx(
        0.01, abs=1e-3
    )


def test_mean_std_refuses_a_constant_map() -> None:
    with pytest.raises(DegenerateInputError, match="zero variance"):
        mean_std_threshold(np.full(100, 2.0))


def test_mean_std_dispatches_with_gamma() -> None:
    rng = np.random.default_rng(2)
    scores = rng.normal(size=5000)
    assert compute_threshold(scores, "mean_std", gamma=2.0) == pytest.approx(
        mean_std_threshold(scores, 2.0)
    )


# --------------------------------------------------------------------------
# f1_optimal -- the paper's §V protocol
# --------------------------------------------------------------------------


def test_f1_optimal_finds_a_perfect_split() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    gt = np.array([0, 0, 0, 1, 1], dtype=bool)

    thr = f1_optimal_threshold(scores, gt)

    predicted = scores >= thr
    assert np.array_equal(predicted, gt)


def test_f1_optimal_beats_every_other_cut() -> None:
    """Verified by brute force against all candidate thresholds."""
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 30)])
    gt = np.zeros(230, dtype=bool)
    gt[200:] = True

    thr = f1_optimal_threshold(scores, gt)

    def f1_at(t: float) -> float:
        pred = scores >= t
        tp = np.count_nonzero(pred & gt)
        fp = np.count_nonzero(pred & ~gt)
        fn = np.count_nonzero(~pred & gt)
        if tp == 0:
            return 0.0
        p, r = tp / (tp + fp), tp / (tp + fn)
        return 2 * p * r / (p + r)

    best = max(f1_at(t) for t in np.unique(scores))
    assert f1_at(thr) == pytest.approx(best)


def test_f1_optimal_handles_ties() -> None:
    """A cut inside a run of equal scores is not realisable by `>= tau`."""
    scores = np.array([1.0, 1.0, 1.0, 5.0, 5.0])
    gt = np.array([0, 0, 0, 1, 1], dtype=bool)

    thr = f1_optimal_threshold(scores, gt)
    assert thr == pytest.approx(5.0)


def test_f1_optimal_refuses_a_split_with_no_anomalies() -> None:
    with pytest.raises(DegenerateInputError, match="no anomaly pixels"):
        f1_optimal_threshold(np.linspace(0, 1, 50), np.zeros(50, dtype=bool))


def test_f1_optimal_refuses_shape_mismatch() -> None:
    with pytest.raises(DegenerateInputError, match="shape mismatch"):
        f1_optimal_threshold(np.zeros(10), np.zeros(9, dtype=bool))


def test_f1_optimal_refuses_nan_scores() -> None:
    scores = np.array([0.1, np.nan, 0.9])
    gt = np.array([0, 0, 1], dtype=bool)
    with pytest.raises(DegenerateInputError):
        f1_optimal_threshold(scores, gt)


def test_dispatch_refuses_f1_optimal_without_a_held_out_split() -> None:
    """The guard that keeps the paper's §V number honest.

    Fitting the threshold on the images the run reports against would inflate
    precision, recall and F1, so it is refused rather than silently allowed.
    """
    scores = np.array([0.1, 0.5, 0.9])
    with pytest.raises(DegenerateInputError, match="held-out validation split"):
        compute_threshold(scores, "f1_optimal")


def test_dispatch_uses_the_validation_split_not_the_scored_data() -> None:
    val_scores = np.array([0.0, 0.1, 0.8, 0.9])
    val_gt = np.array([0, 0, 1, 1], dtype=bool)
    reported = np.array([100.0, 200.0, 300.0])

    thr = compute_threshold(
        reported,
        "f1_optimal",
        validation_scores=val_scores,
        validation_ground_truth=val_gt,
    )

    # The threshold comes from the validation range, not the reported range --
    # which is exactly what "held out" means.
    assert 0.0 <= thr <= 1.0
    assert thr == pytest.approx(f1_optimal_threshold(val_scores, val_gt))
