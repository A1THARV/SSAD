"""RX detector, paper Eq. 2 (FR-9.2).

The primary test is a hand-computed 3-pixel toy scene: the Mahalanobis distance
is worked out below in closed form, so the test checks the arithmetic rather
than that the function returns an array of the right shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.errors import DegenerateInputError
from pipeline.hyperspectral.rx_detector import background_statistics, rx_score


def test_hand_computed_two_band_scene() -> None:
    """Four pixels, two bands, worked through Eq. 2 by hand.

    Pixels (0,0), (2,0), (0,2), (2,2). Mean is (1,1); the sample covariance
    (n-1 = 3 denominator) is diag(4/3, 4/3), so its inverse is diag(3/4, 3/4).
    Each pixel sits at (+-1, +-1) from the mean, giving

        RX = 3/4 * 1 + 3/4 * 1 = 1.5

    for all four. Ridge is zero here so the closed form holds exactly.
    """
    cube = np.array([[[0.0, 0.0], [2.0, 0.0]], [[0.0, 2.0], [2.0, 2.0]]])
    scores = rx_score(cube, ridge=0.0)

    assert scores.shape == (2, 2)
    assert scores == pytest.approx(np.full((2, 2), 1.5))


def test_hand_computed_background_statistics() -> None:
    cube = np.array([[[0.0, 0.0], [2.0, 0.0]], [[0.0, 2.0], [2.0, 2.0]]])
    mean, inv_cov = background_statistics(cube, ridge=0.0)

    assert mean == pytest.approx([1.0, 1.0])
    assert inv_cov == pytest.approx(np.diag([0.75, 0.75]))


def test_mahalanobis_reduces_to_scaled_distance_in_one_band() -> None:
    """With a single band, RX is the squared z-score."""
    values = np.array([[[0.0]], [[1.0]], [[2.0]], [[3.0]]], dtype=float).reshape(4, 1, 1)
    scores = rx_score(values, ridge=0.0)

    data = np.array([0.0, 1.0, 2.0, 3.0])
    expected = ((data - data.mean()) ** 2) / data.var(ddof=1)
    assert scores.ravel() == pytest.approx(expected)


# --------------------------------------------------------------------------
# The statistical property that matters -- FR-9.2's acceptance test
# --------------------------------------------------------------------------


def test_outliers_score_strictly_above_background() -> None:
    """FR-9.2: on Gaussian background with injected outliers, every outlier
    must outscore every background pixel -- not merely run without error."""
    rng = np.random.default_rng(0)
    cube = rng.normal(0.0, 1.0, size=(40, 40, 12))

    outliers = [(5, 5), (12, 30), (33, 8)]
    for r, c in outliers:
        cube[r, c, :] = 12.0

    scores = rx_score(cube, ridge=1e-8)

    mask = np.zeros(scores.shape, dtype=bool)
    for r, c in outliers:
        mask[r, c] = True

    assert scores[mask].min() > scores[~mask].max()


def test_score_is_invariant_to_band_scaling() -> None:
    """Mahalanobis distance is scale-invariant, unlike Euclidean distance.

    Rescaling one band must not change the ranking -- if it did, RX would be
    reporting sensor gain rather than anomalousness.
    """
    rng = np.random.default_rng(1)
    cube = rng.normal(0.0, 1.0, size=(20, 20, 6))
    cube[7, 7, :] = 8.0

    rescaled = cube.copy()
    rescaled[:, :, 2] *= 1000.0

    a = rx_score(cube, ridge=1e-10)
    b = rx_score(rescaled, ridge=1e-10)
    assert np.argmax(a) == np.argmax(b)
    assert a == pytest.approx(b, rel=1e-4)


def test_scores_are_non_negative() -> None:
    rng = np.random.default_rng(2)
    scores = rx_score(rng.normal(size=(15, 15, 5)))
    assert scores.min() >= 0.0


def test_background_mask_excludes_anomalies_from_statistics() -> None:
    """Estimating background on background alone should sharpen the contrast."""
    rng = np.random.default_rng(3)
    cube = rng.normal(0.0, 1.0, size=(30, 30, 8))
    cube[10:13, 10:13, :] = 6.0

    anomaly = np.zeros(cube.shape[:2], dtype=bool)
    anomaly[10:13, 10:13] = True

    with_all = rx_score(cube, ridge=1e-8)
    with_clean = rx_score(cube, ridge=1e-8, background_mask=~anomaly)

    contrast_all = with_all[anomaly].mean() / with_all[~anomaly].mean()
    contrast_clean = with_clean[anomaly].mean() / with_clean[~anomaly].mean()
    assert contrast_clean > contrast_all


# --------------------------------------------------------------------------
# §11 edge case 2 -- near-singular covariance
# --------------------------------------------------------------------------


def test_homogeneous_scene_does_not_raise_linalg_error() -> None:
    """A near-homogeneous scene has a near-singular covariance."""
    rng = np.random.default_rng(4)
    cube = np.ones((20, 20, 10)) * 5.0
    cube += rng.normal(0.0, 1e-9, size=cube.shape)

    scores = rx_score(cube, ridge=1e-6)
    assert np.all(np.isfinite(scores))


def test_more_bands_than_pixels_is_handled() -> None:
    """B > N makes the sample covariance rank-deficient by construction."""
    rng = np.random.default_rng(5)
    cube = rng.normal(size=(3, 3, 50))  # 9 pixels, 50 bands
    scores = rx_score(cube, ridge=1e-4)
    assert np.all(np.isfinite(scores))
    assert scores.shape == (3, 3)


def test_perfectly_constant_scene_is_refused() -> None:
    """No background variation means nothing to score against."""
    with pytest.raises(DegenerateInputError, match="no background variation|zero or non-finite"):
        rx_score(np.full((10, 10, 4), 3.0), ridge=1e-6)


def test_duplicated_band_does_not_break_inversion() -> None:
    """An exactly collinear band makes the covariance singular."""
    rng = np.random.default_rng(6)
    base = rng.normal(size=(20, 20, 4))
    cube = np.concatenate([base, base[:, :, :1]], axis=2)  # band 4 == band 0

    scores = rx_score(cube, ridge=1e-6)
    assert np.all(np.isfinite(scores))


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_two_dimensional_input_is_refused() -> None:
    with pytest.raises(DegenerateInputError, match=r"\(H, W, B\)"):
        rx_score(np.zeros((10, 10)))


def test_nan_input_is_refused() -> None:
    cube = np.random.default_rng(7).normal(size=(10, 10, 4))
    cube[3, 3, 1] = np.nan
    with pytest.raises(DegenerateInputError, match="non-finite"):
        rx_score(cube)


def test_negative_ridge_is_refused() -> None:
    cube = np.random.default_rng(8).normal(size=(10, 10, 4))
    with pytest.raises(DegenerateInputError, match="non-negative"):
        rx_score(cube, ridge=-1.0)


def test_band_count_is_never_assumed() -> None:
    """§11 edge case 5 -- ABU alone spans seven band counts."""
    rng = np.random.default_rng(9)
    for n_bands in (3, 102, 162, 188, 205, 276):
        scores = rx_score(rng.normal(size=(8, 8, n_bands)), ridge=1e-4)
        assert scores.shape == (8, 8)


# --------------------------------------------------------------------------
# Regularisation -- it changes accuracy, not just numerical safety
# --------------------------------------------------------------------------


def test_ridge_is_scaled_to_the_data_not_absolute() -> None:
    """A fixed absolute ridge would behave differently on scenes whose radiance
    units differ by orders of magnitude. Scaling by trace(Sigma)/B makes one
    setting portable, so rescaling a scene must not change its score ranking."""
    rng = np.random.default_rng(20)
    cube = rng.normal(size=(20, 20, 8))
    cube[5, 5, :] = 7.0

    a = rx_score(cube, ridge=1e-3)
    b = rx_score(cube * 1000.0, ridge=1e-3)

    assert a == pytest.approx(b, rel=1e-6)


def test_stronger_ridge_damps_trailing_directions() -> None:
    """The mechanism behind the spurious U-shaped dimensionality curve.

    RX weights each direction by 1/lambda, so a near-null direction produces an
    enormous, noise-dominated term. Raising the ridge must shrink the score of
    a pixel that is anomalous only along that direction.
    """
    rng = np.random.default_rng(21)
    n = 2000
    strong = rng.normal(0.0, 10.0, n)
    weak = rng.normal(0.0, 1e-3, n)
    flat = np.column_stack([strong, weak])
    cube = flat.reshape(n, 1, 2).copy()
    cube[0, 0, :] = [0.0, 5e-3]  # ordinary along PC1, extreme along the weak axis

    weak_ridge = rx_score(cube, ridge=1e-8)[0, 0]
    strong_ridge = rx_score(cube, ridge=1e-1)[0, 0]

    assert strong_ridge < weak_ridge


def test_ledoit_wolf_needs_no_tuning_and_stays_finite() -> None:
    rng = np.random.default_rng(22)
    cube = rng.normal(size=(30, 30, 12))
    cube[4, 4, :] = 9.0

    scores = rx_score(cube, shrinkage="ledoit_wolf")

    assert np.all(np.isfinite(scores))
    assert scores[4, 4] == scores.max()


def test_ledoit_wolf_survives_more_bands_than_pixels() -> None:
    """The regime where the sample covariance is rank-deficient by construction."""
    rng = np.random.default_rng(23)
    scores = rx_score(rng.normal(size=(4, 4, 60)), shrinkage="ledoit_wolf")
    assert np.all(np.isfinite(scores))


def test_unknown_shrinkage_is_refused() -> None:
    rng = np.random.default_rng(24)
    with pytest.raises(DegenerateInputError, match="Unknown shrinkage"):
        rx_score(rng.normal(size=(8, 8, 4)), shrinkage="oracle")
