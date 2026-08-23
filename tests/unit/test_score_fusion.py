"""Fig. 1 stage 5: normalisation and fusion of the two branches."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.errors import DegenerateInputError, IncompatibleInputError
from pipeline.hyperspectral.score_fusion import fuse_scores, normalise_scores


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


def test_max_normalisation_puts_the_peak_at_one():
    scores = np.array([[0.0, 2.0], [4.0, 8.0]])
    out = normalise_scores(scores, "max")
    assert out.max() == pytest.approx(1.0)
    assert out[0, 0] == pytest.approx(0.0)


def test_max_normalisation_keeps_the_floor_where_it_was():
    # The documented difference from minmax: a branch whose scores sit high
    # above zero stays compressed into the top of the range.
    scores = np.array([100.0, 101.0, 102.0])
    out = normalise_scores(scores, "max")
    assert out.min() > 0.9
    assert normalise_scores(scores, "minmax").min() == pytest.approx(0.0)


def test_minmax_spans_the_unit_interval():
    scores = np.array([3.0, 5.0, 9.0])
    out = normalise_scores(scores, "minmax")
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_rank_normalisation_is_monotone_and_scale_free():
    scores = np.array([1.0, 50.0, 3.0, 900.0])
    a = normalise_scores(scores, "rank")
    b = normalise_scores(scores * 1000.0 + 7.0, "rank")
    assert a == pytest.approx(b)
    assert np.argsort(a).tolist() == np.argsort(scores).tolist()


def test_rank_normalisation_survives_a_single_huge_outlier():
    # minmax would crush the informative pixels into the bottom 0.1%.
    scores = np.array([1.0, 2.0, 3.0, 1e9])
    ranked = normalise_scores(scores, "rank")
    assert ranked[1] - ranked[0] == pytest.approx(ranked[2] - ranked[1])


def test_constant_map_normalises_to_zero_not_one():
    # A constant map carries no ranking information. Returning ones would make
    # every pixel maximally anomalous once fused.
    out = normalise_scores(np.full((3, 3), 7.0), "max")
    assert np.all(out == 0.0)


def test_normalisation_preserves_shape():
    assert normalise_scores(np.random.default_rng(0).normal(size=(4, 6))**2).shape == (4, 6)


def test_nan_scores_are_refused():
    with pytest.raises(DegenerateInputError):
        normalise_scores(np.array([1.0, np.nan]))


def test_unknown_normalisation_is_refused():
    with pytest.raises(DegenerateInputError, match="minmax"):
        normalise_scores(np.array([1.0, 2.0]), "zscore")


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def test_alpha_one_is_rx_alone():
    rx = np.array([[1.0, 4.0], [2.0, 8.0]])
    recon = np.zeros((2, 2))
    assert fuse_scores(rx, recon, alpha=1.0) == pytest.approx(normalise_scores(rx))


def test_alpha_zero_is_the_autoencoder_alone():
    # This is how the ablation runs: same code path, one weight.
    recon = np.array([[1.0, 4.0], [2.0, 8.0]])
    rx = np.zeros((2, 2))
    assert fuse_scores(rx, recon, alpha=0.0) == pytest.approx(normalise_scores(recon))


def test_weighted_fusion_is_the_convex_combination():
    rx = np.array([2.0, 4.0])
    recon = np.array([8.0, 4.0])
    got = fuse_scores(rx, recon, mode="weighted", alpha=0.25)
    want = 0.25 * normalise_scores(rx) + 0.75 * normalise_scores(recon)
    assert got == pytest.approx(want)


def test_max_fusion_takes_the_stronger_branch_per_pixel():
    rx = np.array([10.0, 0.0])
    recon = np.array([0.0, 10.0])
    assert fuse_scores(rx, recon, mode="max") == pytest.approx([1.0, 1.0])


def test_fusion_rescues_a_target_only_one_branch_sees():
    # The reason fusion exists at all. RX ranks pixel 3 top; the autoencoder
    # ranks pixel 1 top; only the fused map ranks both above the background.
    rx = np.array([1.0, 1.0, 1.0, 9.0])
    recon = np.array([9.0, 1.0, 1.0, 1.0])
    fused = fuse_scores(rx, recon, mode="max")
    assert fused[0] > fused[1] and fused[3] > fused[2]


def test_branches_on_different_grids_are_refused():
    with pytest.raises(IncompatibleInputError, match="same grid"):
        fuse_scores(np.zeros((8, 8)), np.zeros((6, 6)))


def test_alpha_outside_the_unit_interval_is_refused():
    with pytest.raises(DegenerateInputError):
        fuse_scores(np.zeros((2, 2)), np.zeros((2, 2)), alpha=1.5)


def test_unknown_mode_is_refused():
    with pytest.raises(DegenerateInputError, match="weighted"):
        fuse_scores(np.zeros((2, 2)), np.zeros((2, 2)), mode="mean")


def test_fusion_output_stays_in_the_unit_interval():
    rng = np.random.default_rng(0)
    rx = rng.gamma(2.0, size=(16, 16))
    recon = rng.gamma(2.0, size=(16, 16))
    for mode in ("weighted", "max"):
        for norm in ("max", "minmax", "rank"):
            out = fuse_scores(rx, recon, mode=mode, normalisation=norm)
            assert out.min() >= 0.0 and out.max() <= 1.0
