"""Band masking, clipping and PCA reduction (FR-9.1, FR-9.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.config import PreprocessConfig
from pipeline.errors import (
    DegenerateInputError,
    IncompatibleInputError,
    ModelNotFittedError,
)
from pipeline.hyperspectral.loader import HSICube
from pipeline.hyperspectral.preprocess import (
    BandReducer,
    CorpusBlocks,
    fit_corpus_reducer,
    clip_outliers,
    compute_good_bands,
    preprocess_cube,
)

REPO = Path(__file__).resolve().parents[2]
ABU = REPO / "data/hyperspectral/abu"


def cube_with(data: np.ndarray, wavelengths=None) -> HSICube:
    return HSICube(data=data, wavelengths=wavelengths, name="t")


# --------------------------------------------------------------------------
# §11 edge case 1 -- water-absorption and dead bands
# --------------------------------------------------------------------------


def test_water_absorption_bands_are_masked() -> None:
    wavelengths = np.array([500.0, 1400.0, 1600.0, 1900.0, 2200.0])
    rng = np.random.default_rng(0)
    cube = cube_with(rng.normal(size=(8, 8, 5)), wavelengths)

    keep = compute_good_bands(cube, PreprocessConfig())

    assert keep.tolist() == [True, False, True, False, True]


def test_deliberately_zeroed_band_is_excluded() -> None:
    """FR-9.1's acceptance test, and it must work without a wavelength table."""
    rng = np.random.default_rng(1)
    data = rng.normal(size=(10, 10, 6))
    data[:, :, 3] = 0.0

    keep = compute_good_bands(cube_with(data), PreprocessConfig())

    assert keep[3] == np.False_
    assert keep.sum() == 5


def test_masking_works_without_wavelengths() -> None:
    rng = np.random.default_rng(2)
    keep = compute_good_bands(cube_with(rng.normal(size=(6, 6, 4))), PreprocessConfig())
    assert keep.all()


def test_preset_overrides_wavelength_masking() -> None:
    """C3 -- a benchmark's own selection wins over our derived one."""
    rng = np.random.default_rng(3)
    cube = cube_with(rng.normal(size=(4, 4, 425)), np.linspace(380, 2500, 425))
    cfg = PreprocessConfig(good_bands_preset="had100_aviris_ng")

    keep = compute_good_bands(cube, cfg)

    assert keep.sum() == 276


def test_masking_everything_is_refused() -> None:
    wavelengths = np.array([1400.0, 1420.0])
    rng = np.random.default_rng(4)
    cube = cube_with(rng.normal(size=(5, 5, 2)), wavelengths)

    with pytest.raises(DegenerateInputError, match="remove all"):
        compute_good_bands(cube, PreprocessConfig())


# --------------------------------------------------------------------------
# §11 edge case 6 -- saturated / dead pixels
# --------------------------------------------------------------------------


def test_clipping_bounds_extreme_values() -> None:
    rng = np.random.default_rng(5)
    data = rng.normal(0.0, 1.0, size=(30, 30, 3))
    data[0, 0, :] = 1e6

    clipped = clip_outliers(data, 99.0)

    assert clipped.max() < 1e5
    assert clipped[0, 0, 0] == pytest.approx(np.percentile(data[..., 0], 99.0))


def test_clipping_is_per_band() -> None:
    """Bands sit at different radiance levels; one global ceiling flattens them."""
    data = np.zeros((10, 10, 2))
    data[..., 0] = 1.0
    data[..., 1] = 1000.0
    data[0, 0, 0] = 500.0

    clipped = clip_outliers(data, 95.0)

    assert clipped[..., 1].min() == pytest.approx(1000.0)
    assert clipped[0, 0, 0] < 500.0


def test_clipping_at_100_is_a_no_op() -> None:
    rng = np.random.default_rng(6)
    data = rng.normal(size=(5, 5, 3))
    assert np.array_equal(clip_outliers(data, 100.0), data)


def test_outliers_do_not_dominate_pca_after_clipping() -> None:
    """The point of edge case 6: one saturated pixel must not steer PC1."""
    rng = np.random.default_rng(7)
    data = rng.normal(size=(40, 40, 8))
    data[..., 0] *= 5.0          # genuine dominant direction
    contaminated = data.copy()
    contaminated[0, 0, 7] = 1e7  # a single saturated pixel in a quiet band

    clean = BandReducer(n_components=2).fit(data)
    dirty = BandReducer(n_components=2).fit(clip_outliers(contaminated, 99.9))

    lead_clean = int(np.argmax(np.abs(clean.components_[0])))
    lead_dirty = int(np.argmax(np.abs(dirty.components_[0])))
    assert lead_clean == lead_dirty


# --------------------------------------------------------------------------
# FR-9.3 -- fit once, transform many
# --------------------------------------------------------------------------


def test_transform_before_fit_is_refused() -> None:
    with pytest.raises(ModelNotFittedError, match="before fit"):
        BandReducer(n_components=2).transform(np.zeros((4, 4, 6)))


def test_fitted_reducer_is_reusable_and_consistent() -> None:
    rng = np.random.default_rng(8)
    train = rng.normal(size=(50, 50, 10))
    scene = rng.normal(size=(20, 20, 10))

    reducer = BandReducer(n_components=4).fit(train)
    once = reducer.transform(scene)
    twice = reducer.transform(scene)

    assert once.shape == (20, 20, 4)
    assert np.array_equal(once, twice)


def test_transform_preserves_spatial_shape() -> None:
    rng = np.random.default_rng(9)
    reducer = BandReducer(n_components=3).fit(rng.normal(size=(30, 30, 12)))
    assert reducer.transform(rng.normal(size=(7, 11, 12))).shape == (7, 11, 3)


def test_cross_sensor_transform_is_refused() -> None:
    """The C2 constraint, enforced: 162-band and 276-band cubes cannot share axes."""
    rng = np.random.default_rng(10)
    reducer = BandReducer(n_components=5).fit(rng.normal(size=(40, 40, 276)))

    with pytest.raises(IncompatibleInputError, match="fitted on 276 bands"):
        reducer.transform(rng.normal(size=(10, 10, 162)))


def test_variance_target_selects_components() -> None:
    rng = np.random.default_rng(11)
    base = rng.normal(size=(200, 3))
    data = np.hstack([base, base @ rng.normal(size=(3, 7))]).reshape(200, 1, 10)

    reducer = BandReducer(n_components=None, variance=0.99).fit(data)

    assert reducer.n_output_components <= 4
    assert reducer.explained_variance() >= 0.99


def test_asking_for_more_components_than_supported_is_refused() -> None:
    """sklearn would silently return fewer, changing the model width."""
    rng = np.random.default_rng(12)
    with pytest.raises(DegenerateInputError, match="at most"):
        BandReducer(n_components=50).fit(rng.normal(size=(4, 4, 8)))


def test_pca_is_deterministic_under_a_seed() -> None:
    rng = np.random.default_rng(13)
    data = rng.normal(size=(60, 60, 15))
    a = BandReducer(n_components=5).fit(data, seed=0).transform(data)
    b = BandReducer(n_components=5).fit(data, seed=0).transform(data)
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_preprocess_cube_reduces_to_requested_width(tmp_path: Path) -> None:
    rng = np.random.default_rng(14)
    cube = cube_with(rng.normal(size=(24, 24, 40)).astype(np.float32))
    cfg = PreprocessConfig(pca_n_components=30, cache_reduced=False)

    reduced, reducer = preprocess_cube(cube, cfg)

    assert reduced.shape == (24, 24, 30)
    assert reduced.dtype == np.float32
    assert reducer.is_fitted


def test_preprocess_sets_good_bands_on_the_cube() -> None:
    rng = np.random.default_rng(15)
    data = rng.normal(size=(12, 12, 10)).astype(np.float32)
    data[:, :, 2] = 0.0
    cube = cube_with(data)

    _, _ = preprocess_cube(cube, PreprocessConfig(pca_n_components=4, cache_reduced=False))

    assert cube.good_bands[2] == np.False_
    assert cube.good_bands.sum() == 9


def test_cache_round_trips(tmp_path: Path) -> None:
    rng = np.random.default_rng(16)
    scene = tmp_path / "scene.npy"
    data = rng.normal(size=(16, 16, 20)).astype(np.float32)
    np.save(scene, data)

    cube = HSICube(data=data, name="scene", source_path=str(scene))
    cfg = PreprocessConfig(pca_n_components=6, cache_reduced=True,
                           cache_dir=str(tmp_path / "cache"))

    first, reducer = preprocess_cube(cube, cfg)
    assert list((tmp_path / "cache").glob("*.npy")), "nothing was cached"

    second, _ = preprocess_cube(cube, cfg, reducer=reducer)
    assert np.array_equal(first, second)


@pytest.mark.needs_data
@pytest.mark.skipif(not ABU.is_dir(), reason="ABU not downloaded")
def test_real_scenes_of_any_band_count_reduce_to_the_same_width() -> None:
    """§11 edge case 5 through the whole preprocessing path."""
    from pipeline.config import DatasetConfig
    from pipeline.hyperspectral.loader import load_cube

    cfg = PreprocessConfig(pca_n_components=30, cache_reduced=False)
    widths = set()

    for path in sorted(ABU.glob("abu-urban-*.mat")):
        ds = DatasetConfig(name=path.stem, kind="hyperspectral", path=str(path),
                           data_key="data", ground_truth_path=str(path),
                           ground_truth_key="map")
        reduced, _ = preprocess_cube(load_cube(ds), cfg)
        widths.add(reduced.shape[2])

    assert widths == {30}, f"all scenes must reduce to 30 components, got {widths}"


# --------------------------------------------------------------------------
# Streaming corpus PCA -- exact, not sampled
# --------------------------------------------------------------------------


def test_streaming_matches_in_memory_exactly() -> None:
    """The whole justification: streaming is exact, so no sample to defend."""
    rng = np.random.default_rng(40)
    blocks = [rng.normal(size=(12, 12, 9)) + 50.0 for _ in range(7)]
    stacked = np.concatenate([b.reshape(-1, 9) for b in blocks])

    direct = BandReducer(n_components=4).fit(stacked)
    streamed = BandReducer(n_components=4).fit_streaming(blocks)

    assert streamed.n_samples_seen_ == stacked.shape[0]
    assert streamed.mean_ == pytest.approx(direct.mean_)
    assert streamed.explained_variance_ == pytest.approx(direct.explained_variance_)
    assert streamed.components_ == pytest.approx(direct.components_, abs=1e-9)


def test_streaming_projection_is_identical() -> None:
    rng = np.random.default_rng(41)
    blocks = [rng.normal(size=(10, 10, 8)) * 3.0 + 100.0 for _ in range(5)]
    stacked = np.concatenate([b.reshape(-1, 8) for b in blocks])
    scene = rng.normal(size=(6, 6, 8)) + 100.0

    a = BandReducer(n_components=3).fit(stacked).transform(scene)
    b = BandReducer(n_components=3).fit_streaming(blocks).transform(scene)
    assert a == pytest.approx(b, abs=1e-6)


def test_streaming_holds_precision_on_a_large_offset() -> None:
    """Why two passes rather than E[xx^T] - mu mu^T.

    The one-pass identity cancels catastrophically when the mean dwarfs the
    variance -- the regime reflectance data sits in.
    """
    rng = np.random.default_rng(42)
    blocks = [rng.normal(scale=1e-3, size=(20, 6)) + 1e5 for _ in range(4)]
    stacked = np.concatenate(blocks)

    streamed = BandReducer(n_components=3).fit_streaming(blocks)
    direct = BandReducer(n_components=3).fit(stacked)

    assert streamed.explained_variance_ == pytest.approx(
        direct.explained_variance_, rel=1e-6
    )
    assert np.all(streamed.explained_variance_ > 0)


def test_streaming_rejects_a_one_shot_iterator() -> None:
    """A generator would silently leave the second pass empty."""
    rng = np.random.default_rng(43)
    gen = (rng.normal(size=(5, 4)) for _ in range(3))
    with pytest.raises(DegenerateInputError, match="re-iterable"):
        BandReducer(n_components=2).fit_streaming(gen)


def test_streaming_refuses_a_mixed_sensor_corpus() -> None:
    """162-band and 276-band crops cannot share one basis (change C2)."""
    rng = np.random.default_rng(44)
    blocks = [rng.normal(size=(10, 162)), rng.normal(size=(10, 276))]
    with pytest.raises(IncompatibleInputError, match="mixes band counts"):
        BandReducer(n_components=5).fit_streaming(blocks)


def test_streaming_refuses_an_empty_corpus() -> None:
    with pytest.raises(DegenerateInputError, match="empty corpus"):
        BandReducer(n_components=2).fit_streaming([])


def test_corpus_blocks_is_re_iterable(tmp_path: Path) -> None:
    rng = np.random.default_rng(45)
    paths = []
    for i in range(4):
        p = tmp_path / f"crop_{i}.npy"
        np.save(p, rng.normal(size=(8, 8, 6)).astype(np.float32))
        paths.append(p)

    blocks = CorpusBlocks(paths)
    first = [b.shape for b in blocks]
    second = [b.shape for b in blocks]
    assert first == second == [(8, 8, 6)] * 4


def test_fit_corpus_reducer_end_to_end(tmp_path: Path) -> None:
    rng = np.random.default_rng(46)
    paths = []
    for i in range(6):
        p = tmp_path / f"crop_{i}.npy"
        np.save(p, (rng.normal(size=(8, 8, 10)) + 20.0).astype(np.float32))
        paths.append(p)

    cfg = PreprocessConfig(pca_n_components=4, cache_reduced=False)
    reducer = fit_corpus_reducer(paths, cfg)

    assert reducer.is_fitted
    assert reducer.n_output_components == 4
    assert reducer.n_samples_seen_ == 6 * 64


# --------------------------------------------------------------------------
# Subspace angle -- the right way to ask whether two fits agree
# --------------------------------------------------------------------------


def test_identical_fits_have_zero_subspace_angle() -> None:
    rng = np.random.default_rng(47)
    data = rng.normal(size=(400, 12))
    a = BandReducer(n_components=5).fit(data)
    b = BandReducer(n_components=5).fit(data)
    # Bounded at 1e-4 deg rather than 0: the angle comes from arccos of a
    # singular value, and arccos is ill-conditioned at 1, so identical bases
    # bottom out around sqrt(eps). Unlike SAM there is no atan2 reformulation
    # available here, since the input is a singular value rather than a pair of
    # vectors -- and 1e-4 deg is far below any angle that would signal
    # disagreement.
    assert a.subspace_angle_deg(b) == pytest.approx(0.0, abs=1e-4)


def test_disjoint_samples_of_one_distribution_agree_closely() -> None:
    """What a sufficiency check looks like when sampling is used at all."""
    rng = np.random.default_rng(48)
    latent = rng.normal(size=(8000, 4))
    mixing = rng.normal(size=(4, 20))
    data = latent @ mixing + rng.normal(scale=0.01, size=(8000, 20))

    a = BandReducer(n_components=4).fit(data[:4000])
    b = BandReducer(n_components=4).fit(data[4000:])
    assert a.subspace_angle_deg(b) < 2.0


def test_unrelated_data_gives_a_large_subspace_angle() -> None:
    rng = np.random.default_rng(49)
    a = BandReducer(n_components=3).fit(rng.normal(size=(500, 10)) @ np.eye(10))
    other = np.zeros((500, 10))
    other[:, 7:] = rng.normal(size=(500, 3)) * 50
    b = BandReducer(n_components=3).fit(other + rng.normal(scale=1e-3, size=(500, 10)))
    assert a.subspace_angle_deg(b) > 10.0


def test_subspace_angle_refuses_mismatched_widths() -> None:
    rng = np.random.default_rng(50)
    data = rng.normal(size=(200, 10))
    with pytest.raises(IncompatibleInputError, match="Cannot compare"):
        BandReducer(n_components=3).fit(data).subspace_angle_deg(
            BandReducer(n_components=5).fit(data)
        )


# --------------------------------------------------------------------------
# Whitening -- the autoencoder's input decision
# --------------------------------------------------------------------------


def test_whitening_equalises_component_variance() -> None:
    """Unwhitened, component 1 can dwarf component k and dominate an MSE loss."""
    rng = np.random.default_rng(51)
    latent = rng.normal(size=(3000, 5)) * np.array([100.0, 30.0, 5.0, 1.0, 0.2])
    data = latent @ rng.normal(size=(5, 15))

    reducer = BandReducer(n_components=5).fit(data)
    raw = reducer.transform(data)
    white = reducer.transform(data, whiten=True)

    raw_spread = raw.var(axis=0)
    assert raw_spread.max() / raw_spread.min() > 100

    white_spread = white.var(axis=0)
    assert white_spread == pytest.approx(np.ones(5), rel=0.05)


def test_whitening_preserves_direction_only_rescaling() -> None:
    rng = np.random.default_rng(52)
    data = rng.normal(size=(500, 8)) * np.array([9, 7, 5, 3, 2, 1, 1, 1])
    reducer = BandReducer(n_components=4).fit(data)

    raw = reducer.transform(data)
    white = reducer.transform(data, whiten=True)
    ratio = raw[:, 0] / white[:, 0]

    assert np.allclose(ratio, ratio[0], rtol=1e-5)


def test_whitening_is_off_by_default() -> None:
    rng = np.random.default_rng(53)
    data = rng.normal(size=(300, 9)) * np.arange(1, 10)
    reducer = BandReducer(n_components=4).fit(data)
    assert reducer.transform(data) == pytest.approx(reducer.transform(data, whiten=False))
