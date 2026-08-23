"""Scene loading and the HSICube contract (FR-9.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.config import DatasetConfig
from pipeline.errors import (
    DataUnavailableError,
    DegenerateInputError,
    IncompatibleInputError,
    MissingGroundTruthError,
)
from pipeline.hyperspectral.loader import HSICube, list_corpus, load_cube

REPO = Path(__file__).resolve().parents[2]
ABU = REPO / "data/hyperspectral/abu"
HAD100 = REPO / "data/hyperspectral/had100/HAD100/HAD100Dataset"


def make_cube(h=4, w=5, b=6, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(h, w, b)).astype(np.float32)


# --------------------------------------------------------------------------
# HSICube invariants
# --------------------------------------------------------------------------


def test_good_bands_defaults_to_all_true() -> None:
    cube = HSICube(data=make_cube())
    assert cube.good_bands.shape == (6,)
    assert cube.good_bands.all()


def test_shape_helpers() -> None:
    cube = HSICube(data=make_cube(4, 5, 6))
    assert cube.shape == (4, 5, 6)
    assert cube.n_bands == 6
    assert cube.n_pixels == 20


def test_two_dimensional_data_is_refused() -> None:
    with pytest.raises(DegenerateInputError, match="3-D"):
        HSICube(data=np.zeros((10, 10)))


def test_mismatched_wavelength_table_is_refused() -> None:
    """A wavelength table off by one puts water masking on the wrong bands."""
    with pytest.raises(IncompatibleInputError, match="wavelengths"):
        HSICube(data=make_cube(b=6), wavelengths=np.arange(5, dtype=float))


def test_mismatched_ground_truth_shape_is_refused() -> None:
    with pytest.raises(IncompatibleInputError, match="share a grid"):
        HSICube(data=make_cube(4, 5, 6), ground_truth=np.zeros((5, 4), dtype=bool))


def test_unlabelled_scene_raises_a_narrow_error() -> None:
    """C6 -- callers can recover by producing a map without metrics."""
    cube = HSICube(data=make_cube(), name="unlabelled")
    assert cube.has_labels is False
    with pytest.raises(MissingGroundTruthError, match="no ground truth"):
        cube.require_ground_truth()


def test_anomaly_fraction() -> None:
    gt = np.zeros((10, 10), dtype=bool)
    gt[:2, :5] = True
    cube = HSICube(data=make_cube(10, 10, 3), ground_truth=gt)
    assert cube.anomaly_fraction() == pytest.approx(0.10)


# --------------------------------------------------------------------------
# Label conventions
# --------------------------------------------------------------------------


def _cfg(**kw) -> DatasetConfig:
    base = dict(name="t", kind="hyperspectral", path="unused")
    base.update(kw)
    return DatasetConfig(**base)


def test_zero_anomaly_convention_reads_no_file(tmp_path: Path) -> None:
    """C5 -- Salinas declares itself anomaly-free rather than deriving a mask."""
    scene = tmp_path / "scene.npy"
    np.save(scene, make_cube(8, 9, 4))

    cfg = _cfg(path=str(scene), file_format="npy", label_convention="zero_anomaly",
               ground_truth_path=None)
    cube = load_cube(cfg)

    assert cube.ground_truth is not None
    assert cube.ground_truth.shape == (8, 9)
    assert not cube.ground_truth.any()
    assert "no file read" in cube.labels["ground_truth_source"]


def test_rare_classes_without_ids_is_refused(tmp_path: Path) -> None:
    """Refusing to invent a rule is the point of C5."""
    import scipy.io

    scene = tmp_path / "s.mat"
    gt = tmp_path / "gt.mat"
    scipy.io.savemat(scene, {"data": make_cube(6, 6, 4)})
    scipy.io.savemat(gt, {"map": np.arange(36).reshape(6, 6) % 5})

    cfg = _cfg(path=str(scene), data_key="data", ground_truth_path=str(gt),
               ground_truth_key="map", label_convention="rare_classes",
               anomaly_class_ids=[])
    with pytest.raises(DegenerateInputError, match="no stated rule"):
        load_cube(cfg)


def test_class_map_under_binary_convention_is_refused(tmp_path: Path) -> None:
    import scipy.io

    scene = tmp_path / "s.mat"
    gt = tmp_path / "gt.mat"
    scipy.io.savemat(scene, {"data": make_cube(6, 6, 4)})
    scipy.io.savemat(gt, {"map": np.arange(36).reshape(6, 6) % 7})

    cfg = _cfg(path=str(scene), data_key="data", ground_truth_path=str(gt),
               ground_truth_key="map", label_convention="binary_mask")
    with pytest.raises(DegenerateInputError, match="class map"):
        load_cube(cfg)


def test_rare_classes_selects_listed_ids(tmp_path: Path) -> None:
    import scipy.io

    scene = tmp_path / "s.mat"
    gt = tmp_path / "gt.mat"
    labels = np.array([[0, 1, 2], [3, 4, 5]])
    scipy.io.savemat(scene, {"data": make_cube(2, 3, 4)})
    scipy.io.savemat(gt, {"map": labels})

    cfg = _cfg(path=str(scene), data_key="data", ground_truth_path=str(gt),
               ground_truth_key="map", label_convention="rare_classes",
               anomaly_class_ids=[2, 5])
    cube = load_cube(cfg)
    assert cube.ground_truth.sum() == 2
    assert cube.ground_truth[0, 2] and cube.ground_truth[1, 2]


# --------------------------------------------------------------------------
# Actionable failures -- §11 edge case 14
# --------------------------------------------------------------------------


def test_missing_scene_names_path_and_remedy() -> None:
    cfg = _cfg(path="data/hyperspectral/nope/missing.mat", data_key="data")
    with pytest.raises(DataUnavailableError) as exc:
        load_cube(cfg)
    assert exc.value.path.name == "missing.mat"
    assert "download_data" in exc.value.remedy


def test_wrong_mat_key_lists_available_variables(tmp_path: Path) -> None:
    import scipy.io

    scene = tmp_path / "s.mat"
    scipy.io.savemat(scene, {"data": make_cube(4, 4, 3)})

    cfg = _cfg(path=str(scene), data_key="cube")
    with pytest.raises(DataUnavailableError) as exc:
        load_cube(cfg)
    assert "data" in str(exc.value)


def test_missing_corpus_directory_points_at_main_py() -> None:
    with pytest.raises(DataUnavailableError) as exc:
        list_corpus("data/hyperspectral/had100/nowhere")
    assert "main.py" in exc.value.remedy


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------


@pytest.mark.needs_data
@pytest.mark.skipif(not ABU.is_dir(), reason="ABU not downloaded")
def test_abu_scene_loads_with_binary_mask() -> None:
    cfg = _cfg(name="abu", path=str(ABU / "abu-airport-1.mat"), data_key="data",
               ground_truth_path=str(ABU / "abu-airport-1.mat"), ground_truth_key="map",
               label_convention="binary_mask", sensor="aviris")
    cube = load_cube(cfg)

    assert cube.shape == (100, 100, 205)
    assert cube.ground_truth.dtype == bool
    assert cube.ground_truth.sum() == 144
    assert cube.anomaly_fraction() == pytest.approx(0.0144)


@pytest.mark.needs_data
@pytest.mark.skipif(not ABU.is_dir(), reason="ABU not downloaded")
def test_every_abu_scene_loads_whatever_its_band_count() -> None:
    """§11 edge case 5, on real data: seven band counts, three dtypes."""
    band_counts = set()
    for path in sorted(ABU.glob("*.mat")):
        cfg = _cfg(name=path.stem, path=str(path), data_key="data",
                   ground_truth_path=str(path), ground_truth_key="map")
        cube = load_cube(cfg)
        band_counts.add(cube.n_bands)
        assert cube.ground_truth.shape == cube.shape[:2]

    assert len(band_counts) >= 6, f"expected varied band counts, got {sorted(band_counts)}"


@pytest.mark.needs_data
@pytest.mark.skipif(not HAD100.is_dir(), reason="HAD100 not preprocessed")
def test_had100_test_image_loads_with_png_mask() -> None:
    scene = sorted((HAD100 / "test/aviris_ng").glob("*.npy"))[0]
    cfg = _cfg(name="had100", path=str(scene), file_format="npy", sensor="aviris_ng",
               ground_truth_path=str(HAD100 / "ground_truth/aviris_ng"),
               label_convention="binary_mask")
    cube = load_cube(cfg)

    assert cube.shape == (64, 64, 276)
    assert cube.ground_truth.dtype == bool
    assert cube.ground_truth.any(), "test images should contain anomalies"


@pytest.mark.needs_data
@pytest.mark.skipif(not HAD100.is_dir(), reason="HAD100 not preprocessed")
def test_mmap_keeps_the_corpus_off_the_heap() -> None:
    """The 1,040-crop corpus is ~4.7 GB if loaded eagerly."""
    scene = sorted((HAD100 / "train/aviris_ng").glob("*.npy"))[0]
    cfg = _cfg(name="c", path=str(scene), file_format="npy", mmap=True,
               label_convention="none")
    cube = load_cube(cfg)

    assert isinstance(cube.data, np.memmap)
    assert cube.labels["mmap"] == "true"


@pytest.mark.needs_data
@pytest.mark.skipif(not HAD100.is_dir(), reason="HAD100 not preprocessed")
def test_corpus_listing_is_the_expected_size() -> None:
    assert len(list_corpus(HAD100 / "train/aviris_ng")) == 1040
    assert len(list_corpus(HAD100 / "test/aviris_ng")) == 100
