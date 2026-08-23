"""Reading hyperspectral scenes into a common container (FR-9.1).

Four on-disk shapes are in play and none of them agree:

* ``.mat`` for ABU (cube and mask in one file, keys ``data``/``map``) and
  Salinas (cube and class map in separate files);
* ``.npy`` for HAD100's preprocessed 64x64 crops and HyperAD's arrays, with
  ground truth alongside as binary PNG;
* ENVI ``.dat``/``.hdr`` pairs for HAD100's raw scenes;
* directories of the above, for corpora rather than single scenes.

Everything downstream sees :class:`HSICube` regardless. Nothing in this module
assumes a band count -- ABU alone spans seven of them (§11 edge case 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DatasetConfig
from ..errors import (
    DataUnavailableError,
    DegenerateInputError,
    IncompatibleInputError,
    MissingGroundTruthError,
)

__all__ = ["HSICube", "load_cube", "load_corpus", "list_corpus"]


@dataclass
class HSICube:
    """One hyperspectral scene plus whatever is known about it (PRD §9).

    Attributes:
        data: ``(H, W, B)``. Kept in the file's own dtype until preprocessing
            casts it, so a memory-mapped array stays memory-mapped.
        wavelengths: ``(B,)`` in nm, or ``None`` when the source ships no
            wavelength table. Water-absorption masking needs this; without it
            that step is skipped and the run says so.
        good_bands: ``(B,)`` bool. All-true until preprocessing narrows it.
        ground_truth: ``(H, W)`` bool, or ``None`` for an unlabelled scene.
        name: Scene identifier, used in run ids and report rows.
        sensor: Instrument name where known.
        source_path: Where it came from, for traceability.
        labels: Provenance flags that must reach the report (PRD §7.3).
    """

    data: np.ndarray
    wavelengths: np.ndarray | None = None
    good_bands: np.ndarray | None = None
    ground_truth: np.ndarray | None = None
    name: str = ""
    sensor: str | None = None
    source_path: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise DegenerateInputError(
                f"{self.name or 'cube'}: expected a 3-D (H, W, B) cube, got shape "
                f"{self.data.shape}. A 2-D array is a single band or a mask, not a cube."
            )
        if self.good_bands is None:
            self.good_bands = np.ones(self.n_bands, dtype=bool)

        if self.good_bands.shape != (self.n_bands,):
            raise IncompatibleInputError(
                f"{self.name or 'cube'}: good_bands has {self.good_bands.shape[0]} "
                f"entries for a {self.n_bands}-band cube."
            )
        if self.wavelengths is not None and self.wavelengths.shape != (self.n_bands,):
            raise IncompatibleInputError(
                f"{self.name or 'cube'}: {self.wavelengths.shape[0]} wavelengths for a "
                f"{self.n_bands}-band cube. A mismatched wavelength table would put "
                "water-absorption masking on the wrong bands."
            )
        if self.ground_truth is not None and self.ground_truth.shape != self.shape[:2]:
            raise IncompatibleInputError(
                f"{self.name or 'cube'}: ground truth is {self.ground_truth.shape} but "
                f"the cube is {self.shape[:2]}. These must share a grid."
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def n_bands(self) -> int:
        return int(self.data.shape[2])

    @property
    def n_pixels(self) -> int:
        return int(self.data.shape[0] * self.data.shape[1])

    @property
    def has_labels(self) -> bool:
        return self.ground_truth is not None

    def require_ground_truth(self) -> np.ndarray:
        """Ground truth, or an explanation of why metrics cannot be computed.

        Raises:
            MissingGroundTruthError: the scene is unlabelled. Callers that can
                still produce an anomaly map should catch this narrowly rather
                than checking ``has_labels`` and inventing a mask.
        """
        if self.ground_truth is None:
            raise MissingGroundTruthError(
                f"Scene {self.name or self.source_path!r} carries no ground truth, so "
                "detection metrics cannot be computed for it. The anomaly map and "
                "figures are still valid; only the metrics table is unavailable."
            )
        return self.ground_truth

    def anomaly_fraction(self) -> float:
        """Share of labelled anomaly pixels, for stratified reporting."""
        gt = self.require_ground_truth()
        return float(np.count_nonzero(gt) / gt.size)


# --------------------------------------------------------------------------
# Format readers
# --------------------------------------------------------------------------


def _require_file(path: Path, what: str, remedy: str) -> Path:
    if not path.exists():
        raise DataUnavailableError(
            reason=f"{what} not found",
            path=path,
            remedy=remedy,
        )
    return path


_DOWNLOAD_HINT = "Run: python scripts/download_data.py --verify"


def _read_mat(path: Path, key: str | None) -> np.ndarray:
    import scipy.io

    _require_file(path, "Scene file", _DOWNLOAD_HINT)
    try:
        contents = scipy.io.loadmat(str(path))
    except Exception as exc:  # scipy raises a wide variety here
        raise DataUnavailableError(
            reason=f"Could not read the .mat file -- {exc}",
            path=path,
            remedy="The file may be truncated. Re-run scripts/download_data.py.",
        ) from exc

    variables = {k: v for k, v in contents.items() if not k.startswith("__")}
    if key is None:
        arrays = {k: v for k, v in variables.items() if getattr(v, "ndim", 0) >= 2}
        if len(arrays) != 1:
            raise DataUnavailableError(
                reason=(
                    f"No data_key given and the file holds {len(arrays)} candidate "
                    f"arrays ({sorted(arrays)}), so the right one is ambiguous"
                ),
                path=path,
                remedy="Set dataset.data_key in the config to the variable you want.",
            )
        return next(iter(arrays.values()))

    if key not in variables:
        raise DataUnavailableError(
            reason=f"Variable {key!r} is not in the file; it holds {sorted(variables)}",
            path=path,
            remedy="Correct dataset.data_key (or ground_truth_key) in the config.",
        )
    return variables[key]


def _read_npy(path: Path, mmap: bool) -> np.ndarray:
    _require_file(path, "Scene file", _DOWNLOAD_HINT)
    try:
        return np.load(path, mmap_mode="r" if mmap else None)
    except Exception as exc:
        raise DataUnavailableError(
            reason=f"Could not read the .npy file -- {exc}",
            path=path,
            remedy="The file may be truncated. Re-run scripts/download_data.py.",
        ) from exc


def _read_envi(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read an ENVI ``.dat``/``.hdr`` pair, returning the cube and wavelengths."""
    import spectral

    hdr = path.with_suffix(".hdr")
    _require_file(path, "ENVI data file", _DOWNLOAD_HINT)
    _require_file(hdr, "ENVI header", "The .hdr must sit beside the .dat with the same stem.")

    img = spectral.envi.open(str(hdr), str(path))
    cube = np.asarray(img.load())

    wavelengths = None
    raw = img.metadata.get("wavelength")
    if raw:
        try:
            wavelengths = np.asarray([float(w) for w in raw], dtype=np.float64)
            # ENVI headers record micrometres about as often as nanometres.
            if wavelengths.max() < 100:
                wavelengths *= 1000.0
        except (TypeError, ValueError):
            wavelengths = None
    return cube, wavelengths


def _read_mask(path: Path, key: str | None) -> np.ndarray:
    """Read a ground-truth mask from ``.mat``, ``.npy``, or an image file."""
    _require_file(path, "Ground-truth file", _DOWNLOAD_HINT)
    suffix = path.suffix.lower()

    if suffix == ".mat":
        mask = _read_mat(path, key)
    elif suffix == ".npy":
        mask = np.load(path)
    else:
        from PIL import Image

        # HAD100 writes masks as RGBA PNG via matplotlib; any channel carries
        # the same binary content, so the first one is taken.
        mask = np.asarray(Image.open(path))
        if mask.ndim == 3:
            mask = mask[..., 0]
    return np.asarray(mask)


def _binarise(mask: np.ndarray, convention: str, anomaly_class_ids: list[int]) -> np.ndarray:
    """Reduce a raw label array to a boolean anomaly mask under a stated convention."""
    values = np.unique(mask)

    if convention == "binary_mask":
        if values.size > 2:
            raise DegenerateInputError(
                f"label_convention is 'binary_mask' but the mask holds {values.size} "
                f"distinct values {values[:8].tolist()}. This is a class map; pick a "
                "convention that says how it becomes binary."
            )
        return mask > 0

    if convention == "rare_classes":
        if not anomaly_class_ids:
            raise DegenerateInputError(
                "label_convention is 'rare_classes' but anomaly_class_ids is empty, so "
                "there is no stated rule for which classes count as anomalies. Refusing "
                "to invent one."
            )
        return np.isin(mask, anomaly_class_ids)

    raise DegenerateInputError(f"Cannot binarise labels under convention {convention!r}.")


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def load_cube(cfg: DatasetConfig, path: str | Path | None = None) -> HSICube:
    """Load one scene described by a :class:`DatasetConfig`.

    Args:
        cfg: Dataset description from the run config.
        path: Override for ``cfg.path``, used when sweeping a directory of
            scenes that share one config.

    Returns:
        An :class:`HSICube` with labels resolved per ``cfg.label_convention``.
    """
    scene_path = Path(path) if path is not None else Path(cfg.path)
    wavelengths: np.ndarray | None = None
    labels: dict[str, str] = {"file_format": cfg.file_format}

    if cfg.file_format == "mat":
        cube = _read_mat(scene_path, cfg.data_key)
    elif cfg.file_format == "npy":
        cube = _read_npy(scene_path, cfg.mmap)
        labels["mmap"] = str(cfg.mmap).lower()
    elif cfg.file_format == "envi":
        cube, wavelengths = _read_envi(scene_path)
    else:
        raise DegenerateInputError(
            f"file_format {cfg.file_format!r} is not a hyperspectral scene format."
        )

    # np.memmap is an ndarray subclass, and np.asarray() would strip it back to
    # a plain view -- still lazily paged, but no longer identifiable as mapped.
    # Only convert what is not already an ndarray.
    if not isinstance(cube, np.ndarray):
        cube = np.asarray(cube)
    if cube.ndim != 3:
        raise DegenerateInputError(
            f"{scene_path.name}: expected an (H, W, B) cube, got shape {cube.shape}."
        )

    if cfg.wavelengths_nm_path:
        wl_path = Path(cfg.wavelengths_nm_path)
        _require_file(wl_path, "Wavelength table", "Point wavelengths_nm_path at a real file.")
        wavelengths = np.loadtxt(wl_path, dtype=np.float64)

    ground_truth = _resolve_ground_truth(cfg, scene_path, cube.shape[:2], labels)

    return HSICube(
        data=cube,
        wavelengths=wavelengths,
        ground_truth=ground_truth,
        name=scene_path.stem or cfg.name,
        sensor=cfg.sensor,
        source_path=str(scene_path),
        labels=labels,
    )


def _resolve_ground_truth(
    cfg: DatasetConfig,
    scene_path: Path,
    spatial_shape: tuple[int, int],
    labels: dict[str, str],
) -> np.ndarray | None:
    """Apply ``cfg.label_convention`` to produce a boolean mask, or ``None``."""
    convention = cfg.label_convention
    labels["label_convention"] = convention

    if convention == "none":
        return None

    if convention == "zero_anomaly":
        # PRD v2 change C5: the scene is declared anomaly-free, so the mask is
        # all-background by construction rather than derived from any file.
        # Every detection at the chosen threshold is a false positive, which is
        # exactly what §11 edge case 3 measures.
        labels["ground_truth_source"] = "declared anomaly-free (no file read)"
        return np.zeros(spatial_shape, dtype=bool)

    if not cfg.ground_truth_path:
        return None

    gt_path = Path(cfg.ground_truth_path)
    if gt_path.is_dir():
        candidates = sorted(gt_path.glob(f"{scene_path.stem}.*"))
        if not candidates:
            raise DataUnavailableError(
                reason=f"No ground-truth file matching scene {scene_path.stem!r}",
                path=gt_path,
                remedy="Check that mask filenames match scene filenames stem-for-stem.",
            )
        gt_path = candidates[0]

    raw = _read_mask(gt_path, cfg.ground_truth_key)
    labels["ground_truth_source"] = str(gt_path)
    return _binarise(raw, convention, cfg.anomaly_class_ids)


def list_corpus(root: str | Path, pattern: str = "*.npy") -> list[Path]:
    """Scene paths under ``root``, sorted for reproducible ordering.

    Raises:
        DataUnavailableError: the directory is missing or holds no matches.
    """
    root = Path(root)
    if not root.is_dir():
        raise DataUnavailableError(
            reason="Corpus directory not found",
            path=root,
            remedy=(
                "For HAD100, extract the archive and run its main.py: "
                "cd data/hyperspectral/had100/HAD100 && python main.py"
            ),
        )
    paths = sorted(root.glob(pattern))
    if not paths:
        raise DataUnavailableError(
            reason=f"Directory holds no files matching {pattern!r}",
            path=root,
            remedy="Check the path, or re-run scripts/download_data.py --verify.",
        )
    return paths


def load_corpus(
    cfg: DatasetConfig,
    root: str | Path,
    pattern: str = "*.npy",
    limit: int | None = None,
) -> list[HSICube]:
    """Load every scene under ``root``.

    Memory note: with ``cfg.mmap`` true and ``npy`` input, the returned cubes
    hold memory maps rather than resident arrays, so the 1,040-crop HAD100
    training corpus costs kilobytes of RAM instead of ~4.7 GB. Anything that
    materialises them (``np.asarray``, arithmetic) reads from disk on demand.
    """
    paths = list_corpus(root, pattern)
    if limit is not None:
        paths = paths[:limit]
    return [load_cube(cfg, p) for p in paths]
