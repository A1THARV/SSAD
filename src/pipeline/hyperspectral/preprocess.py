"""Band masking, outlier clipping and PCA reduction (FR-9.1, FR-9.3).

Order matters here and is fixed: mask bad bands, then clip saturated pixels,
then fit PCA. Clipping after PCA would be pointless -- an extreme value has
already skewed the components by then (§11 edge case 6) -- and fitting PCA over
water-absorption bands wastes components on noise (§11 edge case 1).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import PreprocessConfig
from ..errors import DegenerateInputError, IncompatibleInputError, ModelNotFittedError
from .band_presets import good_band_indices
from .loader import HSICube

__all__ = [
    "mask_and_clip",
    "compute_good_bands",
    "clip_outliers",
    "BandReducer",
    "preprocess_cube",
    "fit_corpus_reducer",
]


def compute_good_bands(cube: HSICube, cfg: PreprocessConfig) -> np.ndarray:
    """Boolean mask over bands, keeping those worth analysing.

    Three criteria, applied in order of authority:

    1. A published preset, when one is configured. This overrides the rest --
       comparability against a benchmark's baselines requires its exact band
       set, not a defensible approximation (change C3).
    2. Water-absorption intervals, when wavelengths are known.
    3. Flat bands (std below ``min_band_std``), which catches dead or
       deliberately zeroed bands whether or not wavelengths are available.

    Returns:
        ``(B,)`` bool. Never all-false; a mask that would keep nothing is an
        error rather than a silently empty cube.
    """
    n_bands = cube.n_bands

    if cfg.good_bands_preset != "none":
        keep = np.zeros(n_bands, dtype=bool)
        keep[good_band_indices(cfg.good_bands_preset, n_bands)] = True
        return keep

    keep = np.ones(n_bands, dtype=bool)

    if cube.wavelengths is not None:
        for lo, hi in cfg.water_absorption_bands_nm:
            keep &= ~((cube.wavelengths >= lo) & (cube.wavelengths <= hi))

    # Flat bands carry no information and make the covariance singular, so they
    # go regardless of whether a wavelength table was available.
    data = np.asarray(cube.data)
    band_std = data.reshape(-1, n_bands).std(axis=0)
    keep &= band_std > cfg.min_band_std

    if cfg.snr_threshold is not None:
        band_mean = np.abs(data.reshape(-1, n_bands).mean(axis=0))
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.where(band_std > 0, band_mean / band_std, 0.0)
        keep &= snr >= cfg.snr_threshold

    if not keep.any():
        raise DegenerateInputError(
            f"{cube.name}: band masking would remove all {n_bands} bands. Check the "
            "water-absorption intervals and snr_threshold against this sensor."
        )
    return keep


def clip_outliers(data: np.ndarray, percentile: float) -> np.ndarray:
    """Clip extreme values so saturated or dead pixels cannot dominate PCA.

    §11 edge case 6. Clipping is per band rather than global: a scene's bands
    sit at very different radiance levels, and one global ceiling would flatten
    the bright bands while leaving the dark ones untouched.

    Args:
        data: ``(H, W, B)`` or ``(N, B)``.
        percentile: Upper clip point; the lower clip is its mirror.
    """
    if not 50.0 <= percentile <= 100.0:
        raise DegenerateInputError(
            f"clip_percentile must be within [50, 100], got {percentile}."
        )
    if percentile == 100.0:
        return data

    flat = data.reshape(-1, data.shape[-1])
    hi = np.percentile(flat, percentile, axis=0)
    lo = np.percentile(flat, 100.0 - percentile, axis=0)
    return np.clip(data, lo, hi)


@dataclass
class BandReducer:
    """PCA fitted once, applied many times (FR-9.3).

    The fit/transform split is the point. Refitting between training and
    inference would put the two on different axes, so :meth:`transform` refuses
    to run before a fit and refuses cubes whose band count differs.

    Both fit paths -- :meth:`fit` for an in-memory array and
    :meth:`fit_streaming` for a corpus too large to materialise -- go through
    the same covariance eigendecomposition, so the two cannot silently diverge.
    """

    n_components: int | None = None
    variance: float = 0.99
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    explained_variance_: np.ndarray | None = None
    explained_variance_ratio_: np.ndarray | None = None
    n_samples_seen_: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.components_ is not None

    @property
    def n_output_components(self) -> int:
        self._require_fitted()
        return int(self.components_.shape[0])

    def _require_fitted(self) -> None:
        if self.components_ is None:
            raise ModelNotFittedError(
                "BandReducer used before fitting. Fit once on background-representative "
                "data and reuse, so training and inference share axes."
            )

    # -- fitting ----------------------------------------------------------

    def fit(self, data: np.ndarray, seed: int = 0) -> "BandReducer":
        """Fit on ``(H, W, B)`` or ``(N, B)`` background-representative data."""
        flat = np.asarray(data, dtype=np.float64).reshape(-1, data.shape[-1])
        n_samples, n_bands = flat.shape
        if n_samples < 2:
            raise DegenerateInputError(f"PCA needs at least 2 samples, got {n_samples}.")

        mean = flat.mean(axis=0)
        centred = flat - mean
        cov = (centred.T @ centred) / (n_samples - 1)
        return self._finalise(mean, cov, n_samples, n_bands)

    def fit_streaming(self, blocks: Iterable[np.ndarray], seed: int = 0) -> "BandReducer":
        """Fit from an iterable of blocks, materialising only one at a time.

        PCA needs the ``B x B`` covariance, not the pixel matrix. At 276 bands
        that is ~0.6 MB resident, so a corpus of 1,040 memory-mapped crops --
        roughly 4.7 GB if loaded eagerly -- can be fitted exactly rather than
        sampled. This removes any sampling fraction to justify.

        Two passes, deliberately. The one-pass identity
        ``Cov = E[x x^T] - mu mu^T`` loses precision to cancellation when the
        mean is large relative to the variance, which is exactly where
        reflectance data sits.

        Args:
            blocks: Any iterable of ``(H, W, B)`` or ``(N, B)`` arrays. It must
                be **re-iterable**, since it is consumed twice; a generator will
                be rejected rather than silently yielding a mean-only fit.
            seed: Unused; accepted so both fit paths share a signature.
        """
        if isinstance(blocks, Iterator):
            raise DegenerateInputError(
                "fit_streaming consumes its input twice (once for the mean, once for "
                "the covariance), so a one-shot iterator would leave the second pass "
                "empty. Pass a re-iterable sequence, or a callable that returns a fresh "
                "iterator."
            )

        # Pass 1 -- mean.
        total = None
        n_samples = 0
        n_bands = None
        for block in blocks:
            flat = np.asarray(block, dtype=np.float64).reshape(-1, np.shape(block)[-1])
            if n_bands is None:
                n_bands = flat.shape[1]
                total = np.zeros(n_bands, dtype=np.float64)
            elif flat.shape[1] != n_bands:
                raise IncompatibleInputError(
                    f"Corpus mixes band counts: saw {n_bands}, then {flat.shape[1]}. "
                    "One PCA basis cannot span two sensors without resampling."
                )
            total += flat.sum(axis=0)
            n_samples += flat.shape[0]

        if n_bands is None or n_samples == 0:
            raise DegenerateInputError("Cannot fit PCA on an empty corpus.")
        if n_samples < 2:
            raise DegenerateInputError(f"PCA needs at least 2 samples, got {n_samples}.")
        mean = total / n_samples

        # Pass 2 -- covariance about that mean.
        cov = np.zeros((n_bands, n_bands), dtype=np.float64)
        for block in blocks:
            flat = np.asarray(block, dtype=np.float64).reshape(-1, n_bands)
            centred = flat - mean
            cov += centred.T @ centred
        cov /= n_samples - 1

        return self._finalise(mean, cov, n_samples, n_bands)

    def _finalise(
        self, mean: np.ndarray, cov: np.ndarray, n_samples: int, n_bands: int
    ) -> "BandReducer":
        """Eigendecompose a covariance into components, shared by both fit paths."""
        ceiling = min(n_samples, n_bands)
        if self.n_components is not None:
            if self.n_components < 1:
                raise DegenerateInputError(
                    f"pca_n_components must be >= 1, got {self.n_components}."
                )
            if self.n_components > ceiling:
                raise DegenerateInputError(
                    f"Requested {self.n_components} PCA components but the data supports "
                    f"at most {ceiling} ({n_samples} samples, {n_bands} bands). Lower "
                    "pca_n_components."
                )

        # eigh returns ascending eigenvalues for a symmetric matrix; reverse to
        # put the leading direction first.
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Tiny negative eigenvalues are float noise on a PSD matrix; clamping
        # keeps the whitening square root real.
        eigenvalues = np.maximum(eigenvalues, 0.0)

        total_variance = eigenvalues.sum()
        if total_variance <= 0:
            raise DegenerateInputError(
                "Corpus has zero total variance -- every pixel is identical, so there "
                "are no principal directions to find."
            )
        ratio = eigenvalues / total_variance

        if self.n_components is not None:
            k = int(self.n_components)
        else:
            k = int(np.searchsorted(np.cumsum(ratio), self.variance) + 1)
            k = min(k, ceiling)

        # Sign convention: make the largest-magnitude loading of each component
        # positive, so a refit on identical data reproduces identical output
        # rather than an arbitrarily flipped axis.
        components = eigenvectors[:, :k].T.copy()
        for i in range(k):
            j = int(np.argmax(np.abs(components[i])))
            if components[i, j] < 0:
                components[i] *= -1.0

        self.mean_ = mean
        self.components_ = components
        self.explained_variance_ = eigenvalues[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.n_samples_seen_ = n_samples
        return self

    # -- applying ---------------------------------------------------------

    def transform(self, data: np.ndarray, whiten: bool = False) -> np.ndarray:
        """Project a cube onto the fitted components, preserving spatial shape.

        Args:
            data: ``(..., B)`` array.
            whiten: Divide each component by ``sqrt(lambda)`` so all retained
                components carry unit variance.

                This matters for the autoencoder and not for RX. An MSE-trained
                network on raw PCA scores is dominated by the leading
                components -- component 1 can carry orders of magnitude more
                variance than component 15 -- so it spends capacity there and
                can reconstruct a target almost perfectly while ignoring the
                low-variance directions the target actually differs in. Eq. 1's
                reconstruction error would then be blind to exactly the
                anomalies it is meant to find. RX needs no such step: Eq. 2
                applies ``Sigma^-1``, which whitens implicitly.

                The counter-risk is amplifying noise in trailing components, so
                this is run both ways at M2 rather than assumed.
        """
        self._require_fitted()
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape[-1] != self.mean_.shape[0]:
            raise IncompatibleInputError(
                f"BandReducer was fitted on {self.mean_.shape[0]} bands but received "
                f"{arr.shape[-1]}. Cubes from different sensors cannot share one "
                "projection without resampling to a common grid."
            )
        spatial = arr.shape[:-1]
        flat = arr.reshape(-1, arr.shape[-1]) - self.mean_
        reduced = flat @ self.components_.T

        if whiten:
            scale = np.sqrt(self.explained_variance_)
            if np.any(scale <= 0):
                raise DegenerateInputError(
                    "A retained component has zero variance, so whitening would divide "
                    "by zero. Reduce pca_n_components."
                )
            reduced = reduced / scale

        return reduced.reshape(*spatial, reduced.shape[-1]).astype(np.float32)

    def explained_variance(self) -> float:
        self._require_fitted()
        return float(self.explained_variance_ratio_.sum())

    def subspace_angle_deg(self, other: "BandReducer") -> float:
        """Largest principal angle between this basis and another, in degrees.

        The right way to ask whether two fits agree. Eigen*values* can match
        while eigen*vectors* have rotated, and it is the vectors that define the
        projection -- so this is where an insufficient sample fails first.
        """
        self._require_fitted()
        other._require_fitted()
        if self.components_.shape != other.components_.shape:
            raise IncompatibleInputError(
                f"Cannot compare a {self.components_.shape[0]}-component basis with a "
                f"{other.components_.shape[0]}-component one."
            )
        singular = np.linalg.svd(self.components_ @ other.components_.T, compute_uv=False)
        return float(np.degrees(np.arccos(np.clip(singular.min(), -1.0, 1.0))))


class CorpusBlocks:
    """Re-iterable view over a corpus of ``.npy`` scenes, one at a time.

    Holds paths, not arrays. Each iteration memory-maps a scene, yields it, and
    lets it fall out of scope, so peak residency is one crop rather than the
    whole corpus. Deliberately a class rather than a generator function, because
    :meth:`BandReducer.fit_streaming` needs two passes.
    """

    def __init__(
        self,
        paths: list[Path],
        good_bands: np.ndarray | None = None,
        clip_percentile: float | None = None,
    ) -> None:
        if not paths:
            raise DegenerateInputError("Corpus is empty; nothing to fit PCA on.")
        self.paths = list(paths)
        self.good_bands = good_bands
        self.clip_percentile = clip_percentile

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self):
        for path in self.paths:
            block = np.load(path, mmap_mode="r")
            arr = np.asarray(block, dtype=np.float32)
            if self.good_bands is not None:
                arr = arr[..., self.good_bands]
            if self.clip_percentile is not None:
                arr = clip_outliers(arr, self.clip_percentile)
            yield arr


def fit_corpus_reducer(
    paths: list[Path],
    cfg: PreprocessConfig,
    good_bands: np.ndarray | None = None,
    seed: int = 0,
) -> BandReducer:
    """Fit one PCA basis across a whole corpus, exactly and without sampling.

    This is the autoencoder's basis (PRD v2 change C2). Corpus-fitting is not
    optional there: train and inference must share fixed input axes, or the
    encoder meets a different basis at test time.

    Args:
        paths: Corpus scene files, from :func:`loader.list_corpus`.
        cfg: Supplies the component count and clipping percentile.
        good_bands: Optional band mask applied to every scene.
        seed: Accepted for signature symmetry.
    """
    blocks = CorpusBlocks(paths, good_bands=good_bands, clip_percentile=cfg.clip_percentile)
    reducer = BandReducer(n_components=cfg.pca_n_components, variance=cfg.pca_variance)
    return reducer.fit_streaming(blocks, seed=seed)


def mask_and_clip(cube: HSICube, cfg: PreprocessConfig) -> np.ndarray:
    """Band-mask and clip one cube, the step every scoring path shares.

    Extracted because three callers need it -- the CLI, the ablation runner and
    the train-on-test-scene comparison -- and they must apply *identical*
    preprocessing or their numbers stop being comparable. Three hand-inlined
    copies would agree today and drift silently the first time one is edited.

    It stops before PCA deliberately. The RX and autoencoder branches project
    the same clipped cube onto different bases, so reduction cannot be folded
    in here without forcing one basis on both.

    Returns:
        ``(H, W, B_kept)`` float32. Sets ``cube.good_bands`` as a side effect,
        which is what carries the mask into the run record.
    """
    keep = compute_good_bands(cube, cfg)
    cube.good_bands = keep

    # np.asarray materialises a memory-mapped cube here, and only here.
    masked = np.asarray(cube.data, dtype=np.float32)[:, :, keep]
    if masked.shape[2] == 0:
        raise DegenerateInputError(f"{cube.name}: no bands survived masking.")
    return clip_outliers(masked, cfg.clip_percentile)


def _cache_key(cube: HSICube, cfg: PreprocessConfig) -> str:
    """Identify a reduced cube by its source and the settings that produced it."""
    parts = [
        cube.source_path,
        str(cube.shape),
        cfg.good_bands_preset,
        str(cfg.pca_n_components),
        f"{cfg.pca_variance:.6f}",
        f"{cfg.clip_percentile:.4f}",
        f"{cfg.min_band_std:.3e}",
        str(cfg.water_absorption_bands_nm),
        str(cfg.snr_threshold),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def preprocess_cube(
    cube: HSICube,
    cfg: PreprocessConfig,
    reducer: BandReducer | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, BandReducer]:
    """Mask, clip and reduce one cube.

    Args:
        cube: The scene.
        cfg: Preprocessing settings.
        reducer: A reducer already fitted on a background corpus. Passing one
            is what keeps a whole corpus on shared axes; omitting it fits a new
            reducer on this cube alone.
        seed: PCA random state, for reproducibility.

    Returns:
        ``(reduced, reducer)`` where ``reduced`` is ``(H, W, C)`` float32.
    """
    clipped = mask_and_clip(cube, cfg)

    cached: Path | None = None
    if cfg.cache_reduced and cube.source_path:
        cached = Path(cfg.cache_dir) / f"{cube.name}_{_cache_key(cube, cfg)}.npy"
        if cached.is_file() and reducer is not None and reducer.is_fitted:
            return np.load(cached), reducer

    if reducer is None:
        reducer = BandReducer(
            n_components=cfg.pca_n_components, variance=cfg.pca_variance
        ).fit(clipped, seed=seed)
    elif not reducer.is_fitted:
        reducer.fit(clipped, seed=seed)

    reduced = reducer.transform(clipped)

    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        np.save(cached, reduced)

    return reduced, reducer
