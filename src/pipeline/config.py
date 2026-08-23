"""Typed configuration: YAML -> dataclasses, plus config hashing and seeding.

PRD §5 chose ``pyyaml`` + ``dataclasses`` over Hydra to keep a prototype simple.
Three properties matter here and are all implemented in this module:

* **Strictness** -- unknown keys raise :class:`ConfigError`. A silently ignored
  typo in a config file is a reproducibility hazard (PRD §10).
* **Hashing** -- :func:`config_hash` hashes the *resolved* config (after
  defaults are applied), so a changed default is detected even when no YAML
  file changed. FR-9.11 requires this hash to be embedded in every report.
* **Seeding** -- :func:`set_seeds` covers ``random``, ``numpy`` and ``torch`` so
  that PRD §11 edge case 15 (re-run reproducibility) holds by construction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import types
import typing
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import yaml

from .errors import ConfigError

__all__ = [
    "PreprocessConfig",
    "RXConfig",
    "AutoencoderConfig",
    "ScoreFusionConfig",
    "SAMConfig",
    "DatasetConfig",
    "HyperspectralConfig",
    "ThermalPreprocessConfig",
    "CNNLSTMConfig",
    "PlanckConfig",
    "ThermalConfig",
    "FusionConfig",
    "load_config",
    "from_dict",
    "to_dict",
    "config_hash",
    "set_seeds",
    "resolve_device",
]


# --------------------------------------------------------------------------
# Hyperspectral branch
# --------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    """Band masking and dimensionality reduction (FR-9.1, FR-9.3).

    Attributes:
        water_absorption_bands_nm: Wavelength intervals (nm) masked as
            atmospheric water absorption. PRD FR-9.1 names ~1350-1450 and
            ~1800-1950 nm.
        snr_threshold: Bands whose estimated SNR falls below this are masked.
            ``None`` disables the SNR criterion.
        min_band_std: Bands flatter than this are treated as dead (a
            deliberately zeroed band must be excluded -- FR-9.1 acceptance
            test, §11 edge case 1).
        clip_percentile: Saturated/dead pixel values above this percentile are
            clipped before PCA so they cannot dominate it (§11 edge case 6).
        good_bands_preset: Use a benchmark's own published band selection
            instead of deriving one from wavelengths. This exists for
            comparability, not convenience: HAD100's 17 baselines (change C3)
            were all evaluated after *its* band selection, so scoring against
            their published table requires using the same bands. ``none``
            falls back to the wavelength-driven masking above.
        pca_variance: Variance retained when ``pca_n_components`` is ``None``.
        pca_n_components: Explicit component count; overrides ``pca_variance``.
            On a 4 GB card this is a hard requirement rather than a speed
            tuning knob: a 32-filter 3D conv over a 276-band cube holds on the
            order of 145 MB of activations per sample, so even a small batch
            exceeds VRAM before gradients and optimizer state are counted.
            Reducing to ~30 components drops that roughly tenfold.
        cache_reduced: Persist the PCA-reduced cube to ``cache_dir`` so
            repeated runs skip refitting and re-projecting.
        cache_dir: Where reduced cubes are cached. Gitignored.
    """

    water_absorption_bands_nm: list[list[float]] = field(
        default_factory=lambda: [[1350.0, 1450.0], [1800.0, 1950.0]]
    )
    snr_threshold: float | None = None
    min_band_std: float = 1e-8
    clip_percentile: float = 99.9
    good_bands_preset: Literal[
        "none",
        "had100_aviris",
        "had100_aviris_ng",
        "had100_band50",
        "had100_band100",
        "had100_band200",
    ] = "none"
    pca_variance: float = 0.99
    pca_n_components: int | None = None
    cache_reduced: bool = True
    cache_dir: str = "data/cache"


@dataclass
class RXConfig:
    """Reed-Xiaoli detector settings (Eq. 2, FR-9.2).

    Attributes:
        ridge: Ridge term added to the covariance diagonal before inversion.
            Guards the near-singular covariance of a homogeneous scene
            (§11 edge case 2), but note it also materially affects accuracy at
            high dimensionality -- see :func:`background_statistics`.
        shrinkage: ``ridge`` or ``ledoit_wolf``. The latter is parameter-free
            and preferred when one setting must serve several datasets, because
            the tuned ridge does not transfer between HAD100 and ABU.
        pca_basis: Which PCA axes RX consumes.

            ``per_scene`` (default) fits PCA on the image under test. This is
            consistent with what RX *is* -- an unsupervised per-scene detector
            that estimates background statistics from the scene itself -- and
            it is what produced the M1 baseline benchmarked against HAD100's 19
            published methods, so changing it would invalidate that comparison.

            ``corpus`` reuses the autoencoder's corpus-fitted axes, making the
            two branches share one basis as the paper's Fig. 1 draws it. The
            risk is that axes fitted across 1,040 diverse crops lead on
            *inter*-scene variance (terrain, illumination) rather than the
            intra-scene structure that makes a target stand out.

            Both are run at M2 as a measured ablation rather than assumed. Note
            the autoencoder has no such choice: corpus-fit is mandatory there,
            since train and inference must share fixed input axes.
    """

    ridge: float = 1e-6
    shrinkage: Literal["ridge", "ledoit_wolf"] = "ridge"
    pca_basis: Literal["per_scene", "corpus"] = "per_scene"


@dataclass
class AutoencoderConfig:
    """Spectral-spatial autoencoder (Eq. 1 and Eq. 3, FR-9.4).

    Defaults are deliberately prototype-scale. PRD §10 caps the full
    three-scene hyperspectral run at 20 minutes CPU-only, and the available GPU
    has 4 GB of VRAM, so training is patch-based rather than whole-cube.

    Attributes:
        patch_size: Spatial edge length of a training patch, in pixels.
        latent_dim: Bottleneck width. Fig. 1 annotates the latent vector as
            ``z in R^d, d << K``, so this must stay well below the PCA
            component count -- a latent wider than its input is not a
            bottleneck at all.
        encoder_channels: 3D-conv channel widths; the decoder mirrors these.
            Defaults match Fig. 1's drawn encoder (32 -> 64 -> 128).
        kernel_spatial: Spatial edge of every 3D kernel. Fig. 1 draws 3x3
            throughout.
        kernel_spectral_depths: Spectral depth per encoder layer. Fig. 1 draws
            3x3x7 -> 3x3x5 -> 3x3x3, narrowing as depth increases; the decoder
            mirrors it. Note this constrains PCA width: a first kernel of depth
            7 needs at least 7 components to convolve over.
        lam: Weight of the SAM term in the composite loss (Eq. 3's lambda).
        max_patches: Cap on sampled training patches; bounds runtime and VRAM.
        training_corpus: Directory of anomaly-free scenes to fit the background
            model on (PRD v2 change C2). This is the change that makes Eq. 1's
            premise defensible -- a background model fit on data containing the
            anomalies it must detect cannot support "anomalies reconstruct
            poorly". ``None`` means no separate corpus is configured.
        train_on_test_scene: Restore PRD v1 behaviour, fitting on the scene
            under test. Kept because self-supervised single-scene training is a
            legitimate published variant (Auto-AD, paper reference [8]), so
            running both and reporting the comparison is a result rather than
            an implementation detail.
        whiten_input: Divide each PCA component by ``sqrt(lambda)`` before the
            network sees it, so the MSE term weights all retained components
            equally.

            Without it an MSE-trained autoencoder is dominated by the leading
            components -- component 1 can carry orders of magnitude more
            variance than component 15 -- so it spends capacity there and can
            reconstruct a target almost perfectly while ignoring the
            low-variance directions the target actually differs in. Eq. 1's
            reconstruction error would then be systematically blind to exactly
            the anomalies it exists to find, which is not hypothetical here:
            the features separating a sub-pixel target live in low-variance
            directions almost by construction.

            The counter-risk is amplifying noise in the trailing components, so
            both settings are run at M2 rather than one being assumed. Applies
            only to the autoencoder: RX needs no equivalent, since Eq. 2's
            ``Sigma^-1`` whitens implicitly.
        sam_domain: Where Eq. 3's SAM term is evaluated. ``spectral``
            back-projects both the input and the reconstruction to reflectance
            first; ``latent`` takes the angle on the PCA scores directly.

            ``spectral`` is the default because Eq. 4's justification -- that
            the angle between two reflectance spectra is invariant to
            illumination -- is a statement about the reflectance domain. PCA
            scores are mean-centred, so their origin is the corpus mean
            spectrum, not darkness: an unremarkable background pixel projects
            near zero, where its direction is noise and its angle to anything is
            unstable. ``latent`` is kept as a measurable ablation, not a
            shortcut.
        amp: Mixed precision via ``torch.autocast``. The target card is Ampere,
            so fp16 both halves activation memory and uses tensor cores.
        vram_budget_gb: Usable VRAM assumed when sizing batches. Deliberately
            below the card's nominal 4 GB, because Windows reserves a slice for
            the display; sizing against the nominal figure OOMs late in an epoch.
        gradient_checkpointing: Trades compute for memory. A last resort, only
            once PCA reduction, batch size and AMP are exhausted.
        device: ``auto`` resolves to CUDA when available, else CPU. GPU is a
            speed-up, never a hard requirement (PRD §10).
    """

    enabled: bool = True
    patch_size: int = 7
    latent_dim: int = 8
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_spatial: int = 3
    kernel_spectral_depths: list[int] = field(default_factory=lambda: [7, 5, 3])
    lam: float = 0.1
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_patches: int | None = 20000
    training_corpus: str | None = None
    train_on_test_scene: bool = False
    whiten_input: bool = False
    sam_domain: Literal["spectral", "latent"] = "spectral"
    amp: bool = True
    vram_budget_gb: float = 3.3
    gradient_checkpointing: bool = False
    device: Literal["auto", "cpu", "cuda"] = "auto"


@dataclass
class ScoreFusionConfig:
    """RX + reconstruction-error fusion and thresholding (FR-9.5).

    Attributes:
        mode: ``weighted`` uses ``alpha * rx + (1 - alpha) * recon``;
            ``max`` takes the per-pixel maximum of the normalized scores.
        threshold_method: ``percentile``, ``otsu``, ``mean_std``, or
            ``f1_optimal``. ``mean_std`` (``T = mu + gamma*sigma``) is the rule
            drawn in the paper's Fig. 1 stage 6.
        threshold_percentile: Percentile used when ``threshold_method`` is
            ``percentile``. PRD FR-9.5 defaults to the 99th.
        threshold_gamma: Sigma multiplier for ``mean_std``.
        validation_split: Fraction of *labelled* images held out to fit an
            ``f1_optimal`` threshold. The paper (§V) reports precision, recall
            and F1 at the threshold maximising F1, which is only an honest
            number when that threshold was chosen on data excluded from the
            reported set -- otherwise it is fitted to the test set and reads
            far better than it is.
        validation_seed: Seed for the held-out split, so the same images are
            held out on every re-run (§11 edge case 15).
    """

    mode: Literal["weighted", "max"] = "weighted"
    alpha: float = 0.5
    threshold_method: Literal[
        "percentile", "otsu", "mean_std", "f1_optimal"
    ] = "percentile"
    threshold_percentile: float = 99.0
    threshold_gamma: float = 3.0
    validation_split: float = 0.2
    validation_seed: int = 12345


@dataclass
class SAMConfig:
    """Spectral angle mapper material matching (Eq. 4, FR-9.6).

    Attributes:
        top_k: Number of candidate materials returned. FR-9.6 forbids
            returning a single hard label with no confidence indication.
        low_confidence_angle_rad: Above this best-match angle the result is
            flagged low confidence rather than asserted (§11 edge case 7).
        library_path: Curated USGS/ECOSTRESS subset built by
            ``scripts/download_data.py`` (PRD §7.2).
    """

    top_k: int = 3
    low_confidence_angle_rad: float = 0.25
    library_path: str = "data/speclib/curated_library.npz"


@dataclass
class DatasetConfig:
    """Where a scene lives on disk and how to read its labels.

    Attributes:
        name: Short identifier used in run ids and report tables.
        kind: ``hyperspectral`` or ``thermal``.
        path: Path to the scene file or directory, relative to the repo root.
        file_format: How to read it. ``mat`` for ABU/Salinas, ``envi`` for
            HAD100's raw ``.dat``/``.hdr`` pairs, ``npy`` for HAD100's
            preprocessed crops and HyperAD's arrays, ``image_dir`` for thermal
            frame folders.
        sensor: Instrument name, e.g. ``aviris`` or ``aviris_ng``. HAD100
            selects bands per sensor, and its two sensors survive band
            selection at different widths (162 vs 276), so a cube's sensor is
            load-bearing rather than documentation.
        data_key: Variable name inside a ``.mat`` file holding the cube.
        ground_truth_path / ground_truth_key: Location of the label mask.
        wavelengths_nm_path: Optional per-band wavelength table (nm). Required
            for water-absorption masking; without it that step is skipped and
            the run is labelled accordingly.
        mmap: Read ``.npy`` scenes memory-mapped rather than into RAM. The
            HAD100 training corpus is 1,040 crops at ~4.5 MB each -- close to
            4.7 GB resident if loaded eagerly, which thrashes a 16 GB laptop.
            Only applies to ``npy``; other formats have no mmap path.
        label_convention: How labels become a binary anomaly mask.
            ``binary_mask`` uses the shipped mask as-is. ``zero_anomaly``
            declares the scene contains no anomalies and builds an
            all-background mask, which is the §11 edge case 3 fixture (PRD v2
            change C5). ``rare_classes`` derives a mask from listed class ids;
            available, but not used by any shipped config -- change C5 rejects
            manufacturing anomaly labels out of a land-cover map. ``none``
            means the scene is unlabelled, so metrics raise
            :class:`MissingGroundTruthError` while the anomaly map still runs.
        anomaly_class_ids: Class ids treated as anomalies under a
            ``rare_classes`` convention.
    """

    name: str
    kind: Literal["hyperspectral", "thermal"]
    path: str
    file_format: Literal["mat", "envi", "npy", "image_dir"] = "mat"
    sensor: str | None = None
    data_key: str | None = None
    ground_truth_path: str | None = None
    ground_truth_key: str | None = None
    wavelengths_nm_path: str | None = None
    mmap: bool = True
    label_convention: Literal[
        "binary_mask", "zero_anomaly", "rare_classes", "none"
    ] = "binary_mask"
    anomaly_class_ids: list[int] = field(default_factory=list)
    notes: str = ""


@dataclass
class HyperspectralConfig:
    """Top-level config for ``run_hyperspectral`` (FR-9.12)."""

    dataset: DatasetConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    rx: RXConfig = field(default_factory=RXConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    score_fusion: ScoreFusionConfig = field(default_factory=ScoreFusionConfig)
    sam: SAMConfig = field(default_factory=SAMConfig)
    seed: int = 0
    deterministic: bool = True
    output_root: str = "results"


# --------------------------------------------------------------------------
# Thermal branch
# --------------------------------------------------------------------------


@dataclass
class ThermalPreprocessConfig:
    """Radiometric calibration and registration (FR-9.7).

    Attributes:
        gain / offset: Digital-number-to-temperature conversion per the
            dataset's documented calibration, used when the file format does
            not carry radiometry itself.
        assume_radiometric: Escape hatch for datasets whose radiometry cannot
            be auto-detected. Defaults to ``False``: unproven radiometry is
            treated as absent, so FR-9.9 refuses quantitative retrieval rather
            than fabricating a temperature (§11 edge case 9).
        register_frames: Align frames of a multi-frame sequence.
    """

    gain: float = 1.0
    offset: float = 0.0
    assume_radiometric: bool = False
    register_frames: bool = True


@dataclass
class CNNLSTMConfig:
    """CNN spatial features + LSTM temporal model (FR-9.8).

    Attributes:
        min_seq_len: Below this many frames the model degrades to CNN-only
            single-frame scoring and logs that it did so (§11 edge case 8).
        temporal_model: ``lstm``, the lighter ``gru`` fallback, or ``none`` to
            force single-frame scoring.
        interpolate_dropped_frames: Explicit gap handling for frame dropout,
            rather than a silent index shift (§11 edge case 10).
    """

    enabled: bool = True
    conv_channels: list[int] = field(default_factory=lambda: [16, 32, 64])
    kernel_size: int = 3
    hidden_size: int = 64
    min_seq_len: int = 3
    temporal_model: Literal["lstm", "gru", "none"] = "lstm"
    interpolate_dropped_frames: bool = True
    epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 1e-3
    device: Literal["auto", "cpu", "cuda"] = "auto"


@dataclass
class PlanckConfig:
    """Planck's law and emissivity retrieval (Eq. 5, FR-9.9).

    Attributes:
        wavelength_um: Band centre used for radiance calculations. LWIR
            thermal cameras are typically 8-14 um, so 10 um is the default.
        default_emissivity: Assumed emissivity when the material is unknown.
        default_emissivity_range: Stated uncertainty accompanying the default.
            §11 edge case 11 forbids a false-precision point estimate for an
            unlisted material, so the range travels with the answer.
        emissivity_table_path: Material-to-emissivity reference table.
    """

    wavelength_um: float = 10.0
    default_emissivity: float = 0.95
    default_emissivity_range: list[float] = field(default_factory=lambda: [0.85, 0.98])
    emissivity_table_path: str = "data/speclib/emissivity_table.json"


@dataclass
class ThermalConfig:
    """Top-level config for ``run_thermal`` (FR-9.12)."""

    dataset: DatasetConfig
    preprocess: ThermalPreprocessConfig = field(default_factory=ThermalPreprocessConfig)
    cnn_lstm: CNNLSTMConfig = field(default_factory=CNNLSTMConfig)
    planck: PlanckConfig = field(default_factory=PlanckConfig)
    score_threshold: ScoreFusionConfig = field(default_factory=ScoreFusionConfig)
    seed: int = 0
    deterministic: bool = True
    output_root: str = "results"


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


@dataclass
class FusionConfig:
    """Decision-level fusion of the two branches (FR-9.10).

    Attributes:
        mode: ``independent`` (the honest default given PRD §7.3's data
            reality) reports both branches side by side, explicitly labelled
            *not spatially fused*. ``coregistered`` performs pixel-level score
            fusion and is only valid for genuinely aligned imagery.
        require_registration_proof: When ``coregistered`` is requested, refuse
            unless the inputs carry evidence of alignment
            (:class:`UnregisteredInputError`, §11 edge case 12).
        weight_hyperspectral / weight_thermal: Decision-level weights.
    """

    mode: Literal["independent", "coregistered"] = "independent"
    require_registration_proof: bool = True
    weight_hyperspectral: float = 0.5
    weight_thermal: float = 0.5
    seed: int = 0
    output_root: str = "results"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

_NONE_TYPE = type(None)


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", str(tp))


def _convert(value: Any, tp: Any, path: str) -> Any:
    """Coerce ``value`` to type ``tp``, raising ConfigError with a path on failure."""
    origin = get_origin(tp)

    # Optional[X] / X | None / unions
    if origin is Union or isinstance(tp, types.UnionType):
        args = get_args(tp)
        if value is None:
            if _NONE_TYPE in args:
                return None
            raise ConfigError(f"{path}: null is not allowed here (expected {_type_name(tp)})")
        for arg in args:
            if arg is _NONE_TYPE:
                continue
            try:
                return _convert(value, arg, path)
            except ConfigError:
                continue
        raise ConfigError(f"{path}: {value!r} does not match any of {_type_name(tp)}")

    if origin is Literal:
        allowed = get_args(tp)
        if value not in allowed:
            options = ", ".join(repr(a) for a in allowed)
            raise ConfigError(f"{path}: {value!r} is not one of [{options}]")
        return value

    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping for {_type_name(tp)}, got {type(value).__name__}")
        return from_dict(tp, value, _path=path)

    if origin in (list, set, frozenset):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        (item_tp,) = get_args(tp) or (Any,)
        return [_convert(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(v, args[0], f"{path}[{i}]") for i, v in enumerate(value))
        if len(args) != len(value):
            raise ConfigError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(_convert(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args)))

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        key_tp, val_tp = get_args(tp) or (Any, Any)
        return {
            _convert(k, key_tp, f"{path}.<key>"): _convert(v, val_tp, f"{path}.{k}")
            for k, v in value.items()
        }

    if tp is Any:
        return value

    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value

    if tp is float:
        # bool is a subclass of int, so reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)

    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value

    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value

    if isinstance(tp, type) and isinstance(value, tp):
        return value

    raise ConfigError(f"{path}: cannot interpret {value!r} as {_type_name(tp)}")


def from_dict(cls: type, data: dict[str, Any], _path: str = "") -> Any:
    """Build a dataclass from a plain mapping, strictly.

    Unknown keys raise :class:`ConfigError` rather than being ignored, because
    a silently dropped typo is a reproducibility hazard (PRD §10). Missing keys
    without defaults are likewise an error.
    """
    if not is_dataclass(cls):
        raise ConfigError(f"{cls!r} is not a dataclass")
    if not isinstance(data, dict):
        raise ConfigError(f"{_path or cls.__name__}: expected a mapping, got {type(data).__name__}")

    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    unknown = sorted(set(data) - set(fields))
    if unknown:
        known = ", ".join(sorted(fields))
        where = _path or cls.__name__
        raise ConfigError(
            f"{where}: unknown option(s) {unknown}. Known options are: {known}. "
            "Unknown keys are rejected so a typo cannot silently change a run."
        )

    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        child = f"{_path}.{name}" if _path else name
        if name in data:
            kwargs[name] = _convert(data[name], hints[name], child)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:  # type: ignore[misc]
            where = _path or cls.__name__
            raise ConfigError(f"{where}: missing required option '{name}'")
    return cls(**kwargs)


def load_config(path: str | Path, cls: type) -> Any:
    """Load a YAML file into ``cls``.

    Raises:
        ConfigError: the file is missing, is not a mapping, or fails validation.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(
            f"Config file not found: {path}. "
            f"Available configs live in 'configs/'."
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML -- {exc}") from exc
    if raw is None:
        raise ConfigError(f"{path}: file is empty")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return from_dict(cls, raw, _path=path.name)


def to_dict(cfg: Any) -> dict[str, Any]:
    """Convert a config dataclass back to a plain, YAML-safe mapping."""
    if not is_dataclass(cfg) or isinstance(cfg, type):
        raise ConfigError("to_dict expects a dataclass instance")
    return dataclasses.asdict(cfg)


def config_hash(cfg: Any, length: int = 12) -> str:
    """Stable short hash of a *resolved* config (FR-9.11 traceability).

    Hashing the resolved config rather than the YAML source means a changed
    default is detected even when no config file changed. Key order is
    canonicalised so formatting differences do not alter the hash.
    """
    payload = to_dict(cfg) if is_dataclass(cfg) else cfg
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def set_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the pipeline uses (PRD §10, §11 edge case 15).

    ``torch`` is imported lazily so that modules which never touch the deep
    learning path stay fast to import and testable without it.

    Args:
        seed: Value applied to ``random``, ``numpy`` and ``torch``.
        deterministic: Also request deterministic cuDNN kernels. Slightly
            slower, but required for the bit-for-bit re-run check.
    """
    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # torch is optional for the classical-only path
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> str:
    """Resolve an ``auto``/``cpu``/``cuda`` request to a concrete device.

    PRD §10 makes GPU an optional speed-up, so ``auto`` falls back to CPU and an
    explicit ``cuda`` request on a machine without CUDA is an error rather than
    a silent downgrade -- a run that quietly changed device would blow the
    runtime budget without explanation.
    """
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        available = torch.cuda.is_available()
    except ImportError:
        available = False

    if requested == "cuda":
        if not available:
            raise ConfigError(
                "device: 'cuda' was requested but CUDA is not available. "
                "Use 'auto' to fall back to CPU, or 'cpu' explicitly."
            )
        return "cuda"
    if requested == "auto":
        return "cuda" if available else "cpu"
    raise ConfigError(f"device: unknown value {requested!r} (expected auto, cpu or cuda)")
