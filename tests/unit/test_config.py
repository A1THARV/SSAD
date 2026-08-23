"""Config loading, strictness, hashing and seeding (Roadmap 0.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.config import (
    AutoencoderConfig,
    FusionConfig,
    HyperspectralConfig,
    ThermalConfig,
    config_hash,
    from_dict,
    load_config,
    resolve_device,
    set_seeds,
    to_dict,
)
from pipeline.errors import ConfigError

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

SHIPPED_CONFIGS = [
    ("had100.yaml", HyperspectralConfig),
    ("salinas.yaml", HyperspectralConfig),
    ("abu_urban.yaml", HyperspectralConfig),
    ("thermal_default.yaml", ThermalConfig),
    ("fusion_default.yaml", FusionConfig),
]


@pytest.mark.parametrize("filename,cls", SHIPPED_CONFIGS)
def test_shipped_configs_load(filename: str, cls: type) -> None:
    """Every config in configs/ parses into its typed dataclass."""
    cfg = load_config(CONFIG_DIR / filename, cls)
    assert isinstance(cfg, cls)


def test_typed_fields_round_trip() -> None:
    """Loading yields real typed values, not raw strings from YAML."""
    cfg = load_config(CONFIG_DIR / "abu_urban.yaml", HyperspectralConfig)

    assert cfg.dataset.name == "abu_urban"
    assert cfg.dataset.kind == "hyperspectral"
    assert cfg.rx.ridge == pytest.approx(1e-6)
    assert isinstance(cfg.preprocess.water_absorption_bands_nm[0][0], float)
    assert cfg.autoencoder.encoder_channels == [32, 64, 128]
    assert cfg.seed == 0

    # Defaults fill in for anything the file omits.
    assert cfg.autoencoder.kernel_spatial == 3
    assert cfg.sam.library_path.endswith(".npz")


def test_salinas_is_the_zero_anomaly_fixture() -> None:
    """PRD v2 change C5 -- no anomaly labels are manufactured from a class map.

    The shipped 17-class ground truth is left unread; Salinas measures the
    false-positive rate on a scene declared to hold no anomalies.
    """
    cfg = load_config(CONFIG_DIR / "salinas.yaml", HyperspectralConfig)
    assert cfg.dataset.label_convention == "zero_anomaly"
    assert cfg.dataset.ground_truth_path is None
    assert cfg.dataset.anomaly_class_ids == []


def test_salinas_uses_an_adaptive_threshold() -> None:
    """A percentile threshold flags a fixed fraction whatever the scene holds.

    On a zero-anomaly scene that manufactures false positives by construction,
    so this fixture needs a method that can decline to split.
    """
    cfg = load_config(CONFIG_DIR / "salinas.yaml", HyperspectralConfig)
    assert cfg.score_fusion.threshold_method == "otsu"


def test_had100_trains_on_background_only() -> None:
    """PRD v2 change C2 -- the premise of Eq. 1 depends on this."""
    cfg = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    assert cfg.autoencoder.train_on_test_scene is False
    assert cfg.autoencoder.training_corpus is not None
    assert "train" in cfg.autoencoder.training_corpus


def test_had100_does_not_reapply_band_selection() -> None:
    """The preset must NOT run on data HAD100's main.py already selected.

    Its main.py reduces 425 raw bands to 276 before writing the .npy crops.
    Applying the preset again would index a 276-band cube with indices meant
    for a 425-band one -- band_presets refuses that, and this config must not
    ask for it in the first place.
    """
    cfg = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    assert cfg.preprocess.good_bands_preset == "none"
    assert cfg.dataset.sensor == "aviris_ng"


def test_had100_pca_width_comes_from_the_ablation() -> None:
    """PCA width is both a VRAM requirement and an ablated choice.

    15 was selected from the RX sweep across HAD100 *and* ABU, which disagree
    on their individual optima (8-12 vs 15-50). Picking HAD100's own peak would
    fit the test set this run reports against.
    """
    cfg = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    assert cfg.preprocess.pca_n_components == 15
    assert cfg.autoencoder.amp is True
    assert cfg.dataset.mmap is True


def test_both_hyperspectral_configs_share_one_dimensionality() -> None:
    """A single declared width, so cross-dataset numbers stay comparable."""
    had = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    abu = load_config(CONFIG_DIR / "abu_urban.yaml", HyperspectralConfig)
    assert had.preprocess.pca_n_components == abu.preprocess.pca_n_components


def test_had100_training_corpus_matches_the_test_sensor() -> None:
    """Mixing sensors would mean 162-band and 276-band cubes in one model."""
    cfg = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    assert cfg.autoencoder.training_corpus.endswith(cfg.dataset.sensor)


def test_fusion_defaults_to_independent_mode() -> None:
    """FR-9.10 -- fabricating pixel alignment is never the default."""
    cfg = load_config(CONFIG_DIR / "fusion_default.yaml", FusionConfig)
    assert cfg.mode == "independent"
    assert cfg.require_registration_proof is True


def test_thermal_does_not_assume_radiometry() -> None:
    """§11 edge case 9 -- unproven radiometry is treated as absent."""
    cfg = load_config(CONFIG_DIR / "thermal_default.yaml", ThermalConfig)
    assert cfg.preprocess.assume_radiometric is False


# --------------------------------------------------------------------------
# Strictness -- a typo must never be silently ignored (PRD §10)
# --------------------------------------------------------------------------


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError) as exc:
        from_dict(AutoencoderConfig, {"latnet_dim": 32})
    assert "latnet_dim" in str(exc.value)
    assert "latent_dim" in str(exc.value)  # the message lists valid options


def test_unknown_nested_key_reports_its_path() -> None:
    raw = {
        "dataset": {"name": "x", "kind": "hyperspectral", "path": "p", "bogus": 1},
    }
    with pytest.raises(ConfigError) as exc:
        from_dict(HyperspectralConfig, raw)
    assert "dataset" in str(exc.value)
    assert "bogus" in str(exc.value)


def test_missing_required_field_is_rejected() -> None:
    with pytest.raises(ConfigError, match="dataset"):
        from_dict(HyperspectralConfig, {})


def test_invalid_literal_is_rejected() -> None:
    with pytest.raises(ConfigError, match="mode"):
        from_dict(FusionConfig, {"mode": "pixel_level"})


def test_wrong_scalar_type_is_rejected() -> None:
    with pytest.raises(ConfigError, match="ridge|number"):
        from_dict(HyperspectralConfig, {
            "dataset": {"name": "x", "kind": "hyperspectral", "path": "p"},
            "rx": {"ridge": "small"},
        })


def test_optional_field_accepts_null() -> None:
    cfg = from_dict(AutoencoderConfig, {"max_patches": None})
    assert cfg.max_patches is None


def test_missing_config_file_is_actionable() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(CONFIG_DIR / "does_not_exist.yaml", HyperspectralConfig)


# --------------------------------------------------------------------------
# Hashing -- FR-9.11 traceability
# --------------------------------------------------------------------------


def test_hash_is_stable_across_loads() -> None:
    a = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    b = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    assert config_hash(a) == config_hash(b)


def test_hash_changes_when_any_value_changes() -> None:
    cfg = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    before = config_hash(cfg)
    cfg.rx.ridge = 1e-5
    assert config_hash(cfg) != before


def test_hash_covers_resolved_defaults_not_just_the_file() -> None:
    """A changed default must move the hash even though no YAML changed."""
    cfg = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    before = config_hash(cfg)
    cfg.autoencoder.kernel_spatial = 5  # never mentioned in the YAML file
    assert config_hash(cfg) != before


def test_different_configs_hash_differently() -> None:
    a = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    b = load_config(CONFIG_DIR / "salinas.yaml", HyperspectralConfig)
    assert config_hash(a) != config_hash(b)


def test_to_dict_is_yaml_safe() -> None:
    cfg = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    payload = to_dict(cfg)
    assert payload["dataset"]["name"] == "had100_aviris_ng"
    assert isinstance(payload, dict)


# --------------------------------------------------------------------------
# Seeding -- §11 edge case 15 depends on this holding
# --------------------------------------------------------------------------


def test_set_seeds_makes_numpy_reproducible() -> None:
    set_seeds(42)
    first = np.random.rand(8)
    set_seeds(42)
    second = np.random.rand(8)
    assert np.array_equal(first, second)


def test_different_seeds_differ() -> None:
    set_seeds(1)
    first = np.random.rand(8)
    set_seeds(2)
    second = np.random.rand(8)
    assert not np.array_equal(first, second)


def test_set_seeds_makes_torch_reproducible() -> None:
    torch = pytest.importorskip("torch")
    set_seeds(7)
    first = torch.rand(8)
    set_seeds(7)
    second = torch.rand(8)
    assert torch.equal(first, second)


def test_resolve_device_honours_explicit_cpu() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_never_fails() -> None:
    """PRD §10 -- GPU is a speed-up, never a hard requirement."""
    assert resolve_device("auto") in {"cpu", "cuda"}


def test_resolve_device_rejects_nonsense() -> None:
    with pytest.raises(ConfigError):
        resolve_device("tpu")


# --------------------------------------------------------------------------
# Conformance with the paper's Fig. 1 -- the architecture must not drift
# --------------------------------------------------------------------------


def test_autoencoder_defaults_match_figure_1() -> None:
    """Fig. 1 draws Conv3D 32/64/128 with kernels 3x3x7 -> 3x3x5 -> 3x3x3.

    If the implementation diverges from the published figure, either the code
    or the figure is wrong. Pinning it here means a drift shows up as a failing
    test rather than as a reviewer's question.
    """
    cfg = AutoencoderConfig()
    assert cfg.encoder_channels == [32, 64, 128]
    assert cfg.kernel_spatial == 3
    assert cfg.kernel_spectral_depths == [7, 5, 3]


def test_latent_is_narrower_than_the_pca_width() -> None:
    """Fig. 1 annotates the latent as z in R^d with d << K.

    A latent wider than its own input is not a bottleneck, so the
    reconstruction-error premise of Eq. 1 would not hold.
    """
    cfg = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    k = cfg.preprocess.pca_n_components
    assert k is not None
    assert cfg.autoencoder.latent_dim < k


def test_pca_width_supports_the_deepest_spectral_kernel() -> None:
    """Fig. 1's first encoder kernel has spectral depth 7, so K must be >= 7.

    This is a real constraint on the PCA sweep: HAD100's own optimum sits near
    8 components, which leaves a depth-7 kernel almost no spectral extent to
    convolve over.
    """
    cfg = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    assert cfg.preprocess.pca_n_components >= max(cfg.autoencoder.kernel_spectral_depths)


def test_detection_configs_do_not_default_to_percentile() -> None:
    """A percentile flags a fixed fraction whatever the scene contains.

    The Salinas zero-anomaly run flagged 1.00% of a scene with no anomalies at
    all under p99, every pixel a false positive; Otsu flagged 0.22%. Pinned as
    a test so the default cannot quietly regress.
    """
    for name in ("had100.yaml", "abu_urban.yaml", "salinas.yaml"):
        cfg = load_config(CONFIG_DIR / name, HyperspectralConfig)
        assert cfg.score_fusion.threshold_method != "percentile", name


def test_rx_pca_basis_defaults_to_per_scene() -> None:
    """RX is a per-scene unsupervised detector; its basis default reflects that.

    It is also what produced the M1 baseline benchmarked against HAD100's 19
    published methods, so the default cannot drift without invalidating that
    comparison. The corpus alternative is run at M2 as an ablation.
    """
    for name in ("had100.yaml", "abu_urban.yaml", "salinas.yaml"):
        cfg = load_config(CONFIG_DIR / name, HyperspectralConfig)
        assert cfg.rx.pca_basis == "per_scene", name


def test_rx_basis_and_autoencoder_basis_are_independent_settings() -> None:
    """The autoencoder has no choice -- corpus-fit is mandatory for fixed axes.

    Only RX has a basis option, so the two are separate fields rather than one
    shared switch.
    """
    cfg = load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)
    assert hasattr(cfg.rx, "pca_basis")
    assert not hasattr(cfg.autoencoder, "pca_basis")
    assert cfg.autoencoder.training_corpus is not None
