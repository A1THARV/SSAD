"""The end-to-end runner's wiring, without touching a dataset."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.cli.run_hyperspectral import apply_overrides, build_parser, score_scene
from pipeline.config import HyperspectralConfig, load_config
from pipeline.errors import PipelineError
from pipeline.hyperspectral.loader import HSICube
from pipeline.hyperspectral.preprocess import BandReducer

CONFIG = "configs/had100.yaml"


def _cfg() -> HyperspectralConfig:
    return load_config(CONFIG, HyperspectralConfig)


# --------------------------------------------------------------------------
# --set overrides
# --------------------------------------------------------------------------


def test_override_reaches_a_nested_field():
    cfg = apply_overrides(_cfg(), ["autoencoder.lam=0.5"])
    assert cfg.autoencoder.lam == 0.5


def test_override_coerces_to_the_field_type():
    cfg = apply_overrides(_cfg(), ["autoencoder.epochs=3", "preprocess.pca_n_components=20"])
    assert cfg.autoencoder.epochs == 3 and isinstance(cfg.autoencoder.epochs, int)
    assert cfg.preprocess.pca_n_components == 20


def test_boolean_override_reads_words_not_just_digits():
    assert apply_overrides(_cfg(), ["autoencoder.whiten_input=true"]).autoencoder.whiten_input
    assert not apply_overrides(_cfg(), ["autoencoder.whiten_input=false"]).autoencoder.whiten_input


def test_numeric_typo_fails_at_parse_time():
    # The alternative is a string silently reaching training and comparing
    # unequal to every number it is tested against.
    with pytest.raises(ValueError):
        apply_overrides(_cfg(), ["autoencoder.epochs=twenty"])


def test_unknown_field_is_refused():
    with pytest.raises(PipelineError, match="no config field"):
        apply_overrides(_cfg(), ["autoencoder.lambduh=0.5"])


def test_unknown_section_is_refused():
    with pytest.raises(PipelineError, match="no config section"):
        apply_overrides(_cfg(), ["autoncoder.lam=0.5"])


def test_malformed_override_is_refused():
    with pytest.raises(PipelineError, match="PATH=VALUE"):
        apply_overrides(_cfg(), ["autoencoder.lam"])


def test_list_valued_fields_are_not_silently_stringified():
    with pytest.raises(PipelineError, match="edit the YAML"):
        apply_overrides(_cfg(), ["autoencoder.encoder_channels=[8,16]"])


# --------------------------------------------------------------------------
# Branch wiring
# --------------------------------------------------------------------------


def _toy_scene(h=12, w=12, b=24, seed=0):
    rng = np.random.default_rng(seed)
    data = (rng.normal(size=(h, w, b)) * 0.05 + np.linspace(0.2, 0.6, b)).astype(np.float32)
    return HSICube(name="toy", data=data, wavelengths=None, ground_truth=None)


def test_rx_only_scene_yields_just_the_rx_branch():
    cfg = _cfg()
    cfg.preprocess.pca_n_components = 6
    cfg.preprocess.good_bands_preset = "none"
    out = score_scene(_toy_scene(), cfg, model=None, corpus_pca=BandReducer(), device="cpu")
    assert set(out) == {"rx"}
    assert out["rx"].shape == (12, 12)


def test_corpus_basis_is_used_when_configured():
    # The Fig. 1 question: one PCA block feeding both branches. If the flag were
    # ignored, the unfitted reducer passed here would go unnoticed.
    cfg = _cfg()
    cfg.preprocess.pca_n_components = 6
    cfg.preprocess.good_bands_preset = "none"
    cfg.rx.pca_basis = "corpus"
    from pipeline.errors import ModelNotFittedError

    with pytest.raises(ModelNotFittedError):
        score_scene(_toy_scene(), cfg, model=None, corpus_pca=BandReducer(), device="cpu")


def test_per_scene_basis_does_not_need_a_corpus_fit():
    cfg = _cfg()
    cfg.preprocess.pca_n_components = 6
    cfg.preprocess.good_bands_preset = "none"
    cfg.rx.pca_basis = "per_scene"
    out = score_scene(_toy_scene(), cfg, model=None, corpus_pca=BandReducer(), device="cpu")
    assert np.all(np.isfinite(out["rx"]))


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


def test_parser_requires_a_config():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parser_accepts_repeated_overrides():
    args = build_parser().parse_args(
        ["--config", CONFIG, "--set", "a.b=1", "--set", "c.d=2"]
    )
    assert args.set == ["a.b=1", "c.d=2"]
