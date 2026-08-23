"""Eq. 1, Eq. 3, and Fig. 1's network.

Torch-dependent tests are skipped rather than failed when torch is absent --
PRD §10 makes the deep path optional, and Eq. 1 must stay testable without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import AutoencoderConfig, PreprocessConfig
from pipeline.errors import (
    DegenerateInputError,
    IncompatibleInputError,
    ModelNotFittedError,
)
from pipeline.hyperspectral.autoencoder import (
    SpectralBackProjector,
    SpectralSpatialAutoencoder,
    composite_loss,
    extract_patches,
    reconstruction_error,
    reconstruction_error_map,
    sample_corpus_patches,
    spectral_angle_torch,
    train_autoencoder,
)
from pipeline.hyperspectral.preprocess import BandReducer
from pipeline.hyperspectral.sam import spectral_angle_paired

torch = pytest.importorskip("torch", reason="deep path is optional (PRD §10)")


# --------------------------------------------------------------------------
# Eq. 1
# --------------------------------------------------------------------------


def test_perfect_reconstruction_has_zero_error():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert reconstruction_error(x, x) == pytest.approx(0.0)


def test_error_is_the_unsquared_l2_norm_of_eq_1():
    x = np.array([[0.0, 0.0, 0.0]])
    x_hat = np.array([[3.0, 4.0, 0.0]])
    # Eq. 1 as written has no square: ||(3,4,0)|| = 5, not 25.
    assert reconstruction_error(x, x_hat)[0] == pytest.approx(5.0)


def test_error_preserves_spatial_shape():
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(6, 5, 4))
    assert reconstruction_error(cube, cube + 0.1).shape == (6, 5)


def test_mismatched_shapes_are_refused():
    with pytest.raises(IncompatibleInputError):
        reconstruction_error(np.zeros((4, 3)), np.zeros((4, 5)))


def test_nan_reconstruction_is_refused_not_ranked():
    # A diverged autoencoder emits NaN. NaN scores tie in AUC rather than
    # erroring, so a silent pass here would report a plausible-looking number
    # from a broken model.
    x = np.zeros((2, 3))
    bad = np.full((2, 3), np.nan)
    with pytest.raises(DegenerateInputError):
        reconstruction_error(x, bad)


# --------------------------------------------------------------------------
# Eq. 3's SAM term
# --------------------------------------------------------------------------


def test_torch_sam_matches_the_numpy_path():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 12)) + 5.0
    r = rng.normal(size=(16, 12)) + 5.0
    expected = spectral_angle_paired(x, r)
    got = spectral_angle_torch(
        torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(r, dtype=torch.float64)
    ).numpy()
    assert got == pytest.approx(expected, abs=1e-9)


def test_torch_sam_is_magnitude_invariant():
    rng = np.random.default_rng(4)
    u = torch.as_tensor(rng.normal(size=(5, 9)) + 3.0, dtype=torch.float64)
    v = torch.as_tensor(rng.normal(size=(5, 9)) + 3.0, dtype=torch.float64)
    base = spectral_angle_torch(u, v)
    assert spectral_angle_torch(u, 7.0 * v).numpy() == pytest.approx(
        base.numpy(), abs=1e-9
    )
    assert spectral_angle_torch(2.5 * u, v).numpy() == pytest.approx(
        base.numpy(), abs=1e-9
    )


def test_sam_gradient_is_finite_at_perfect_reconstruction():
    # The norm's gradient is d/||d||, i.e. 0/0 when the reconstruction is exact
    # -- the point the optimiser is driving towards. Without the smoothing eps
    # a converged batch backpropagates NaN and destroys the weights.
    x = torch.ones(4, 6, dtype=torch.float64, requires_grad=True)
    r = torch.ones(4, 6, dtype=torch.float64, requires_grad=True)
    spectral_angle_torch(x, r).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(r.grad).all()


def test_composite_loss_is_mse_plus_lambda_sam():
    rng = np.random.default_rng(5)
    x = torch.as_tensor(rng.normal(size=(8, 10)) + 2.0, dtype=torch.float32)
    r = torch.as_tensor(rng.normal(size=(8, 10)) + 2.0, dtype=torch.float32)
    total, mse, sam = composite_loss(x, r, lam=0.3)
    assert total.item() == pytest.approx(mse.item() + 0.3 * sam.item(), rel=1e-6)


def test_lambda_zero_reduces_to_plain_mse():
    rng = np.random.default_rng(6)
    x = torch.as_tensor(rng.normal(size=(8, 10)), dtype=torch.float32)
    r = torch.as_tensor(rng.normal(size=(8, 10)), dtype=torch.float32)
    total, mse, sam = composite_loss(x, r, lam=0.0)
    assert sam.item() == 0.0
    assert total.item() == pytest.approx(mse.item())


def test_negative_lambda_is_refused():
    x = torch.zeros(2, 4)
    with pytest.raises(DegenerateInputError):
        composite_loss(x, x, lam=-0.1)


# --------------------------------------------------------------------------
# The SAM domain decision
# --------------------------------------------------------------------------


def _fitted_reducer(seed: int = 0, k: int = 6, n_bands: int = 20) -> BandReducer:
    rng = np.random.default_rng(seed)
    # Reflectance-like: strictly positive, with a large mean offset.
    data = rng.normal(size=(400, n_bands)) * 0.05 + np.linspace(0.2, 0.6, n_bands)
    return BandReducer(n_components=k).fit(data)


def test_back_projection_recovers_the_reflectance_domain():
    reducer = _fitted_reducer(k=20, n_bands=20)  # full rank -> exact round trip
    rng = np.random.default_rng(1)
    spectra = rng.normal(size=(7, 20)) * 0.05 + np.linspace(0.2, 0.6, 20)
    scores = reducer.transform(spectra)
    proj = SpectralBackProjector(reducer, whitened=False, device="cpu")
    back = proj(torch.as_tensor(scores, dtype=torch.float32)).numpy()
    assert back == pytest.approx(spectra, abs=1e-4)


def test_back_projection_undoes_whitening():
    reducer = _fitted_reducer(k=20, n_bands=20)
    rng = np.random.default_rng(2)
    spectra = rng.normal(size=(7, 20)) * 0.05 + np.linspace(0.2, 0.6, 20)
    scores = reducer.transform(spectra, whiten=True)
    proj = SpectralBackProjector(reducer, whitened=True, device="cpu")
    back = proj(torch.as_tensor(scores, dtype=torch.float32)).numpy()
    assert back == pytest.approx(spectra, abs=1e-4)


def test_latent_and_spectral_sam_are_not_the_same_quantity():
    # The reason sam_domain exists. If these agreed, the flag would be noise.
    reducer = _fitted_reducer()
    rng = np.random.default_rng(7)
    spectra = rng.normal(size=(64, 20)) * 0.05 + np.linspace(0.2, 0.6, 20)
    recon = spectra + rng.normal(size=spectra.shape) * 0.01

    z = torch.as_tensor(reducer.transform(spectra), dtype=torch.float32)
    zh = torch.as_tensor(reducer.transform(recon), dtype=torch.float32)
    proj = SpectralBackProjector(reducer, whitened=False, device="cpu")

    latent = spectral_angle_torch(z, zh).mean().item()
    spectral = spectral_angle_torch(proj(z), proj(zh)).mean().item()
    # Mean-centring puts the origin at the corpus mean, so latent angles are
    # dramatically larger for the same physical spectra.
    assert latent > 5 * spectral


def test_spectral_sam_domain_requires_a_reducer():
    cfg = AutoencoderConfig(
        latent_dim=2, encoder_channels=[4], kernel_spectral_depths=[3], epochs=1
    )
    patches = np.zeros((4, 1, 6, 3, 3), dtype=np.float32)
    with pytest.raises(ModelNotFittedError):
        train_autoencoder(patches, cfg, reducer=None, device="cpu")


# --------------------------------------------------------------------------
# Fig. 1 conformance
# --------------------------------------------------------------------------


def test_default_config_matches_figure_1():
    cfg = AutoencoderConfig()
    assert cfg.encoder_channels == [32, 64, 128]
    assert cfg.kernel_spatial == 3
    assert cfg.kernel_spectral_depths == [7, 5, 3]


def test_latent_wider_than_input_is_not_a_bottleneck():
    # Fig. 1 annotates z in R^d with d << K. This exact misconfiguration
    # (latent 32 against 30 components) shipped once and was caught here.
    with pytest.raises(IncompatibleInputError, match="bottleneck"):
        SpectralSpatialAutoencoder.build(
            n_components=30,
            patch_size=7,
            latent_dim=32,
            encoder_channels=[32, 64, 128],
            kernel_spatial=3,
            kernel_spectral_depths=[7, 5, 3],
        )


def test_kernel_deeper_than_the_pca_width_is_refused():
    with pytest.raises(IncompatibleInputError, match="spectral depth"):
        SpectralSpatialAutoencoder.build(
            n_components=5,
            patch_size=7,
            latent_dim=2,
            encoder_channels=[8],
            kernel_spectral_depths=[7],
            kernel_spatial=3,
        )


def test_mismatched_layer_and_kernel_counts_are_refused():
    with pytest.raises(IncompatibleInputError):
        SpectralSpatialAutoencoder.build(
            n_components=15,
            patch_size=7,
            latent_dim=4,
            encoder_channels=[32, 64],
            kernel_spectral_depths=[7, 5, 3],
            kernel_spatial=3,
        )


def test_even_patch_size_has_no_centre_pixel():
    with pytest.raises(DegenerateInputError, match="centre"):
        SpectralSpatialAutoencoder.build(
            n_components=15,
            patch_size=6,
            latent_dim=4,
            encoder_channels=[8],
            kernel_spectral_depths=[3],
            kernel_spatial=3,
        )


def _small_model(n_components=10, patch_size=5, latent_dim=4):
    return SpectralSpatialAutoencoder.build(
        n_components=n_components,
        patch_size=patch_size,
        latent_dim=latent_dim,
        encoder_channels=[4, 8],
        kernel_spatial=3,
        kernel_spectral_depths=[5, 3],
    )


def test_forward_preserves_patch_shape():
    model = _small_model()
    x = torch.randn(3, 1, 10, 5, 5)
    assert model(x).shape == x.shape


def test_latent_is_the_configured_width():
    model = _small_model(latent_dim=4)
    assert model.encode(torch.randn(3, 1, 10, 5, 5)).shape == (3, 4)


def test_output_layer_is_linear_so_negative_scores_survive():
    # PCA scores are signed. A ReLU or sigmoid output would clip every negative
    # component to zero, capping the reconstruction error of exactly the pixels
    # furthest from the background mean -- the anomalies.
    model = _small_model()
    torch.manual_seed(0)
    out = model(torch.randn(64, 1, 10, 5, 5) * 5.0)
    assert (out < 0).any(), "decoder cannot represent negative PCA scores"


def test_centre_spectra_picks_the_middle_pixel():
    model = _small_model(patch_size=5)
    x = torch.zeros(2, 1, 10, 5, 5)
    x[:, 0, :, 2, 2] = 9.0
    assert torch.all(model.centre_spectra(x) == 9.0)


# --------------------------------------------------------------------------
# Patches
# --------------------------------------------------------------------------


def test_patch_count_matches_pixel_count():
    cube = np.random.default_rng(0).normal(size=(8, 6, 4))
    patches = extract_patches(cube, 3)
    assert patches.shape == (48, 1, 4, 3, 3)


def test_patch_centre_is_its_own_pixel():
    cube = np.arange(5 * 4 * 2, dtype=np.float32).reshape(5, 4, 2)
    patches = extract_patches(cube, 3)
    for i in range(5):
        for j in range(4):
            assert patches[i * 4 + j, 0, :, 1, 1] == pytest.approx(cube[i, j])


def test_border_pixels_are_kept_not_cropped():
    # Dropping them would leave the score map smaller than the ground-truth
    # mask, and a partially visible target is most likely to sit at an edge.
    cube = np.random.default_rng(1).normal(size=(4, 4, 3))
    assert extract_patches(cube, 5).shape[0] == 16


def test_patch_size_one_is_the_purely_spectral_case():
    cube = np.random.default_rng(2).normal(size=(3, 3, 5))
    patches = extract_patches(cube, 1)
    assert patches.shape == (9, 1, 5, 1, 1)
    assert patches[:, 0, :, 0, 0] == pytest.approx(cube.reshape(9, 5))


def test_even_patch_size_is_refused_at_extraction_too():
    with pytest.raises(DegenerateInputError):
        extract_patches(np.zeros((4, 4, 3)), 2)


def test_corpus_sampling_respects_the_cap(tmp_path):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(3):
        p = tmp_path / f"crop_{i}.npy"
        np.save(p, rng.normal(size=(8, 8, 12)).astype(np.float32) + 1.0)
        paths.append(p)

    reducer = BandReducer(n_components=6).fit(rng.normal(size=(200, 12)) + 1.0)
    patches = sample_corpus_patches(
        paths, reducer, PreprocessConfig(), patch_size=3, max_patches=30, seed=0
    )
    assert patches.shape[0] == 30
    assert patches.shape[1:] == (1, 6, 3, 3)


def test_corpus_sampling_draws_from_every_crop(tmp_path):
    # A cap smaller than one crop's pixel count must not collapse to crop 0.
    paths = []
    for i in range(4):
        p = tmp_path / f"crop_{i}.npy"
        arr = np.full((8, 8, 12), float(i + 1), dtype=np.float32)
        arr += np.random.default_rng(i).normal(size=arr.shape) * 1e-3
        np.save(p, arr)
        paths.append(p)

    reducer = BandReducer(n_components=4).fit(
        np.random.default_rng(9).normal(size=(300, 12)) + 2.5
    )
    patches = sample_corpus_patches(
        paths, reducer, PreprocessConfig(clip_percentile=100.0),
        patch_size=1, max_patches=40, seed=0,
    )
    # Four well-separated constant crops -> four distinct score clusters.
    centres = patches[:, 0, 0, 0, 0]
    assert len(np.unique(np.round(centres, 1))) >= 4


def test_empty_corpus_says_what_to_do():
    with pytest.raises(DegenerateInputError, match="train_on_test_scene"):
        sample_corpus_patches([], BandReducer(), PreprocessConfig(), 3, 10)


# --------------------------------------------------------------------------
# Training and inference
# --------------------------------------------------------------------------


def _tiny_cfg(**kw):
    base = dict(
        latent_dim=3,
        encoder_channels=[4, 8],
        kernel_spatial=3,
        kernel_spectral_depths=[5, 3],
        lam=0.1,
        epochs=3,
        batch_size=16,
        learning_rate=1e-2,
        amp=False,
        device="cpu",
    )
    base.update(kw)
    return AutoencoderConfig(**base)


def _toy_corpus(n=64, k=8, p=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 1, k, p, p)).astype(np.float32) * 0.1


def test_training_reduces_the_loss():
    model, history = train_autoencoder(
        _toy_corpus(), _tiny_cfg(lam=0.0, epochs=8), device="cpu", seed=0
    )
    assert history.loss[-1] < history.loss[0]
    assert len(history.loss) == 8


def test_training_is_reproducible_under_a_seed():
    _, a = train_autoencoder(_toy_corpus(), _tiny_cfg(lam=0.0), device="cpu", seed=7)
    _, b = train_autoencoder(_toy_corpus(), _tiny_cfg(lam=0.0), device="cpu", seed=7)
    assert a.loss == pytest.approx(b.loss, rel=1e-9)


def test_history_records_both_loss_terms_separately():
    reducer = _fitted_reducer(k=8, n_bands=20)
    _, history = train_autoencoder(
        _toy_corpus(), _tiny_cfg(lam=0.5), reducer=reducer, device="cpu", seed=0
    )
    assert all(v >= 0 for v in history.sam)
    assert history.loss[0] == pytest.approx(
        history.mse[0] + 0.5 * history.sam[0], rel=1e-4
    )


def test_history_serialises_for_the_run_record():
    _, history = train_autoencoder(
        _toy_corpus(), _tiny_cfg(lam=0.0, epochs=2), device="cpu", seed=0
    )
    d = history.as_dict()
    assert d["epochs"] == 2 and d["n_patches"] == 64 and d["device"] == "cpu"
    assert len(d["loss"]) == 2


def test_wrong_patch_rank_is_refused():
    with pytest.raises(IncompatibleInputError):
        train_autoencoder(np.zeros((4, 8, 5, 5), dtype=np.float32), _tiny_cfg(), device="cpu")


def test_error_map_has_the_scene_shape():
    model, _ = train_autoencoder(
        _toy_corpus(k=8, p=5), _tiny_cfg(lam=0.0, epochs=1), device="cpu", seed=0
    )
    cube = np.random.default_rng(1).normal(size=(9, 7, 8)).astype(np.float32) * 0.1
    scores = reconstruction_error_map(model, cube, patch_size=5, device="cpu")
    assert scores.shape == (9, 7)
    assert np.all(np.isfinite(scores))


def test_error_map_refuses_a_different_pca_width():
    model, _ = train_autoencoder(
        _toy_corpus(k=8), _tiny_cfg(lam=0.0, epochs=1), device="cpu", seed=0
    )
    cube = np.zeros((5, 5, 12), dtype=np.float32)
    with pytest.raises(IncompatibleInputError, match="one PCA basis"):
        reconstruction_error_map(model, cube, patch_size=5, device="cpu")


def test_a_planted_anomaly_scores_above_the_background():
    # The end-to-end premise of Eq. 1: train on background only, and a spectrum
    # unlike that background must reconstruct worse.
    rng = np.random.default_rng(0)
    k, p = 8, 5
    background = (rng.normal(size=(600, 1, k, p, p)) * 0.05).astype(np.float32)
    model, _ = train_autoencoder(
        background, _tiny_cfg(lam=0.0, epochs=25, batch_size=64), device="cpu", seed=0
    )

    scene = (rng.normal(size=(16, 16, k)) * 0.05).astype(np.float32)
    scene[8, 8, :] += 3.0  # a spectrum the background model has never seen
    scores = reconstruction_error_map(model, scene, patch_size=p, device="cpu")

    mask = np.ones_like(scores, dtype=bool)
    mask[7:10, 7:10] = False  # exclude the target's own neighbourhood
    assert scores[8, 8] > scores[mask].max()
