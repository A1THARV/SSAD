"""Spectral-spatial autoencoder: Eq. 1, Eq. 3, and the Fig. 1 network (FR-9.4).

The paper's three pieces, kept separable so each can be tested without the
others:

* **Eq. 1** -- ``E(x) = ||x - x_hat||_2``, the per-pixel reconstruction error
  that becomes the anomaly score. Pure NumPy, no network involved.
* **Eq. 3** -- ``L = MSE(x, x_hat) + lambda * SAM(x, x_hat)``, the composite
  training loss. Differentiable, and its SAM term mirrors :mod:`sam`'s atan2
  formulation rather than re-deriving an arccos one.
* **Fig. 1** -- a 3D-convolutional encoder/decoder over spectral-spatial
  patches, widths and kernel depths taken from the figure and held in config so
  the drawn architecture and the built one cannot drift apart.

Training is patch-based, not whole-cube. That is forced by the 4 GB card (PRD
§10, §14 risk): a 32-filter 3D conv over a full 276-band crop holds on the
order of 145 MB of activations *per sample*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import (
    DegenerateInputError,
    IncompatibleInputError,
    ModelNotFittedError,
)

__all__ = [
    "reconstruction_error",
    "spectral_angle_torch",
    "composite_loss",
    "SpectralBackProjector",
    "SpectralSpatialAutoencoder",
    "extract_patches",
    "sample_corpus_patches",
    "TrainingHistory",
    "train_autoencoder",
    "reconstruction_error_map",
]


# --------------------------------------------------------------------------
# Eq. 1 -- reconstruction error
# --------------------------------------------------------------------------


def reconstruction_error(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Per-pixel reconstruction error, Eq. 1: ``E(x) = ||x - x_hat||_2``.

    Note this is the *un*-squared L2 norm, matching Eq. 1 as written. It is the
    opposite convention to :func:`rx_detector.rx_score`, which follows Eq. 2's
    squared Mahalanobis distance -- the two equations genuinely differ in the
    paper, and Fig. 1's stage-5 fusion normalises each branch by its own maximum
    before combining, so the difference is not cosmetic. See the Fig. 1 edit
    list: box 4c draws a square root that Eq. 2 does not have.

    Args:
        x: ``(..., C)`` original spectra.
        x_hat: ``(..., C)`` reconstructions, same shape.

    Returns:
        ``(...,)`` errors -- one scalar per pixel, spatial shape preserved.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(x_hat, dtype=np.float64)
    if a.shape != b.shape:
        raise IncompatibleInputError(
            f"Reconstruction error needs matching shapes, got {a.shape} and {b.shape}."
        )
    if a.shape[-1] < 1:
        raise DegenerateInputError("Cannot take a reconstruction error over 0 bands.")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise DegenerateInputError(
            "Non-finite values reached the reconstruction error. A diverged "
            "autoencoder produces NaN scores that silently rank as ties in AUC."
        )
    return np.linalg.norm(a - b, axis=-1)


# --------------------------------------------------------------------------
# Eq. 3 -- composite loss
# --------------------------------------------------------------------------


def spectral_angle_torch(x, r, eps: float = 1e-12):
    """Differentiable Eq. 4, mirroring :func:`sam.spectral_angle_paired`.

    Uses ``2*atan2(||u-v||, ||u+v||)`` on unit-normalised rows, the same form as
    the NumPy path and for the same reason: arccos is ill-conditioned exactly
    where a well-trained autoencoder operates, near zero angle.

    The ``eps`` is not cosmetic. ``||u - v||`` is non-differentiable at ``u = v``
    -- the norm's gradient is ``d/||d||``, which is ``0/0`` there -- so a batch
    that reconstructs perfectly would backpropagate NaN. Smoothing the norm to
    ``sqrt(sum(d^2) + eps)`` keeps the gradient finite at the one point the
    optimiser is actively driving towards.

    Args:
        x: ``(N, C)`` spectra.
        r: ``(N, C)`` reconstructions.
        eps: Norm smoothing term.

    Returns:
        ``(N,)`` angles in radians.
    """
    import torch

    if x.shape != r.shape:
        raise IncompatibleInputError(
            f"Paired spectral angles need matching shapes, got {tuple(x.shape)} and "
            f"{tuple(r.shape)}."
        )

    def _norm(t):
        return torch.sqrt(torch.sum(t * t, dim=-1) + eps)

    u = x / _norm(x).unsqueeze(-1)
    v = r / _norm(r).unsqueeze(-1)
    return 2.0 * torch.atan2(_norm(u - v), _norm(u + v))


class SpectralBackProjector:
    """Maps PCA scores back to reflectance so Eq. 4 keeps its physical meaning.

    Eq. 4's justification is that the angle between two *reflectance* spectra is
    invariant to illumination -- a shadowed and a sunlit patch of the same
    material differ in magnitude, not direction. That argument is about the
    reflectance domain and does not survive the move to PCA scores:

    * PCA scores are mean-centred, so the origin is the corpus *mean spectrum*,
      not darkness. A perfectly ordinary background pixel projects to near zero,
      where its direction is dominated by noise and the angle to anything is
      unstable.
    * The angle between score vectors is not the angle between the spectra they
      encode. PCA components are orthonormal in band space but the mean shift
      breaks the correspondence.

    So the SAM term is evaluated after back-projecting both the input and the
    reconstruction to ``B`` bands: ``x = z @ W + mu`` (undoing whitening first
    when it is enabled). That is a fixed linear map, cheap and differentiable,
    and it puts Eq. 4 back in the domain where Eq. 4 is defined. The
    back-projected spectra also sit far from the origin -- ``mu`` dominates --
    so the near-zero instability disappears.

    Set ``AutoencoderConfig.sam_domain = 'latent'`` to skip this and take the
    angle on the scores directly; it is kept as an ablation, not a default.
    """

    def __init__(self, reducer, whitened: bool, device: str, dtype=None):
        import torch

        if not reducer.is_fitted:
            raise ModelNotFittedError(
                "SpectralBackProjector needs a fitted BandReducer -- it back-projects "
                "through the same components the encoder's input was produced with."
            )
        dtype = dtype or torch.float32
        scale = np.sqrt(reducer.explained_variance_) if whitened else np.ones(
            reducer.n_output_components
        )
        # components_ is (k, B); scores are (N, k). x = (z * scale) @ components_ + mu
        self.weight = torch.as_tensor(
            (scale[:, None] * reducer.components_), dtype=dtype, device=device
        )
        self.mean = torch.as_tensor(reducer.mean_, dtype=dtype, device=device)

    def __call__(self, z):
        return z @ self.weight + self.mean


def composite_loss(x, x_hat, lam: float, projector: SpectralBackProjector | None = None):
    """Eq. 3: ``L = MSE(x, x_hat) + lambda * SAM(x, x_hat)``.

    Args:
        x: ``(N, C)`` inputs (PCA scores).
        x_hat: ``(N, C)`` reconstructions.
        lam: Eq. 3's ``lambda``, weighting the angular term.
        projector: Back-projection to reflectance for the SAM term. ``None``
            takes the angle on the scores directly -- see
            :class:`SpectralBackProjector` for why that is the ablation and not
            the default.

    Returns:
        ``(total, mse, sam_mean)`` -- the three as separate tensors, so training
        can report which term is actually moving rather than only their sum.
    """
    import torch

    if lam < 0:
        raise DegenerateInputError(f"Eq. 3's lambda must be >= 0, got {lam}.")

    # Deliberately computed in fp32 even under autocast. Residuals late in
    # training are small, and fp16 underflows the sum of squares inside the
    # angle's norm long before the reconstruction is actually perfect.
    xf = x.float()
    hf = x_hat.float()

    mse = torch.mean((xf - hf) ** 2)

    if lam == 0:
        sam = torch.zeros((), device=xf.device, dtype=xf.dtype)
    else:
        a, b = (projector(xf), projector(hf)) if projector is not None else (xf, hf)
        sam = torch.mean(spectral_angle_torch(a, b))

    return mse + lam * sam, mse, sam


# --------------------------------------------------------------------------
# Fig. 1 -- the network
# --------------------------------------------------------------------------


class SpectralSpatialAutoencoder:
    """Fig. 1's 3D-conv encoder/decoder, built lazily so torch stays optional.

    Constructed via :meth:`build`, which returns an ``nn.Module``. Kept as a
    factory rather than a subclass of ``nn.Module`` so that importing this
    module -- and running Eq. 1's tests -- does not require torch.
    """

    @staticmethod
    def build(
        n_components: int,
        patch_size: int,
        latent_dim: int,
        encoder_channels: list[int],
        kernel_spatial: int,
        kernel_spectral_depths: list[int],
    ):
        import torch.nn as nn

        _validate_geometry(
            n_components,
            patch_size,
            latent_dim,
            encoder_channels,
            kernel_spatial,
            kernel_spectral_depths,
        )

        class _AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_components = n_components
                self.patch_size = patch_size
                self.latent_dim = latent_dim

                enc: list = []
                in_ch = 1
                for out_ch, depth in zip(encoder_channels, kernel_spectral_depths):
                    k = (depth, kernel_spatial, kernel_spatial)
                    pad = tuple(d // 2 for d in k)
                    enc.append(nn.Conv3d(in_ch, out_ch, kernel_size=k, padding=pad))
                    enc.append(nn.ReLU(inplace=True))
                    in_ch = out_ch
                self.encoder_conv = nn.Sequential(*enc)

                self.flat_dim = encoder_channels[-1] * n_components * patch_size * patch_size
                self.to_latent = nn.Linear(self.flat_dim, latent_dim)
                self.from_latent = nn.Linear(latent_dim, self.flat_dim)

                dec: list = []
                widths = list(encoder_channels[::-1])
                depths = list(kernel_spectral_depths[::-1])
                for i, depth in enumerate(depths):
                    in_c = widths[i]
                    out_c = widths[i + 1] if i + 1 < len(widths) else 1
                    k = (depth, kernel_spatial, kernel_spatial)
                    pad = tuple(d // 2 for d in k)
                    dec.append(
                        nn.ConvTranspose3d(in_c, out_c, kernel_size=k, padding=pad)
                    )
                    # No activation on the output layer. PCA scores are signed
                    # and unbounded; a ReLU or sigmoid here would clip every
                    # negative component to zero and cap the reconstruction
                    # error of exactly the pixels furthest from the mean.
                    if i + 1 < len(depths):
                        dec.append(nn.ReLU(inplace=True))
                self.decoder_conv = nn.Sequential(*dec)

            def encode(self, x):
                h = self.encoder_conv(x)
                return self.to_latent(h.flatten(1))

            def decode(self, z):
                h = self.from_latent(z).view(
                    -1,
                    encoder_channels[-1],
                    self.n_components,
                    self.patch_size,
                    self.patch_size,
                )
                return self.decoder_conv(h)

            def forward(self, x):
                return self.decode(self.encode(x))

            def centre_spectra(self, x):
                """Pull the centre pixel's spectrum out of a ``(N,1,C,p,p)`` batch."""
                c = self.patch_size // 2
                return x[:, 0, :, c, c]

        return _AE()


def _validate_geometry(
    n_components: int,
    patch_size: int,
    latent_dim: int,
    encoder_channels: list[int],
    kernel_spatial: int,
    kernel_spectral_depths: list[int],
) -> None:
    """Refuse architectures that contradict Fig. 1 before any VRAM is spent."""
    if len(encoder_channels) != len(kernel_spectral_depths):
        raise IncompatibleInputError(
            f"Fig. 1 pairs one kernel depth with each conv layer, but got "
            f"{len(encoder_channels)} channel widths and "
            f"{len(kernel_spectral_depths)} kernel depths."
        )
    if not encoder_channels:
        raise DegenerateInputError("An encoder needs at least one convolutional layer.")
    if patch_size < 1 or patch_size % 2 == 0:
        raise DegenerateInputError(
            f"patch_size must be odd so the patch has a centre pixel to score, "
            f"got {patch_size}."
        )
    if kernel_spatial % 2 == 0:
        raise DegenerateInputError(
            f"kernel_spatial must be odd for symmetric padding, got {kernel_spatial}."
        )
    if any(d % 2 == 0 for d in kernel_spectral_depths):
        raise DegenerateInputError(
            f"kernel_spectral_depths must all be odd for symmetric padding, got "
            f"{kernel_spectral_depths}."
        )
    if kernel_spatial > patch_size:
        raise IncompatibleInputError(
            f"A {kernel_spatial}x{kernel_spatial} kernel over a {patch_size}x"
            f"{patch_size} patch sees more padding than data. Widen patch_size."
        )
    deepest = max(kernel_spectral_depths)
    if deepest > n_components:
        raise IncompatibleInputError(
            f"Fig. 1's first encoder kernel has spectral depth {deepest}, but PCA "
            f"retains only {n_components} components -- the kernel would convolve "
            "mostly over padding. Raise preprocess.pca_n_components or shorten "
            "kernel_spectral_depths."
        )
    if latent_dim >= n_components:
        raise IncompatibleInputError(
            f"Fig. 1 annotates the latent as z in R^d with d << K, but latent_dim="
            f"{latent_dim} against K={n_components} components is not a bottleneck. "
            "Lower autoencoder.latent_dim or raise preprocess.pca_n_components."
        )


# --------------------------------------------------------------------------
# Patches
# --------------------------------------------------------------------------


def extract_patches(cube: np.ndarray, patch_size: int) -> np.ndarray:
    """Every pixel's spectral-spatial neighbourhood, in row-major pixel order.

    Edges are reflect-padded rather than dropped. Dropping them would leave the
    score map smaller than the ground-truth mask, and cropping the mask to match
    would quietly discard the border pixels -- where a partially-visible target
    is most likely to sit.

    Args:
        cube: ``(H, W, C)`` reduced cube.
        patch_size: Odd spatial edge length.

    Returns:
        ``(H*W, 1, C, patch_size, patch_size)`` float32, ordered so that
        ``out[i * W + j]`` is the patch centred on pixel ``(i, j)``.
    """
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim != 3:
        raise IncompatibleInputError(
            f"extract_patches expects a (H, W, C) cube, got shape {arr.shape}."
        )
    if patch_size % 2 == 0 or patch_size < 1:
        raise DegenerateInputError(f"patch_size must be odd and >= 1, got {patch_size}.")

    r = patch_size // 2
    padded = np.pad(arr, ((r, r), (r, r), (0, 0)), mode="reflect") if r else arr
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (patch_size, patch_size), axis=(0, 1)
    )  # (H, W, C, p, p)
    h, w = arr.shape[0], arr.shape[1]
    return np.ascontiguousarray(windows.reshape(h * w, 1, arr.shape[2], patch_size, patch_size))


def sample_corpus_patches(
    paths: list[Path],
    reducer,
    cfg,
    patch_size: int,
    max_patches: int | None,
    whiten: bool = False,
    good_bands: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Draw training patches from the anomaly-free corpus (change C2).

    One crop is memory-mapped at a time and its patches subsampled before the
    next is opened, so peak residency stays at one crop rather than the corpus.

    The corpus is what makes Eq. 1's premise defensible: a background model
    fitted on data containing the anomalies it must detect cannot support
    "anomalies reconstruct poorly".

    Args:
        paths: Corpus crops.
        reducer: PCA basis fitted across the whole corpus, so training and
            inference share axes.
        cfg: Preprocess config, for the clipping percentile.
        patch_size: Odd spatial edge.
        max_patches: Total cap. ``None`` takes every pixel of every crop.
        whiten: Divide components by ``sqrt(lambda)`` (ablation, see config).
        good_bands: Optional band mask, applied before projection.
        seed: RNG seed for the subsample.

    Returns:
        ``(N, 1, C, patch_size, patch_size)`` float32.
    """
    from .preprocess import clip_outliers

    if not paths:
        raise DegenerateInputError(
            "The autoencoder training corpus is empty. Set "
            "autoencoder.training_corpus, or enable train_on_test_scene to run "
            "the single-scene variant deliberately."
        )

    rng = np.random.default_rng(seed)
    per_crop = None
    if max_patches is not None:
        per_crop = max(1, int(np.ceil(max_patches / len(paths))))

    collected: list[np.ndarray] = []
    total = 0
    for path in paths:
        raw = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        if good_bands is not None:
            raw = raw[..., good_bands]
        raw = clip_outliers(raw, cfg.clip_percentile)
        reduced = reducer.transform(raw, whiten=whiten)
        patches = extract_patches(reduced, patch_size)

        if per_crop is not None and per_crop < patches.shape[0]:
            idx = rng.choice(patches.shape[0], size=per_crop, replace=False)
            patches = patches[idx]
        collected.append(patches)
        total += patches.shape[0]
        if max_patches is not None and total >= max_patches:
            break

    out = np.concatenate(collected, axis=0)
    if max_patches is not None and out.shape[0] > max_patches:
        keep = rng.choice(out.shape[0], size=max_patches, replace=False)
        out = out[keep]
    return out


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """What a training run actually did, for the results record."""

    epochs: int
    n_patches: int
    device: str
    seconds: float
    loss: list[float] = field(default_factory=list)
    mse: list[float] = field(default_factory=list)
    sam: list[float] = field(default_factory=list)
    peak_vram_gb: float | None = None

    def as_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "n_patches": self.n_patches,
            "device": self.device,
            "seconds": round(self.seconds, 2),
            "loss": [round(v, 8) for v in self.loss],
            "mse": [round(v, 8) for v in self.mse],
            "sam_rad": [round(v, 8) for v in self.sam],
            "peak_vram_gb": (
                None if self.peak_vram_gb is None else round(self.peak_vram_gb, 3)
            ),
        }


def train_autoencoder(
    patches: np.ndarray,
    cfg,
    reducer=None,
    device: str | None = None,
    seed: int = 0,
    progress: bool = False,
):
    """Train Fig. 1's network on background patches with Eq. 3's loss.

    Args:
        patches: ``(N, 1, C, p, p)`` training patches, from
            :func:`sample_corpus_patches`.
        cfg: :class:`AutoencoderConfig`.
        reducer: Fitted PCA, needed only when ``cfg.sam_domain == 'spectral'``
            so the SAM term can be taken in the reflectance domain.
        device: Overrides ``cfg.device`` resolution.
        seed: Seeds torch and the batch shuffle.
        progress: Print per-epoch losses.

    Returns:
        ``(model, history)``.
    """
    import torch

    from ..config import resolve_device, set_seeds

    if patches.ndim != 5 or patches.shape[1] != 1:
        raise IncompatibleInputError(
            f"Expected (N, 1, C, p, p) patches, got shape {patches.shape}."
        )
    n, _, n_components, ph, pw = patches.shape
    if ph != pw:
        raise IncompatibleInputError(f"Patches must be square, got {ph}x{pw}.")
    if n < 1:
        raise DegenerateInputError("No training patches were sampled.")

    set_seeds(seed, deterministic=True)
    dev = device or resolve_device(cfg.device)

    model = SpectralSpatialAutoencoder.build(
        n_components=n_components,
        patch_size=ph,
        latent_dim=cfg.latent_dim,
        encoder_channels=cfg.encoder_channels,
        kernel_spatial=cfg.kernel_spatial,
        kernel_spectral_depths=cfg.kernel_spectral_depths,
    ).to(dev)

    sam_domain = getattr(cfg, "sam_domain", "spectral")
    projector = None
    if cfg.lam > 0 and sam_domain == "spectral":
        if reducer is None:
            raise ModelNotFittedError(
                "sam_domain='spectral' evaluates Eq. 4 after back-projecting to "
                "reflectance, so train_autoencoder needs the fitted BandReducer. "
                "Pass reducer=, or set sam_domain='latent' to take the angle on PCA "
                "scores (see SpectralBackProjector for why that is the ablation)."
            )
        projector = SpectralBackProjector(
            reducer, whitened=cfg.whiten_input, device=dev
        )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    use_amp = bool(cfg.amp) and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    data = torch.as_tensor(patches, dtype=torch.float32)
    history = TrainingHistory(
        epochs=cfg.epochs, n_patches=n, device=dev, seconds=0.0
    )
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    g = torch.Generator().manual_seed(seed)
    start = time.perf_counter()
    model.train()
    for epoch in range(cfg.epochs):
        order = torch.randperm(n, generator=g)
        sums = np.zeros(3)
        batches = 0
        for lo in range(0, n, cfg.batch_size):
            batch = data[order[lo : lo + cfg.batch_size]].to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                recon = model(batch)
            # The loss is over centre-pixel spectra, not whole patches. The
            # patch is context for the convolution; the score in Eq. 1 is
            # per-pixel, so training on a whole-patch average would optimise a
            # different quantity than the one used at inference.
            total, mse, sam = composite_loss(
                model.centre_spectra(batch),
                model.centre_spectra(recon),
                cfg.lam,
                projector,
            )
            scaler.scale(total).backward()
            scaler.step(opt)
            scaler.update()
            sums += [total.item(), mse.item(), sam.item()]
            batches += 1

        history.loss.append(sums[0] / batches)
        history.mse.append(sums[1] / batches)
        history.sam.append(sums[2] / batches)
        if progress:
            print(
                f"  epoch {epoch + 1:>3}/{cfg.epochs}  loss {history.loss[-1]:.6f}  "
                f"mse {history.mse[-1]:.6f}  sam {history.sam[-1]:.6f} rad"
            )

    history.seconds = time.perf_counter() - start
    if dev == "cuda":
        history.peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    return model, history


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


def reconstruction_error_map(
    model,
    reduced: np.ndarray,
    patch_size: int,
    batch_size: int = 1024,
    device: str | None = None,
) -> np.ndarray:
    """Eq. 1 evaluated at every pixel, as an ``(H, W)`` anomaly score map.

    Scored on the **centre pixel** of each patch, matching what the loss
    optimised. Averaging the error over a whole patch would smear a small target
    across its neighbourhood and inflate the apparent detection of every pixel
    adjacent to it -- which reads as a better score without being one.
    """
    import torch

    if reduced.ndim != 3:
        raise IncompatibleInputError(
            f"reconstruction_error_map expects (H, W, C), got {reduced.shape}."
        )
    h, w, c = reduced.shape
    if c != model.n_components:
        raise IncompatibleInputError(
            f"Model was trained on {model.n_components} components but this cube has "
            f"{c}. Training and inference must share one PCA basis."
        )

    dev = device or next(model.parameters()).device
    patches = extract_patches(reduced, patch_size)
    data = torch.as_tensor(patches, dtype=torch.float32)

    errors = np.empty(h * w, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for lo in range(0, data.shape[0], batch_size):
            batch = data[lo : lo + batch_size].to(dev)
            recon = model(batch)
            x = model.centre_spectra(batch).cpu().numpy()
            xh = model.centre_spectra(recon).cpu().numpy()
            errors[lo : lo + batch.shape[0]] = reconstruction_error(x, xh)

    return errors.reshape(h, w)
