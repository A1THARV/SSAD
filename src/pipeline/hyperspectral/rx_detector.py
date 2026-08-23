"""The Reed-Xiaoli anomaly detector (paper Eq. 2, FR-9.2).

Implements Eq. 2 of the paper:

.. math:: RX(x) = (x - \\mu)^T \\Sigma^{-1} (x - \\mu)

the squared Mahalanobis distance of a pixel from the background mean. It is the
statistical baseline every later result is measured against, and -- as GRX --
one of the 19 methods in HAD100's published table, so the same detector is on
both sides of that comparison.
"""

from __future__ import annotations

import numpy as np

from ..errors import DegenerateInputError

__all__ = ["rx_score", "background_statistics"]


def background_statistics(
    cube: np.ndarray,
    ridge: float = 1e-6,
    mask: np.ndarray | None = None,
    shrinkage: str = "ridge",
) -> tuple[np.ndarray, np.ndarray]:
    """Background mean and regularised inverse covariance.

    Regularisation here is not merely a guard against singularity -- it
    materially changes detection quality, by an amount that grows with
    dimensionality. RX weights each direction by ``1/lambda_i``, so trailing
    low-variance PCA components contribute enormous, noise-dominated terms
    unless they are damped. Under-regularising therefore *looks* like a penalty
    for using more components: an RX sweep at ``ridge=1e-6`` produces a
    spurious U-shaped accuracy-versus-dimensionality curve, which disappears
    once the ridge is set appropriately (see ROADMAP's ablation).

    Two modes:

    * ``ridge`` -- add ``ridge * trace(Sigma)/B`` to the diagonal. Scaled to
      the data's own magnitude so one setting spans scenes whose radiance units
      differ by orders of magnitude. The right value grows with ``B``, and
      differs between datasets, so it has to be declared.
    * ``ledoit_wolf`` -- analytically-derived shrinkage toward a scaled
      identity, with **no tunable parameter**. Preferred when a single setting
      must serve several datasets, because the tuned ridge does not transfer.
      Note it optimises covariance-estimation error rather than detection AUC,
      so a per-dataset tuned ridge can still edge it out.

    Args:
        cube: ``(H, W, B)`` or ``(N, B)``.
        ridge: Relative ridge strength; ignored when ``shrinkage`` is
            ``ledoit_wolf``.
        mask: Optional ``(H, W)`` or ``(N,)`` bool selecting background pixels.
            When omitted the whole scene is used, which is standard practice in
            unsupervised HAD -- anomalies are rare enough not to move the mean
            much.
        shrinkage: ``ridge`` or ``ledoit_wolf``.

    Returns:
        ``(mean, inv_cov)`` with shapes ``(B,)`` and ``(B, B)``.
    """
    arr = np.asarray(cube, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])

    if mask is not None:
        flat = flat[np.asarray(mask).ravel().astype(bool)]

    n_samples, n_bands = flat.shape
    if n_samples < 2:
        raise DegenerateInputError(
            f"RX needs at least 2 background pixels to estimate a covariance, got "
            f"{n_samples}."
        )

    if shrinkage == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        estimator = LedoitWolf(assume_centered=False).fit(flat)
        return estimator.location_, estimator.precision_
    if shrinkage != "ridge":
        raise DegenerateInputError(
            f"Unknown shrinkage {shrinkage!r} (expected 'ridge' or 'ledoit_wolf')."
        )

    mean = flat.mean(axis=0)
    centred = flat - mean
    cov = (centred.T @ centred) / (n_samples - 1)

    if ridge < 0:
        raise DegenerateInputError(f"ridge must be non-negative, got {ridge}.")

    scale = np.trace(cov) / n_bands
    if not np.isfinite(scale) or scale <= 0:
        raise DegenerateInputError(
            "Background covariance has zero or non-finite trace, so every pixel in the "
            "sample is identical. There is no background variation to score against."
        )
    cov.flat[:: n_bands + 1] += ridge * scale

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Fall back to the pseudo-inverse rather than failing: with the ridge
        # already applied, reaching here means a genuinely rank-deficient
        # scene, and the pseudo-inverse is the meaningful answer there.
        inv_cov = np.linalg.pinv(cov)

    return mean, inv_cov


def rx_score(
    cube: np.ndarray,
    ridge: float = 1e-6,
    background_mask: np.ndarray | None = None,
    shrinkage: str = "ridge",
) -> np.ndarray:
    """Per-pixel RX anomaly score (Eq. 2).

    Args:
        cube: ``(H, W, B)`` scene, typically PCA-reduced.
        ridge: Relative ridge added to the covariance diagonal. The default is
            deliberately weak, matching the classical definition; it is *not*
            the accuracy-optimal value at high dimensionality. See
            :func:`background_statistics`.
        background_mask: Optional ``(H, W)`` bool marking pixels to estimate
            background statistics from. Scores are always returned for every
            pixel, including those excluded from the estimate.
        shrinkage: ``ridge`` or ``ledoit_wolf``.

    Returns:
        ``(H, W)`` float64, higher meaning more anomalous. Never contains NaN:
        a non-finite score would propagate into metrics as a plausible-looking
        number, which PRD §10 forbids.

    Raises:
        DegenerateInputError: the cube is not 3-D, has too few pixels, or is
            spectrally constant.
    """
    arr = np.asarray(cube, dtype=np.float64)
    if arr.ndim != 3:
        raise DegenerateInputError(
            f"rx_score expects an (H, W, B) cube, got shape {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise DegenerateInputError(
            f"Cube contains {n_bad} non-finite value(s); RX would propagate them into "
            "the score map. Clean or mask them in preprocessing first."
        )

    height, width, n_bands = arr.shape
    mean, inv_cov = background_statistics(
        arr, ridge=ridge, mask=background_mask, shrinkage=shrinkage
    )

    centred = arr.reshape(-1, n_bands) - mean
    # einsum computes the quadratic form per pixel without materialising the
    # (N, N) product that `centred @ inv_cov @ centred.T` would build -- that
    # array is 10^8 elements even for a modest scene.
    scores = np.einsum("ij,jk,ik->i", centred, inv_cov, centred)

    # The quadratic form is non-negative in exact arithmetic; tiny negatives are
    # floating-point noise near zero and are clamped rather than left to look
    # like meaningful scores.
    scores = np.maximum(scores, 0.0)

    if not np.all(np.isfinite(scores)):
        raise DegenerateInputError(
            "RX produced non-finite scores despite finite input, which indicates a "
            "numerically degenerate covariance. Increase the ridge term."
        )
    return scores.reshape(height, width)
