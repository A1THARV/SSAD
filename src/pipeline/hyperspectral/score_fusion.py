"""Fuse the RX and reconstruction-error branches into one score map (FR-9.5).

Fig. 1's stage 5. The two branches produce scores on incompatible scales -- Eq.
2 is a squared Mahalanobis distance with units of "sigmas squared", Eq. 1 is an
L2 norm in PCA-score units -- so they are normalised before they are combined.
Skipping that step would not weight the branches, it would silently hand the
whole decision to whichever one happens to have the larger numeric range.
"""

from __future__ import annotations

import numpy as np

from ..errors import DegenerateInputError, IncompatibleInputError

__all__ = ["normalise_scores", "fuse_scores"]


def normalise_scores(scores: np.ndarray, method: str = "max") -> np.ndarray:
    """Map a score map onto ``[0, 1]`` so two branches can be compared.

    Args:
        scores: Any real-valued score map.
        method: ``max`` divides by the maximum, which is what Fig. 1 stage 5
            draws. ``minmax`` rescales the full range; ``rank`` replaces each
            score by its empirical percentile.

            ``max`` alone does not move the floor, so a branch whose scores sit
            in a narrow band high above zero -- RX on a homogeneous scene -- is
            compressed into the top of the range and contributes almost no
            discrimination after fusion. ``minmax`` fixes that but is more
            sensitive to a single outlying pixel. ``rank`` is scale-free and
            outlier-proof but discards magnitude, which is real information when
            one anomaly is far stronger than another. The default follows the
            figure; the alternatives exist to be measured against it.

    Returns:
        ``[0, 1]`` array of the input shape.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise DegenerateInputError("Cannot normalise an empty score map.")
    if not np.all(np.isfinite(arr)):
        raise DegenerateInputError(
            "Non-finite scores reached fusion. A NaN would propagate through the "
            "fused map and rank as a tie in AUC rather than as an error."
        )

    # Checked before dispatch, because the pathology is method-independent. A
    # map with no variance ranks nothing: every pixel is exactly as anomalous as
    # every other. `S / max(S)` would map it to 1.0 everywhere, and `max` fusion
    # would then let that branch veto the informative one at every pixel. Zeros
    # say "no evidence here" and cede the decision to the other branch.
    if float(arr.max()) == float(arr.min()):
        return np.zeros_like(arr)

    if method == "rank":
        order = np.argsort(arr.ravel(), kind="stable")
        ranks = np.empty(arr.size, dtype=np.float64)
        ranks[order] = np.arange(arr.size, dtype=np.float64)
        denom = max(arr.size - 1, 1)
        return (ranks / denom).reshape(arr.shape)

    if method == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
    elif method == "max":
        lo, hi = 0.0, float(np.abs(arr).max())
    else:
        raise DegenerateInputError(
            f"Unknown normalisation {method!r} (expected 'max', 'minmax' or 'rank')."
        )

    if hi - lo <= 0:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def fuse_scores(
    rx: np.ndarray,
    recon: np.ndarray,
    mode: str = "weighted",
    alpha: float = 0.5,
    normalisation: str = "max",
) -> np.ndarray:
    """Combine the two branches (Fig. 1 stage 5).

    Args:
        rx: Eq. 2 scores, ``(H, W)``.
        recon: Eq. 1 scores, ``(H, W)``, on the same grid.
        mode: ``weighted`` gives ``alpha*rx_n + (1-alpha)*recon_n``; ``max``
            takes the per-pixel maximum of the two normalised maps.

            The two answer different questions. ``weighted`` asks for agreement,
            so a target both branches half-detect can outrank one only RX finds.
            ``max`` asks for either, so it inherits the union of both branches'
            detections *and* the union of their false alarms.
        alpha: Weight on the RX branch. ``1.0`` is RX alone, ``0.0`` is the
            autoencoder alone -- which is how the ablation is run.
        normalisation: Passed to :func:`normalise_scores`.

    Returns:
        Fused ``(H, W)`` score map.
    """
    a = np.asarray(rx, dtype=np.float64)
    b = np.asarray(recon, dtype=np.float64)
    if a.shape != b.shape:
        raise IncompatibleInputError(
            f"Cannot fuse an RX map of shape {a.shape} with a reconstruction map of "
            f"shape {b.shape}. The two branches must score the same grid -- a "
            "mismatch usually means one of them dropped border pixels."
        )
    if not 0.0 <= alpha <= 1.0:
        raise DegenerateInputError(f"alpha must lie in [0, 1], got {alpha}.")

    an = normalise_scores(a, normalisation)
    bn = normalise_scores(b, normalisation)

    if mode == "weighted":
        return alpha * an + (1.0 - alpha) * bn
    if mode == "max":
        return np.maximum(an, bn)
    raise DegenerateInputError(
        f"Unknown fusion mode {mode!r} (expected 'weighted' or 'max')."
    )
