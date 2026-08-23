"""Spectral Angle Mapper -- paper Eq. 4 (FR-9.6).

Implements Eq. 4:

.. math:: \\theta(x, r) = \\cos^{-1}\\left(
              \\frac{x \\cdot r}{\\lVert x \\rVert \\lVert r \\rVert}\\right)

the angle between a pixel spectrum and a reference spectrum treated as vectors.
Because it compares the *shape* of the spectral curve rather than its
magnitude, SAM is insensitive to illumination and topographic shading -- which
is why the paper (§III-B) picks it for outdoor mining and vegetation-stress
use.

This module is built ahead of the autoencoder despite being milestone M3 work,
because Eq. 3's composite loss ``L = MSE + lambda * SAM`` calls
:func:`spectral_angle`. The PRD's own build order has the dependency backwards.

Two properties make it usable both ways:

* :func:`spectral_angle` is differentiable-friendly and batched, so the
  training loop can call it per pixel without a Python loop;
* :func:`match_material` never returns a bare label. FR-9.6 requires angles
  alongside candidates, and §11 edge case 7 requires an explicit low-confidence
  flag when nothing in the library is close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import DegenerateInputError, IncompatibleInputError

__all__ = [
    "spectral_angle",
    "spectral_angle_batch",
    "spectral_angle_paired",
    "MaterialMatch",
    "match_material",
    "resample_to_grid",
]


def _as_2d(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise DegenerateInputError(
            f"{name} must be a spectrum (B,) or a stack of spectra (N, B), got shape "
            f"{arr.shape}."
        )
    if arr.shape[-1] < 2:
        raise DegenerateInputError(
            f"{name} has {arr.shape[-1]} band(s); a spectral angle needs at least 2."
        )
    if not np.all(np.isfinite(arr)):
        raise DegenerateInputError(
            f"{name} contains non-finite values, which would make the angle undefined."
        )
    return arr


def _unit(arr: np.ndarray, label: str) -> np.ndarray:
    """Normalise each row to unit length, refusing zero-length spectra."""
    norms = np.linalg.norm(arr, axis=1)
    if np.any(norms == 0):
        raise DegenerateInputError(
            f"An all-zero {label} spectrum has no direction, so its spectral angle "
            "is undefined. This usually means a masked or dead pixel reached SAM."
        )
    return arr / norms[:, None]


def spectral_angle_paired(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Element-wise spectral angles between two equal-length stacks (Eq. 4).

    This is the form Eq. 3's composite loss needs: one angle per pixel between
    a spectrum and its reconstruction, not a cross-product against a library.
    Cost is O(N*B) rather than the O(N*M*B) of
    :func:`spectral_angle_batch`, which matters when N is every pixel in a
    training batch.

    Args:
        x: ``(N, B)`` or ``(B,)`` spectra.
        r: ``(N, B)`` or ``(B,)`` matching spectra.

    Returns:
        ``(N,)`` angles in radians.
    """
    xs = _as_2d(x, "query spectrum")
    rs = _as_2d(r, "reference spectrum")

    if xs.shape[1] != rs.shape[1]:
        raise IncompatibleInputError(
            f"Paired spectra must share a wavelength grid, got {xs.shape[1]} and "
            f"{rs.shape[1]} bands."
        )
    if xs.shape[0] != rs.shape[0]:
        if xs.shape[0] == 1:
            xs = np.broadcast_to(xs, rs.shape)
        elif rs.shape[0] == 1:
            rs = np.broadcast_to(rs, xs.shape)
        else:
            raise IncompatibleInputError(
                f"Paired spectral angles need matching stack sizes, got {xs.shape[0]} "
                f"and {rs.shape[0]}."
            )

    u = _unit(np.asarray(xs), "query")
    v = _unit(np.asarray(rs), "reference")
    return 2.0 * np.arctan2(
        np.linalg.norm(u - v, axis=1), np.linalg.norm(u + v, axis=1)
    )


def spectral_angle_batch(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Spectral angles between stacks of spectra (Eq. 4), in radians.

    Computed as ``2*atan2(||u-v||, ||u+v||)`` on unit-normalised spectra rather
    than as ``arccos(u.v)``. The two are equal in exact arithmetic, but arccos
    is ill-conditioned near zero angle -- its derivative diverges there, so a
    cosine accurate to 1e-16 yields an angle accurate only to ~1e-8. That is
    precisely the regime Eq. 3's composite loss operates in, where a
    well-reconstructed patch has a near-zero angle, so the arccos form would
    inject numerical noise into the training signal. The atan2 form is
    well-conditioned across the whole range.

    Args:
        x: ``(B,)`` or ``(N, B)`` query spectra.
        r: ``(B,)`` or ``(M, B)`` reference spectra, on the *same* wavelength
            grid as ``x``.

    Returns:
        ``(N, M)`` angles in radians, within ``[0, pi]``.

    Raises:
        IncompatibleInputError: the two grids have different lengths. Library
            spectra ship on their own grids rather than the sensor's, so this
            is the common failure and it must not be silently broadcast away --
            resample with :func:`resample_to_grid` first.
        DegenerateInputError: a spectrum is all-zero, so it has no direction and
            no angle is defined.
    """
    xs = _as_2d(x, "query spectrum")
    rs = _as_2d(r, "reference spectrum")

    if xs.shape[1] != rs.shape[1]:
        raise IncompatibleInputError(
            f"Query spectra have {xs.shape[1]} bands but reference spectra have "
            f"{rs.shape[1]}. Spectral libraries are published on their own wavelength "
            "grids, not the sensor's -- resample the library to the sensor grid with "
            "resample_to_grid() before computing angles."
        )

    u = _unit(xs, "query")
    v = _unit(rs, "reference")

    # Cross-product form: (N, M, B). Fine for a library of tens of entries
    # against one query; use spectral_angle_paired for the per-pixel case.
    diff = np.linalg.norm(u[:, None, :] - v[None, :, :], axis=-1)
    summ = np.linalg.norm(u[:, None, :] + v[None, :, :], axis=-1)
    return 2.0 * np.arctan2(diff, summ)


def spectral_angle(x: np.ndarray, r: np.ndarray) -> float:
    """Spectral angle between two single spectra (Eq. 4), in radians.

    The scalar form named in the PRD §9 interface contract. See
    :func:`spectral_angle_batch` for the vectorised version used by Eq. 3's
    composite loss.
    """
    xs = _as_2d(x, "query spectrum")
    rs = _as_2d(r, "reference spectrum")
    if xs.shape[0] != 1 or rs.shape[0] != 1:
        raise DegenerateInputError(
            "spectral_angle compares two single spectra; use spectral_angle_batch "
            "for stacks."
        )
    return float(spectral_angle_batch(xs, rs)[0, 0])


def resample_to_grid(
    spectrum: np.ndarray,
    source_wavelengths_nm: np.ndarray,
    target_wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """Interpolate a library spectrum onto the sensor's wavelength grid.

    PRD v2 change C4 calls this "the real work" of the SAM step, and it is:
    ECOSTRESS spectra are published on their own grids, so without resampling
    :func:`spectral_angle` receives mismatched vector lengths.

    Extrapolation is refused rather than clamped. A library spectrum that does
    not span the sensor's range cannot be compared over the missing region, and
    edge-clamping would invent flat spectral shape there -- which SAM, being a
    shape comparison, would read as genuine similarity.

    Args:
        spectrum: ``(S,)`` reflectance or emissivity values.
        source_wavelengths_nm: ``(S,)`` grid the spectrum is on, ascending.
        target_wavelengths_nm: ``(B,)`` sensor grid to interpolate onto.

    Returns:
        ``(B,)`` resampled spectrum.
    """
    src_wl = np.asarray(source_wavelengths_nm, dtype=np.float64)
    tgt_wl = np.asarray(target_wavelengths_nm, dtype=np.float64)
    values = np.asarray(spectrum, dtype=np.float64)

    if values.shape != src_wl.shape:
        raise IncompatibleInputError(
            f"Spectrum has {values.shape[0]} values but its wavelength grid has "
            f"{src_wl.shape[0]} entries."
        )
    if src_wl.size < 2:
        raise DegenerateInputError("A spectrum needs at least 2 samples to interpolate.")
    if not np.all(np.diff(src_wl) > 0):
        order = np.argsort(src_wl)
        src_wl, values = src_wl[order], values[order]

    lo, hi = float(src_wl[0]), float(src_wl[-1])
    if tgt_wl.min() < lo or tgt_wl.max() > hi:
        raise IncompatibleInputError(
            f"Sensor grid spans {tgt_wl.min():.1f}-{tgt_wl.max():.1f} nm but the library "
            f"spectrum only covers {lo:.1f}-{hi:.1f} nm. Extrapolating would invent flat "
            "spectral shape outside the measured range, which SAM would read as real "
            "similarity. Restrict the sensor bands to the library's range, or use a "
            "library entry with the coverage you need."
        )
    return np.interp(tgt_wl, src_wl, values)


@dataclass(frozen=True)
class MaterialMatch:
    """One candidate identification for a detected anomaly (FR-9.6).

    Attributes:
        name: Library entry name.
        angle_rad: Spectral angle to the query. Smaller is more similar.
        low_confidence: ``True`` when ``angle_rad`` exceeds the configured
            threshold. §11 edge case 7 -- a match is never asserted when
            nothing in the library is close.
    """

    name: str
    angle_rad: float
    low_confidence: bool

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(self.angle_rad))


@dataclass
class MaterialIdentification:
    """The full result of matching one spectrum against a library.

    Carries the whole ranked shortlist rather than a single label, because
    FR-9.6 forbids returning a hard label with no confidence indication.

    Attributes:
        candidates: Top-k matches, nearest first.
        low_confidence: ``True`` when even the *best* match is beyond the
            threshold, i.e. the library plausibly does not contain this
            material at all.
        threshold_rad: The angle threshold applied.
        library_size: How many entries were searched, for the report.
    """

    candidates: list[MaterialMatch] = field(default_factory=list)
    low_confidence: bool = False
    threshold_rad: float = 0.25
    library_size: int = 0

    @property
    def best(self) -> MaterialMatch | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict:
        return {
            "candidates": [
                {
                    "name": c.name,
                    "angle_rad": c.angle_rad,
                    "angle_deg": c.angle_deg,
                    "low_confidence": c.low_confidence,
                }
                for c in self.candidates
            ],
            "low_confidence": self.low_confidence,
            "threshold_rad": self.threshold_rad,
            "library_size": self.library_size,
        }


def match_material(
    spectrum: np.ndarray,
    library: dict[str, np.ndarray],
    top_k: int = 3,
    low_confidence_angle_rad: float = 0.25,
) -> MaterialIdentification:
    """Rank library materials against a query spectrum (Eq. 4, FR-9.6).

    Args:
        spectrum: ``(B,)`` query, already on the library's wavelength grid.
        library: Mapping of material name to ``(B,)`` reference spectrum.
        top_k: How many candidates to return.
        low_confidence_angle_rad: Above this angle a match is flagged rather
            than asserted (§11 edge case 7).

    Returns:
        A :class:`MaterialIdentification` whose ``low_confidence`` is set when
        even the closest entry is beyond the threshold.

    Raises:
        DegenerateInputError: the library is empty or ``top_k`` is not positive.
        IncompatibleInputError: a library entry is on a different grid.
    """
    if not library:
        raise DegenerateInputError(
            "Material library is empty. Build the curated ECOSTRESS subset first "
            "(see scripts/download_data.py, key speclib_ecostress)."
        )
    if top_k < 1:
        raise DegenerateInputError(f"top_k must be >= 1, got {top_k}.")

    names = list(library)
    try:
        references = np.stack([np.asarray(library[n], dtype=np.float64) for n in names])
    except ValueError as exc:
        widths = {n: np.asarray(library[n]).shape[-1] for n in names}
        raise IncompatibleInputError(
            f"Library entries are not all on one wavelength grid: {widths}. Resample "
            "them to a common grid before matching."
        ) from exc

    angles = spectral_angle_batch(spectrum, references)[0]
    order = np.argsort(angles)[: min(top_k, len(names))]

    candidates = [
        MaterialMatch(
            name=names[i],
            angle_rad=float(angles[i]),
            low_confidence=bool(angles[i] > low_confidence_angle_rad),
        )
        for i in order
    ]
    return MaterialIdentification(
        candidates=candidates,
        low_confidence=bool(angles[order[0]] > low_confidence_angle_rad),
        threshold_rad=float(low_confidence_angle_rad),
        library_size=len(names),
    )
