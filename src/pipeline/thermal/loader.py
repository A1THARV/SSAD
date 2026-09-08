"""Reading thermal frames, and deciding what they physically are (FR-9.7).

The load-time job that matters here is not parsing pixels -- it is classifying
each input as a **radiometric temperature field** or an **8-bit visualisation**,
and carrying that verdict on the data object so nothing downstream has to guess.

PRD §11 edge case 9 turns on this. A false-colour thermal JPEG looks exactly
like a temperature map to any code that treats it as an array: same shape, same
plausible-looking gradients, and a colormap that even puts the hot regions in
the right place. Inverting Planck's law over it produces confident, precise,
entirely fabricated temperatures. The only defence is refusing at the boundary,
which requires knowing which kind of file arrived.

FLAME 3 makes the distinction concrete: it ships **the same scene twice**, as a
radiometric ``Celsius TIFF`` (float32, one channel, real degrees) and as a
``Raw JPG`` (uint8 RGB, a colormap of that same field). Any classifier worth
having must separate those two, and the test suite checks exactly that pair --
separating two unrelated datasets would prove much less, since it could be
keying on anything incidental.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import DatasetConfig, ThermalPreprocessConfig
from ..errors import (
    DataUnavailableError,
    DegenerateInputError,
    IncompatibleInputError,
    MissingGroundTruthError,
    NonRadiometricInputError,
)

__all__ = [
    "Radiometry",
    "ThermalFrame",
    "classify_radiometry",
    "load_frame",
    "list_frames",
    "load_dataset",
    "KELVIN_OFFSET",
]

#: 0 °C in kelvin. Named because Eq. 5 is defined on absolute temperature and a
#: hard-coded 273.15 in the middle of a Planck call is a units bug waiting.
KELVIN_OFFSET = 273.15

#: TIFF tag 339 (SampleFormat); 3 means IEEE floating point.
_TIFF_SAMPLE_FORMAT = 339
_TIFF_FLOAT = 3

#: Physically implausible even for wildfire, so a "temperature" outside this is
#: evidence the array is not really temperature. Wide on purpose: the job is to
#: catch digital numbers masquerading as degrees, not to police real data.
_PLAUSIBLE_CELSIUS = (-100.0, 1500.0)

_IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def _looks_normalised(lo: float, hi: float) -> bool:
    """True when a float array is confined to the unit interval.

    The justification is physical rather than statistical. A thermal frame of a
    real outdoor scene has spatial contrast: sky against ground, shadow against
    sun, and for FLAME 3 a 500 K spread between fire and background. An entire
    640x512 frame whose values fit inside 0..1 C -- under one kelvin of contrast
    across the whole image -- is not a scene, it is data someone scaled to [0, 1]
    for display.

    Framing it this way also sets the bar in the right place. An earlier version
    fired only when the array *filled* [0, 1] to within 0.01, which missed
    anything normalised to a slightly narrower span and, worse, implied a real
    sub-kelvin scene would be fine as long as it did not quite reach the ends.
    """
    return 0.0 <= lo and hi <= 1.0


@dataclass(frozen=True)
class Radiometry:
    """What a thermal file physically carries, and the evidence for it.

    ``reason`` is not decoration. When the Planck module refuses an input, the
    user needs to know *why* this particular file was judged non-radiometric --
    "it has three channels" and "it is 8-bit" call for different fixes.

    Attributes:
        radiometric: True only when the array carries real per-pixel
            temperature. Digital numbers awaiting a documented gain/offset are
            **not** radiometric until that calibration is applied.
        units: ``celsius``, ``kelvin`` or ``digital_number``.
        reason: One clause naming the deciding evidence.
        detected: False when the verdict came from config rather than the file,
            so a report can distinguish measured provenance from asserted.
    """

    radiometric: bool
    units: str
    reason: str
    detected: bool = True

    def as_labels(self) -> dict[str, str]:
        """Provenance flags for the run record (PRD §7.3)."""
        return {
            "radiometric": str(self.radiometric).lower(),
            "thermal_units": self.units,
            "radiometry_basis": "detected" if self.detected else "asserted-by-config",
            "radiometry_reason": self.reason,
        }


def classify_radiometry(
    array: np.ndarray,
    mode: str | None = None,
    tiff_tags: dict | None = None,
    cfg: ThermalPreprocessConfig | None = None,
) -> Radiometry:
    """Decide whether an array is a temperature field or a picture of one.

    The rules, in order of authority:

    1. **More than one channel is a picture.** A temperature field is one scalar
       per pixel. Three channels means a colormap was applied, and a colormap is
       lossy and non-invertible -- two different temperatures map to the same
       RGB triple wherever the colormap turns. This is what rejects FLAME 3's
       ``Raw JPG`` while accepting its ``Celsius TIFF`` of the same scene.
    2. **Float means temperature.** A file that went to the trouble of storing
       IEEE floats is storing physical values; TIFF SampleFormat = 3 confirms it
       from the container rather than from the decoded dtype.
    3. **8-bit is a visualisation.** 256 levels across a scene spanning hundreds
       of degrees cannot carry useful temperature, whatever it is nominally of.
    4. **Wider integers are digital numbers.** Genuinely ambiguous: a 16-bit
       thermal frame may be raw sensor counts needing the documented gain and
       offset. Reported as ``digital_number`` and *not* radiometric, because an
       uncalibrated count is not a temperature.

    ``cfg.assume_radiometric`` can override a negative verdict for a dataset
    whose radiometry is documented but undetectable. It is deliberately unable
    to rescue multi-channel or 8-bit input: those are not calibration gaps, they
    are lost information, and no config flag brings it back.

    Args:
        array: The decoded pixel array.
        mode: PIL mode string, when the file came through PIL.
        tiff_tags: TIFF tag dictionary, when available.
        cfg: Supplies the ``assume_radiometric`` escape hatch.

    Returns:
        A :class:`Radiometry` verdict carrying its own justification.
    """
    assume = bool(cfg.assume_radiometric) if cfg is not None else False

    if array.ndim == 3 and array.shape[-1] > 1:
        return Radiometry(
            radiometric=False,
            units="digital_number",
            reason=(
                f"{array.shape[-1]}-channel image (PIL mode {mode or '?'}): a colormapped "
                "visualisation, not a scalar temperature field. The mapping is lossy and "
                "non-invertible, so no gain/offset can recover temperature from it."
            ),
        )
    if array.ndim != 2:
        raise DegenerateInputError(
            f"A thermal frame must be 2-D (H, W), got shape {array.shape}."
        )

    tag_says_float = False
    if tiff_tags:
        sample_format = tiff_tags.get(_TIFF_SAMPLE_FORMAT)
        if isinstance(sample_format, (tuple, list)):
            tag_says_float = _TIFF_FLOAT in sample_format
        else:
            tag_says_float = sample_format == _TIFF_FLOAT

    if np.issubdtype(array.dtype, np.floating) or tag_says_float:
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise DegenerateInputError(
                "Thermal frame holds no finite values, so it carries no temperature."
            )
        lo, hi = float(finite.min()), float(finite.max())
        if not (_PLAUSIBLE_CELSIUS[0] <= lo and hi <= _PLAUSIBLE_CELSIUS[1]):
            return Radiometry(
                radiometric=False,
                units="digital_number",
                reason=(
                    f"float array spanning {lo:.4g} to {hi:.4g}, outside the plausible "
                    f"{_PLAUSIBLE_CELSIUS[0]:g}..{_PLAUSIBLE_CELSIUS[1]:g} C range -- "
                    "stored as float but not degrees."
                ),
            )

        # Normalised display data is the one case the range test cannot catch,
        # because [0, 1] is a legitimate temperature range for a near-freezing
        # scene. A float export that fills [0, 1] almost exactly is far more
        # likely to be a picture scaled for display than a scene that happens to
        # span 0.00 to 1.00 C and nothing else.
        #
        # Genuinely ambiguous, so it is not decided silently: the module's rule
        # is that unproven radiometry is treated as absent, and
        # assume_radiometric is the documented way to overrule it.
        if _looks_normalised(lo, hi):
            if assume:
                return Radiometry(
                    radiometric=True,
                    units="celsius",
                    reason=(
                        f"float array spanning {lo:.4g}..{hi:.4g}, which looks normalised, "
                        "accepted as degrees because preprocess.assume_radiometric is true"
                    ),
                    detected=False,
                )
            return Radiometry(
                radiometric=False,
                units="digital_number",
                reason=(
                    f"float array filling {lo:.4g}..{hi:.4g} -- the signature of data "
                    "normalised for display, not degrees. If this really is a 0..1 C "
                    "scene, set preprocess.assume_radiometric to say so explicitly."
                ),
            )
        evidence = "TIFF SampleFormat=3 (IEEE float)" if tag_says_float else f"dtype {array.dtype}"
        return Radiometry(
            radiometric=True,
            units="celsius",
            reason=f"single-channel {evidence}, values {lo:.1f}..{hi:.1f} C",
        )

    if array.dtype == np.uint8:
        if assume:
            raise NonRadiometricInputError(
                "assume_radiometric: true was set on 8-bit input. That flag exists for "
                "datasets whose calibration is documented but undetectable, not to "
                "assert radiometry that the file cannot carry: 256 levels cannot encode "
                "a scene spanning hundreds of degrees. Supply the radiometric export "
                "(for FLAME 3, the Celsius TIFF rather than the Raw JPG)."
            )
        return Radiometry(
            radiometric=False,
            units="digital_number",
            reason=(
                "8-bit single-channel image: a display product. 256 levels cannot "
                "encode a temperature field at useful resolution."
            ),
        )

    return Radiometry(
        radiometric=bool(assume),
        units="celsius" if assume else "digital_number",
        reason=(
            f"{array.dtype} integer counts"
            + (
                " accepted as calibrated because preprocess.assume_radiometric is true"
                if assume
                else " -- raw digital numbers, not temperature until a documented "
                "gain/offset is applied"
            )
        ),
        detected=not assume,
    )


@dataclass
class ThermalFrame:
    """One thermal image plus what is known about it.

    Note the deliberate split between ``label`` and ``ground_truth``:

    * ``label`` is an **image-level** class -- does this frame contain fire?
    * ``ground_truth`` is a **pixel-level** mask.

    FLAME 3's CV subset supplies only the first. Collapsing them into one field
    would let an image-level scalar reach code expecting a mask, and the shape
    error would surface somewhere far from the cause -- so they are separate
    attributes and the accessors below refuse rather than improvise.
    """

    data: np.ndarray
    radiometry: Radiometry
    label: bool | None = None
    ground_truth: np.ndarray | None = None
    name: str = ""
    source_path: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise DegenerateInputError(
                f"{self.name or 'frame'}: expected a 2-D (H, W) thermal frame, got "
                f"shape {self.data.shape}."
            )
        if self.ground_truth is not None and self.ground_truth.shape != self.data.shape:
            raise IncompatibleInputError(
                f"{self.name or 'frame'}: mask is {self.ground_truth.shape} but the "
                f"frame is {self.data.shape}. These must share a grid."
            )
        self.labels = {**self.radiometry.as_labels(), **self.labels}

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def is_radiometric(self) -> bool:
        return self.radiometry.radiometric

    def require_radiometric(self) -> np.ndarray:
        """The temperature field, or a refusal naming why this file is not one.

        The single chokepoint for §11 edge case 9. Eq. 5's inversion calls this
        rather than touching ``data`` directly, so a display JPEG cannot reach
        Planck's law by any route.

        Raises:
            NonRadiometricInputError: the frame carries no true temperature.
        """
        if not self.radiometry.radiometric:
            raise NonRadiometricInputError(
                f"{self.name or self.source_path!r} is not radiometric, so a "
                "temperature cannot be retrieved from it.\n"
                f"  evidence: {self.radiometry.reason}\n"
                "  to fix  : supply the radiometric export of this scene, or set "
                "preprocess.assume_radiometric with a documented gain/offset if the "
                "calibration is known. Detection on this frame is still valid; only "
                "quantitative temperature is refused."
            )
        return self.data

    def to_kelvin(self) -> np.ndarray:
        """Absolute temperature, which is what Eq. 5 is defined on."""
        data = self.require_radiometric()
        if self.radiometry.units == "kelvin":
            return data
        return data + KELVIN_OFFSET

    def require_ground_truth(self) -> np.ndarray:
        """Pixel mask, or an explanation of why pixel metrics are unavailable."""
        if self.ground_truth is None:
            raise MissingGroundTruthError(
                f"{self.name or self.source_path!r} carries no pixel-level mask, so "
                "per-pixel detection metrics cannot be computed. FLAME 3's CV subset "
                "ships image-level fire/no-fire labels only -- use `label` and score "
                "at image level. The anomaly map is still valid; it is simply not "
                "validated per pixel."
            )
        return self.ground_truth

    def require_label(self) -> bool:
        if self.label is None:
            raise MissingGroundTruthError(
                f"{self.name or self.source_path!r} carries no image-level label."
            )
        return self.label


def _read_image(path: Path) -> tuple[np.ndarray, str | None, dict | None]:
    """Decode one image, returning the array plus its container metadata."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a hard dependency
        raise DataUnavailableError(
            reason="Pillow is required to read thermal images",
            path=path,
            remedy="pip install Pillow",
        ) from exc

    if not path.is_file():
        raise DataUnavailableError(
            reason="Thermal frame not found",
            path=path,
            remedy="Run scripts/download_data.py --verify to check the dataset landed.",
        )

    with Image.open(path) as im:
        mode = im.mode
        tags = dict(im.tag_v2) if hasattr(im, "tag_v2") else None
        array = np.array(im)
    return array, mode, tags


def load_frame(
    path: str | Path,
    cfg: ThermalPreprocessConfig | None = None,
    label: bool | None = None,
    name: str | None = None,
) -> ThermalFrame:
    """Read one thermal image and classify what it physically carries."""
    path = Path(path)
    array, mode, tags = _read_image(path)

    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]

    radiometry = classify_radiometry(array, mode=mode, tiff_tags=tags, cfg=cfg)

    data = array
    if data.ndim == 3:
        # Kept 2-D for the container's sake; the verdict already records that
        # this was a colormapped picture, so nothing downstream can mistake the
        # greyscale collapse for a temperature field.
        data = data.mean(axis=-1)

    return ThermalFrame(
        data=np.asarray(data, dtype=np.float32),
        radiometry=radiometry,
        label=label,
        name=name or path.stem,
        source_path=str(path),
    )


def list_frames(root: str | Path, pattern: str = "*") -> list[Path]:
    """Image files under ``root``, sorted, recursing into subdirectories."""
    root = Path(root)
    if not root.is_dir():
        raise DataUnavailableError(
            reason="Thermal dataset directory not found",
            path=root,
            remedy=(
                "Extract FLAME 3's CV subset to data/thermal/flame3_cv_subset, then "
                "run scripts/download_data.py --verify --only flame3."
            ),
        )
    found = sorted(
        p for p in root.rglob(pattern) if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not found:
        raise DataUnavailableError(
            reason=f"No thermal images found under {pattern!r}",
            path=root,
            remedy="Check the path and that extraction completed.",
        )
    return found


#: FLAME 3's CV subset layout. The label is the folder, because the dataset
#: ships no mask files -- see load_dataset's docstring.
FLAME3_SPLITS = {"Fire": True, "No Fire": False}

#: Which of FLAME 3's four per-quartet products to read. The Celsius TIFF is
#: the only radiometric one; the others are display products.
FLAME3_RADIOMETRIC_SUBDIR = Path("Thermal") / "Celsius TIFF"
FLAME3_DISPLAY_SUBDIR = Path("Thermal") / "Raw JPG"


def load_dataset(
    cfg: DatasetConfig,
    preprocess: ThermalPreprocessConfig | None = None,
    limit: int | None = None,
    subdir: Path | str = FLAME3_RADIOMETRIC_SUBDIR,
) -> list[ThermalFrame]:
    """Load FLAME 3's CV subset as labelled frames.

    **Labels are image-level, and that is a property of the data, not a
    simplification made here.** The CV subset ships no mask images and no label
    file -- only a ``Fire/`` and ``No Fire/`` split, so the folder *is* the
    annotation. M4 is therefore an image-level detection result, and per-pixel
    maps produced from these frames are reportable only qualitatively.

    Args:
        cfg: Supplies the dataset root and name.
        preprocess: Passed to radiometry classification.
        limit: Read at most this many frames per split, for quick runs.
        subdir: Which product to read from each split. Defaults to the
            radiometric TIFFs; point it at :data:`FLAME3_DISPLAY_SUBDIR` to load
            the display JPEGs, which is how the refusal path is exercised on
            real data rather than on a fixture.
    """
    root = Path(cfg.path)
    if not root.is_dir():
        raise DataUnavailableError(
            reason=f"Thermal dataset {cfg.name!r} not found",
            path=root,
            remedy="python scripts/download_data.py --verify --only flame3",
        )

    frames: list[ThermalFrame] = []
    for split, is_fire in FLAME3_SPLITS.items():
        split_dir = root / split / Path(subdir)
        if not split_dir.is_dir():
            raise DataUnavailableError(
                reason=f"Expected FLAME 3 split directory {split!r}/{subdir}",
                path=split_dir,
                remedy=(
                    "The CV subset should extract to "
                    "{Fire,No Fire}/{RGB,Thermal}/... -- if Kaggle nested it under an "
                    "extra folder, move the two split folders up one level."
                ),
            )
        paths = list_frames(split_dir)
        if limit is not None:
            paths = paths[:limit]
        for path in paths:
            frame = load_frame(path, cfg=preprocess, label=is_fire,
                               name=f"{split.replace(' ', '')}_{path.stem}")
            frame.labels["split"] = split
            frame.labels["label_granularity"] = "image"
            frames.append(frame)

    if not frames:
        raise DegenerateInputError(f"{cfg.name}: no frames loaded from {root}.")
    return frames
