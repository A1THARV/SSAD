"""Planck's law and emissivity-aware temperature retrieval (Eq. 5, FR-9.9).

Implements Eq. 5 of the paper:

    B(lambda, T) = (2 h c^2 / lambda^5) * 1 / (exp(h c / (lambda k T)) - 1)

the spectral radiance of a blackbody, in W m^-2 sr^-1 m^-1, and its inversion.

This is the thermal branch's substantive contribution, and the M4 result is the
argument for why. A threshold on absolute temperature detects fire perfectly
(RESULTS.md §11) but cannot say whether a warm pixel is steel or concrete --
those differ in *emissivity*, and separating emissivity from temperature is what
radiometry buys that detection demonstrably does not need.

Two things this module refuses to do, both deliberate:

* **Retrieve a temperature from non-radiometric input.** Routed through
  :meth:`ThermalFrame.require_radiometric`, so a colormapped JPEG cannot reach
  Planck's law (§11 edge case 9).
* **Return a point estimate for an unlisted material.** Emissivity is assumed,
  not measured, and the assumption dominates the error budget -- so the
  uncertainty travels with the answer rather than being dropped (§11 edge
  case 11).

Validated against ``astropy.modeling.physical_models.BlackBody`` rather than
against itself. Self-consistency cannot catch a wrong constant or a units error;
an independent implementation can, and that is the M5 exit criterion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import DegenerateInputError, IncompatibleInputError

__all__ = [
    "PLANCK_H",
    "SPEED_OF_LIGHT",
    "BOLTZMANN_K",
    "planck_radiance",
    "inverse_planck_temperature",
    "EmissivityEstimate",
    "EmissivityTable",
    "TemperatureRetrieval",
    "retrieve_temperature",
]

# CODATA 2018, SI units. Named rather than inlined because a wrong constant is
# invisible in the output -- the result is smooth, positive and plausible.
PLANCK_H = 6.62607015e-34          #: Planck constant, J s (exact by definition)
SPEED_OF_LIGHT = 2.99792458e8      #: Speed of light in vacuum, m s^-1 (exact)
BOLTZMANN_K = 1.380649e-23         #: Boltzmann constant, J K^-1 (exact)


def planck_radiance(wavelength_m: float | np.ndarray, temperature_k: float | np.ndarray):
    """Eq. 5: spectral radiance of a blackbody.

    Args:
        wavelength_m: Wavelength in **metres**. A thermal camera's 10 um band is
            ``10e-6``; passing ``10`` asks for radiance at 10 metres and returns
            a number so small it reads as zero rather than as an error, which is
            why the unit is in the parameter name.
        temperature_k: Absolute temperature in **kelvin**. Celsius is refused --
            a negative temperature is unphysical here and would otherwise
            produce a silently wrong exponent.

    Returns:
        Spectral radiance in W m^-2 sr^-1 m^-1, broadcast over the inputs.
    """
    lam = np.asarray(wavelength_m, dtype=np.float64)
    temp = np.asarray(temperature_k, dtype=np.float64)

    if np.any(lam <= 0):
        raise DegenerateInputError(
            "Wavelength must be positive and in metres (10 um is 1e-5 m, not 10)."
        )
    if np.any(temp <= 0):
        raise DegenerateInputError(
            "Planck's law needs absolute temperature in kelvin, and a value <= 0 K is "
            "unphysical. A negative value here usually means Celsius was passed -- "
            "ThermalFrame.to_kelvin() does the conversion."
        )

    # exp(x) - 1 loses precision catastrophically for small x, which is the
    # Rayleigh-Jeans limit (long wavelength, high temperature). expm1 is exact
    # there. It matters: at 10 um and 600 K, x ~ 2.4, but the same code serves
    # microwave wavelengths where x << 1.
    x = (PLANCK_H * SPEED_OF_LIGHT) / (lam * BOLTZMANN_K * temp)
    with np.errstate(over="ignore"):
        denominator = np.expm1(x)
    numerator = 2.0 * PLANCK_H * SPEED_OF_LIGHT**2 / lam**5
    return numerator / denominator


def inverse_planck_temperature(
    wavelength_m: float | np.ndarray, radiance: float | np.ndarray
):
    """Invert Eq. 5 for temperature, given radiance at one wavelength.

    Analytic rather than iterative -- Eq. 5 inverts in closed form:

        T = h c / (lambda k ln(1 + 2 h c^2 / (lambda^5 B)))

    Returns:
        Brightness temperature in kelvin: the temperature a *blackbody* would
        need to emit this radiance. For a real surface with emissivity < 1 this
        underestimates the true temperature, which is what
        :func:`retrieve_temperature` corrects for.
    """
    lam = np.asarray(wavelength_m, dtype=np.float64)
    rad = np.asarray(radiance, dtype=np.float64)

    if np.any(lam <= 0):
        raise DegenerateInputError("Wavelength must be positive and in metres.")
    if np.any(rad <= 0):
        raise DegenerateInputError(
            "Radiance must be positive; a non-positive value has no brightness "
            "temperature. This usually means a background-subtracted signal reached "
            "the inversion."
        )

    # log1p for the same reason expm1 appears above: the argument is small
    # wherever radiance is large, and log(1 + y) loses precision there.
    y = 2.0 * PLANCK_H * SPEED_OF_LIGHT**2 / (lam**5 * rad)
    return (PLANCK_H * SPEED_OF_LIGHT) / (lam * BOLTZMANN_K * np.log1p(y))


@dataclass(frozen=True)
class EmissivityEstimate:
    """An emissivity with its provenance and uncertainty attached.

    The uncertainty is not decoration. Emissivity is *assumed* in every mode
    this module supports, and that assumption dominates the error budget -- a
    0.95-versus-0.85 difference moves a retrieved temperature by several kelvin
    at LWIR. Returning a bare float would present an assumption as a
    measurement (§11 edge case 11).
    """

    value: float
    low: float
    high: float
    material: str
    source: str

    def __post_init__(self) -> None:
        if not 0.0 < self.value <= 1.0:
            raise DegenerateInputError(
                f"Emissivity must lie in (0, 1], got {self.value} for {self.material!r}."
            )
        if not (self.low <= self.value <= self.high):
            raise DegenerateInputError(
                f"Emissivity range [{self.low}, {self.high}] does not contain "
                f"{self.value} for {self.material!r}."
            )

    @property
    def is_assumed(self) -> bool:
        """True when no material-specific value was found."""
        return self.source == "default"

    def as_labels(self) -> dict[str, str]:
        return {
            "emissivity": f"{self.value:.3f}",
            "emissivity_range": f"{self.low:.3f}-{self.high:.3f}",
            "emissivity_material": self.material,
            "emissivity_source": self.source,
        }


class EmissivityTable:
    """Material-to-emissivity lookup with a documented fallback.

    An unlisted material does not get a point estimate. It gets the configured
    default *and* the configured range, flagged as assumed, so a report can say
    "0.95, assumed, plausibly 0.85-0.98" rather than "0.95".
    """

    def __init__(
        self,
        materials: dict[str, dict] | None = None,
        default: float = 0.95,
        default_range: tuple[float, float] = (0.85, 0.98),
    ) -> None:
        self.materials = materials or {}
        self.default = default
        self.default_range = tuple(default_range)
        if not self.default_range[0] <= default <= self.default_range[1]:
            raise DegenerateInputError(
                f"default_emissivity {default} lies outside its own stated range "
                f"{self.default_range}."
            )

    @classmethod
    def load(cls, path: str | Path, default: float = 0.95,
             default_range: tuple[float, float] = (0.85, 0.98)) -> "EmissivityTable":
        """Read a table, or fall back to defaults if none is present.

        A missing table is not an error. Emissivity references are optional
        supplements (change C4); without one every lookup simply returns the
        assumed default, correctly flagged.
        """
        path = Path(path)
        if not path.is_file():
            return cls(None, default, default_range)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload.get("materials", payload), default, default_range)

    def get(self, material: str | None) -> EmissivityEstimate:
        key = (material or "").strip().lower()
        entry = self.materials.get(key)
        if entry is None:
            return EmissivityEstimate(
                value=self.default,
                low=self.default_range[0],
                high=self.default_range[1],
                material=material or "unknown",
                source="default",
            )
        value = float(entry["emissivity"])
        lo, hi = entry.get("range", [value, value])
        return EmissivityEstimate(value, float(lo), float(hi), key, entry.get("source", "table"))


@dataclass(frozen=True)
class TemperatureRetrieval:
    """A retrieved temperature, the assumption behind it, and its spread.

    ``low_k``/``high_k`` come from propagating the emissivity *range* through
    the inversion, not from sensor noise. They answer "how much of this number
    is the emissivity assumption?", which for an unlisted material is most of
    the uncertainty.
    """

    temperature_k: float | np.ndarray
    brightness_temperature_k: float | np.ndarray
    low_k: float | np.ndarray
    high_k: float | np.ndarray
    emissivity: EmissivityEstimate
    wavelength_um: float
    mode: str

    @property
    def temperature_c(self):
        return self.temperature_k - 273.15

    @property
    def spread_k(self):
        """Width of the emissivity-driven uncertainty, in kelvin."""
        return np.asarray(self.high_k) - np.asarray(self.low_k)

    def as_labels(self) -> dict[str, str]:
        return {**self.emissivity.as_labels(), "retrieval_mode": self.mode,
                "retrieval_wavelength_um": f"{self.wavelength_um:g}"}


def retrieve_temperature(
    observed_temperature_k: float | np.ndarray,
    emissivity: EmissivityEstimate,
    wavelength_um: float = 10.0,
) -> TemperatureRetrieval:
    """Correct a brightness temperature for emissivity (FR-9.9).

    A thermal camera reports the temperature a *blackbody* would need to emit
    the radiance it measured. A real surface emits ``epsilon`` times as much, so
    the true surface temperature is **higher** than the reading. The correction
    inverts Eq. 5 on the scaled radiance:

        B_measured = epsilon * B(lambda, T_true)  ->  T_true = B^-1(B_measured / epsilon)

    The mode is named in the result. This is the **assumed-epsilon** mode: one
    band, one assumed emissivity. It is not TES and does not pretend to be --
    separating temperature from emissivity properly needs multiple bands, and
    FLAME 3 is single-band, so the assumption is unavoidable and is therefore
    reported rather than hidden.

    Args:
        observed_temperature_k: Brightness temperature from the camera, kelvin.
        emissivity: From :class:`EmissivityTable`, carrying its own range.
        wavelength_um: Band centre in micrometres.

    Returns:
        A :class:`TemperatureRetrieval` whose bounds come from the emissivity
        range, so the assumption's contribution is visible in the answer.
    """
    if wavelength_um <= 0:
        raise DegenerateInputError(f"wavelength_um must be positive, got {wavelength_um}.")
    lam = wavelength_um * 1e-6

    observed = np.asarray(observed_temperature_k, dtype=np.float64)
    if np.any(observed <= 0):
        raise DegenerateInputError(
            "Brightness temperature must be in kelvin and positive. Celsius input is "
            "the usual cause -- use ThermalFrame.to_kelvin()."
        )

    def _corrected(eps: float):
        if not 0.0 < eps <= 1.0:
            raise DegenerateInputError(f"Emissivity must lie in (0, 1], got {eps}.")
        return inverse_planck_temperature(lam, planck_radiance(lam, observed) / eps)

    true_t = _corrected(emissivity.value)
    # A *higher* emissivity means less correction, so it maps to the LOWER
    # bound. Getting this backwards would invert the error bars while leaving
    # the central estimate right, which no downstream check would catch.
    low = _corrected(emissivity.high)
    high = _corrected(emissivity.low)

    return TemperatureRetrieval(
        temperature_k=true_t,
        brightness_temperature_k=observed,
        low_k=low,
        high_k=high,
        emissivity=emissivity,
        wavelength_um=wavelength_um,
        mode="assumed-emissivity (single-band)",
    )
