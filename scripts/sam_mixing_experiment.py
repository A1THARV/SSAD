"""M3 primary: controlled sub-pixel mixing, with exact identification ground truth.

    python scripts/sam_mixing_experiment.py

Objective 2 has to demonstrate two things: that SAM identifies what an anomaly is
made of, and -- per RESULTS.md §3 -- that it needs SWIR to do it. A real scene
gives one accuracy figure confounded by that scene's mineralogy and its
atmospheric correction. This gives the mechanism instead, and the ground truth is
**exact by construction**:

1. Take a real ECOSTRESS mineral spectrum from ``reflectance_swir.npz``.
2. Resample it to the AVIRIS-NG sensor grid (:func:`sam.resample_to_grid`, which
   refuses to extrapolate).
3. Mix it into a **real** HAD100 background spectrum at a known sub-pixel
   abundance: ``pixel = a*target + (1-a)*background``.
4. Add Gaussian noise at a known SNR.
5. Run SAM against the whole library and record whether the top-1 match is the
   right **material**.

The band-subset axis is the point. §3 measured that HAD100's *detection* signal
lives at 455-700 nm and predicted that *identification* needs the SWIR absorption
features a library encodes. Running the same experiment on a visible-only subset
against the full range turns that prediction into a measurement -- and it is a
measurement no real scene can supply, because a scene cannot be re-flown with
half its spectrum removed.

**This is synthetic, and is labelled synthetic everywhere it is reported.** The
spectra and the background are real; the mixing and the noise are not. That is
standard practice in the unmixing and SAM literature precisely because material
ground truth is otherwise unobtainable -- see RESULTS.md §14 for why Cuprite, the
usual answer, is currently half-unavailable.

Scoring is by MATERIAL, not by spectrum index: the library holds 51 quartz
entries, and asking a matcher to pick the one correct quartz out of 51 would
measure sample recall rather than material identification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import numpy as np  # noqa: E402

from pipeline.errors import DataUnavailableError  # noqa: E402
from pipeline.hyperspectral.band_presets import BAND_PRESETS  # noqa: E402
from pipeline.hyperspectral.sam import (  # noqa: E402
    resample_to_grid,
    spectral_angle_batch,
)

CURATED = Path("data/speclib/curated/reflectance_swir.npz")
ECOSTRESS = Path("data/speclib/ecostress/ecospeclib-all")
CORPUS = Path("data/hyperspectral/had100/HAD100Dataset/train/aviris_ng")
CORPUS_ALT = Path("data/hyperspectral/had100/HAD100/HAD100Dataset/train/aviris_ng")

#: AVIRIS-NG: 425 channels from 380 nm at ~5 nm spacing. HAD100 keeps a subset,
#: and these are the wavelengths of the bands its .npy crops actually carry.
AVIRIS_NG_START_NM = 380.0
AVIRIS_NG_STEP_NM = 5.0


def sensor_wavelengths_nm() -> np.ndarray:
    """Wavelengths of the 276 bands HAD100's AVIRIS-NG crops carry."""
    keep = BAND_PRESETS["had100_aviris_ng"]
    return AVIRIS_NG_START_NM + np.asarray(keep, dtype=np.float64) * AVIRIS_NG_STEP_NM


@dataclass(frozen=True)
class BandSubset:
    """A contiguous slice of the sensor's bands, and why it is interesting."""

    name: str
    lo_nm: float
    hi_nm: float
    rationale: str

    def mask(self, wavelengths: np.ndarray) -> np.ndarray:
        return (wavelengths >= self.lo_nm) & (wavelengths <= self.hi_nm)


#: The ablation §3 has been promising. `visible` is the window HAD100's
#: *detection* signal lives in; `swir` holds the absorption features a spectral
#: library encodes; `full` is everything.
BAND_SUBSETS = [
    BandSubset("full", 0.0, 1e9, "every band the sensor keeps (455-2410 nm)"),
    BandSubset("visible", 400.0, 700.0,
               "where §3 measured HAD100's detection signal to live"),
    BandSubset("vnir", 400.0, 1000.0, "visible plus near-IR, no SWIR"),
    BandSubset("swir", 1500.0, 2450.0,
               "OH/carbonate/sulfate absorptions -- what a library identifies on"),
]

ABUNDANCES = (1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05)
SNRS_DB = (np.inf, 40.0, 30.0, 20.0, 10.0)

#: SNRs the band-subset ablation runs at. Noiseless is deliberately excluded --
#: see the comment at the ablation itself.
ABLATION_SNRS_DB = (20.0, 10.0)

#: Reference-library sizes, in distinct materials. Identification difficulty
#: scales with how many candidates there are, so a figure quoted against a
#: 15-material subset is an upper bound on what a realistic library would give.
LIBRARY_SIZES = (5, 10, 15)

#: The curated subset holds 238 spectra but only 15 keyword-matched materials.
#: A realistic mineral library is far larger, and rather than bound the effect
#: with a caveat it is measured directly: every mineral/rock reflectance
#: spectrum in ECOSTRESS with full SWIR coverage, labelled by species.
FULL_MINERAL_LIBRARY = True


def baselines(labels_of_drawn_class) -> dict:
    """Chance and majority-class rates for a balanced draw over some OTHER label.

    The draw is balanced over **species**, so a species-level chance of
    1/n_species is honest. It is *not* honest at family level: families hold
    wildly different numbers of species (silicate 70, tungstate 1), so under a
    species-balanced draw P(family) is proportional to that count, and the bar a
    family-level result must clear is the **majority family**, not 1/n_families.

    Reporting 1/16 = 0.0625 for families would have repeated exactly the error
    caught at §14's species level, where a uniform chance rate hid a
    majority-class baseline the result sat below.
    """
    from collections import Counter

    hist = Counter(labels_of_drawn_class.values())
    total = sum(hist.values())
    top_label, top_n = hist.most_common(1)[0]
    return {
        "n_classes": len(hist),
        "uniform": 1.0 / len(hist),
        "majority": top_n / total,
        "majority_label": top_label,
        "histogram": dict(hist.most_common()),
    }


def family_of_species(species, families) -> dict:
    """Map each species to its family, so the draw's implied prior can be found."""
    from collections import Counter

    seen: dict[str, Counter] = {}
    for sp, fa in zip(species, families):
        seen.setdefault(sp, Counter())[fa] += 1
    return {sp: c.most_common(1)[0][0] for sp, c in seen.items()}


def load_library(path: Path = CURATED):
    if not path.is_file():
        raise DataUnavailableError(
            reason="Curated SWIR reflectance subset not found",
            path=path,
            remedy="python scripts/build_speclib.py",
        )
    blob = np.load(path, allow_pickle=True)
    return (
        blob["wavelength_um"] * 1000.0,           # nm
        blob["spectra"],
        np.array([str(m) for m in blob["material"]]),
        np.array([str(n) for n in blob["names"]]),
    )


def load_full_mineral_library(root: Path = ECOSTRESS, target_nm=None):
    """Every mineral/rock reflectance spectrum with full SWIR coverage.

    Labelled by species rather than by the 15 curated keywords, so the number of
    candidate materials reflects what a real reference library would present.
    Returns ``None`` if the raw library is absent, so the run degrades to the
    curated arm rather than failing.
    """
    if not root.is_dir():
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_speclib import classify_measurement, load_library as _load

    spectra, names, families = [], [], []
    for sp in _load(root):
        if classify_measurement(sp.metadata) != "reflectance":
            continue
        if not sp.covers((0.4, 2.5)):
            continue
        if not any(k in sp.metadata.get("Type", "").lower() for k in ("mineral", "rock")):
            continue
        label = (sp.metadata.get("Mineral") or sp.metadata.get("Name") or "").strip().lower()
        if not label:
            continue
        spectra.append(sp)
        names.append(label.split()[0].strip(".,"))
        # ECOSTRESS's own mineralogical grouping -- Silicate, Carbonate,
        # Sulfate, Oxide and so on, plus rock classes for rock samples. No hand
        # mapping required, and no chance of imposing a taxonomy of our own.
        families.append((sp.metadata.get("Class") or "unknown").strip().lower())
    if not spectra:
        return None

    grid = np.linspace(400.0, 2500.0, 512)
    stacked = np.vstack([
        np.interp(grid, sp.wavelength_um * 1000.0, sp.value) for sp in spectra
    ])
    return grid, stacked, np.array(names), np.array(families)


def background_spectrum(
    sensor_nm: np.ndarray, target_mean: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Mean spectrum of the HAD100 anomaly-free corpus, rescaled to match units.

    A real background from the real sensor, so the mixture is synthetic only in
    the mixing -- not in what is being mixed into. Falls back to a smooth
    synthetic continuum if the corpus is absent, flagged by the returned
    ``is_real`` so the report cannot quietly claim otherwise.

    **The rescaling is required, not cosmetic.** HAD100's AVIRIS-NG crops are not
    reflectance fractions: they span 0.058-7.19 with a mean of 2.51, against a
    library mean of 0.60. Mixed unscaled, a nominal ``a = 0.75`` pixel takes
    0.564 of its signal from the target and 0.626 from the "25%" background --
    so the abundance axis would be labelled with numbers that are not the
    fractions they claim to be, and identification accuracy would collapse at
    every a < 1 for a reason that has nothing to do with SAM.

    Scaling the background to the library's mean makes ``a`` mean what it says:
    the fraction of the pixel's signal contributed by the target. The spectral
    *shape* -- the only thing SAM sees -- is untouched.
    """
    for root in (CORPUS, CORPUS_ALT):
        if root.is_dir():
            paths = sorted(root.glob("*.npy"))[:40]
            if paths:
                stack = [np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
                         .reshape(-1, len(sensor_nm)).mean(axis=0) for p in paths]
                mean = np.mean(stack, axis=0)
                if np.all(np.isfinite(mean)) and mean.min() > 0:
                    return mean * (target_mean / float(mean.mean())), True
    rng = np.random.default_rng(seed)
    synthetic = 0.25 + 0.05 * np.sin(sensor_nm / 400.0) + rng.normal(0, 0.002, sensor_nm.size)
    return np.clip(synthetic, 1e-3, None), False


#: Relative SNR against wavelength for an AVIRIS-class imaging spectrometer,
#: as (nm, relative SNR) anchors. Uniform noise across bands is the generous
#: assumption for SWIR: a real instrument peaks in the visible/NIR and declines
#: through SWIR, with deep troughs where atmospheric water vapour absorbs at
#: ~1400 nm and ~1900 nm and almost no photons return.
#:
#: Shape only -- the absolute level is set by the run's nominal SNR, so a
#: "20 dB" run means 20 dB *at the peak* and less elsewhere.
AVIRIS_SNR_PROFILE = (
    (400.0, 0.55), (500.0, 0.95), (700.0, 1.00), (900.0, 0.95),
    (1100.0, 0.85), (1300.0, 0.70),
    (1400.0, 0.08),    # water vapour
    (1500.0, 0.55), (1700.0, 0.60), (1800.0, 0.50),
    (1900.0, 0.05),    # water vapour
    (2000.0, 0.45), (2200.0, 0.45), (2400.0, 0.30), (2500.0, 0.25),
)


def snr_weights(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Per-band relative SNR from :data:`AVIRIS_SNR_PROFILE`."""
    anchors_nm = np.array([a for a, _ in AVIRIS_SNR_PROFILE])
    anchors_w = np.array([w for _, w in AVIRIS_SNR_PROFILE])
    return np.interp(np.asarray(wavelengths_nm, dtype=np.float64), anchors_nm, anchors_w)


def add_noise(spectrum: np.ndarray, snr_db: float, rng, weights=None) -> np.ndarray:
    """Additive Gaussian noise at a stated SNR, defined on signal power.

    ``weights`` gives per-band relative SNR. Without it the noise is uniform
    across bands, which is the simpler assumption and the one that **favours
    SWIR**: it grants the SWIR bands the same signal quality as the visible,
    where a real spectrometer delivers considerably less. The uniform result is
    therefore an upper bound on the SWIR advantage, and the weighted run is what
    bounds it from the other side.
    """
    if not np.isfinite(snr_db):
        return spectrum
    signal_power = float(np.mean(spectrum**2))
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_power)
    if weights is not None:
        # Higher relative SNR -> proportionally less noise in that band.
        sigma = sigma / np.clip(np.asarray(weights, dtype=np.float64), 1e-3, None)
    return spectrum + rng.normal(0.0, 1.0, spectrum.shape) * sigma


def cell_seed(subset_name, abundance, snr_db, remove_background, n_materials, base):
    """A seed determined by WHAT is being measured, not by evaluation order.

    Every cell drew from one shared RNG stream at first, which meant a cell's
    result depended on how many cells had run before it -- so the same
    configuration appearing in two tables gave two different numbers (0.404 and
    0.364 for full range at a=0.10, 20 dB), and adding a table silently changed
    every number after it. Deriving the seed from the cell's own parameters
    makes identical configurations identical wherever they appear, and makes the
    whole run independent of table order.
    """
    # hashlib, NOT Python's hash(): the builtin is randomised per process for
    # strings unless PYTHONHASHSEED is set, so seeding from it would make every
    # run produce different numbers -- reintroducing the irreproducibility this
    # function exists to remove, in a form that is harder to notice.
    key = "|".join(str(x) for x in (
        subset_name, round(float(abundance), 6),
        "inf" if not np.isfinite(snr_db) else round(float(snr_db), 3),
        bool(remove_background), int(n_materials), int(base),
    ))
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def run(
    library_nm, library, materials, sensor_nm, background, subset, abundance,
    snr_db, n_trials, rng=None, remove_background: bool = True, resampled=None,
    base_seed: int = 0, band_weights=None,
):
    """Top-1 material identification accuracy for one experimental cell.

    ``remove_background`` selects between the two things "run SAM on an anomaly"
    can mean, and they behave completely differently:

    * **False** -- match the raw mixed pixel against pure library spectra. This
      fails by construction for any minority target. As ``a`` falls, every
      mixture converges on the background *regardless of what was mixed in*, so
      the nearest library entry becomes whichever one happens to sit closest to
      the background, the same one every trial. It is not a flaw in SAM; it is
      what a spectral angle measures.
    * **True** -- match the *deviation* from background, on both sides:
      ``observed - bg`` against ``library_i - bg``. This is what the pipeline
      actually has available, because RX defines an anomaly as a departure from
      background in the first place, and it is what makes sub-pixel
      identification possible at all.

    The pair is the experiment. Reporting only the second would overstate SAM;
    reporting only the first would understate it and misdescribe the pipeline.
    """
    mask = subset.mask(sensor_nm)
    if mask.sum() < 2:
        return None

    rng = np.random.default_rng(
        cell_seed(subset.name, abundance, snr_db, remove_background,
                  len(set(materials)), base_seed + (7919 if band_weights is not None else 0))
    )
    weights = None if band_weights is None else band_weights[mask]

    if resampled is None:
        resampled = np.vstack(
            [resample_to_grid(library[i], library_nm, sensor_nm)
             for i in range(library.shape[0])]
        )
    ref = resampled[:, mask]
    bg = background[mask]
    if remove_background:
        ref = ref - bg

    # Draw the MATERIAL uniformly, then a spectrum of that material -- not a
    # spectrum uniformly. The subset is heavily imbalanced (feldspar 55, quartz
    # 51, goethite 1), so drawing over spectra would make the trivial
    # majority-class strategy score 55/238 = 0.231 and quietly set the bar
    # there. Balancing makes 1/n_materials the honest chance rate.
    by_material = {}
    for i, m in enumerate(materials):
        by_material.setdefault(m, []).append(i)
    classes = sorted(by_material)

    hits = 0
    for _ in range(n_trials):
        material = classes[int(rng.integers(len(classes)))]
        idx = int(rng.choice(by_material[material]))
        target = resampled[idx][mask]
        mixed = abundance * target + (1.0 - abundance) * bg
        observed = add_noise(mixed, snr_db, rng, weights)
        if remove_background:
            observed = observed - bg
        if not np.all(np.isfinite(observed)) or np.linalg.norm(observed) == 0:
            continue
        angles = spectral_angle_batch(observed, ref)[0]
        hits += int(materials[int(np.argmin(angles))] == material)
    return hits / n_trials


def resample_library(library, library_nm, sensor_nm):
    """Resample every library spectrum onto the sensor grid, once."""
    return np.vstack(
        [resample_to_grid(library[i], library_nm, sensor_nm)
         for i in range(library.shape[0])]
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/sam_mixing_experiment.json")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    library_nm, library, materials, _ = load_library()
    sensor_nm = sensor_wavelengths_nm()
    background, real_background = background_spectrum(
        sensor_nm, target_mean=float(library.mean())
    )

    print(f"library : {library.shape[0]} spectra, {len(set(materials))} materials, "
          f"{library_nm[0]:.0f}-{library_nm[-1]:.0f} nm")
    print(f"sensor  : {sensor_nm.size} bands, {sensor_nm[0]:.0f}-{sensor_nm[-1]:.0f} nm")
    print(f"backgrnd: {'real HAD100 corpus mean' if real_background else 'SYNTHETIC continuum'}"
          f", rescaled to the library's mean ({library.mean():.3f}) so that abundance "
          f"means signal fraction")
    print(f"trials  : {args.trials} per cell\n")

    rng = np.random.default_rng(args.seed)
    resampled = resample_library(library, library_nm, sensor_nm)
    report = {
        "synthetic": True,
        "trials_per_cell": args.trials,
        "n_library_spectra": int(library.shape[0]),
        "n_materials": len(set(materials)),
        "real_background": bool(real_background),
        "abundances": list(ABUNDANCES),
        "snr_db": [None if not np.isfinite(s) else s for s in SNRS_DB],
        "band_subsets": {b.name: {"lo_nm": b.lo_nm, "hi_nm": b.hi_nm,
                                  "rationale": b.rationale} for b in BAND_SUBSETS},
        "accuracy": {},
    }

    def table(title, key, rows, fn):
        print(f"\n{title}")
        print("  " + " " * 12 + "".join(f"{a:>8.2f}" for a in ABUNDANCES)
              + "    <- target abundance")
        for label, kwargs in rows:
            row = [fn(kwargs, a) for a in ABUNDANCES]
            for a, v in zip(ABUNDANCES, row):
                report["accuracy"].setdefault(key, {})[f"{label}|a={a}"] = v
            print(f"  {label:<12}" + "".join(f"{v:>8.3f}" for v in row))

    full = BAND_SUBSETS[0]

    # --- The two meanings of "run SAM on an anomaly" ------------------------
    table(
        "Raw mixed pixel against pure library spectra (no background removal)",
        "raw",
        [("full range", {"subset": full, "remove_background": False})],
        lambda kw, a: run(library_nm, library, materials, sensor_nm, background,
                          kw["subset"], a, np.inf, args.trials,
                          remove_background=False, resampled=resampled, base_seed=args.seed),
    )

    # --- The headline ablation, on the pipeline's actual formulation --------
    #
    # Run UNDER NOISE, and that is not a detail. Noiseless, this experiment is
    # exactly solvable: observed - bg = a*(target - bg) is the reference scaled
    # by a, and SAM is magnitude-invariant, so the angle to the correct entry is
    # identically zero and every band subset scores 1.000. A noiseless ablation
    # would have reported perfect accuracy on visible-only bands and appeared to
    # REFUTE §3, when in fact it measured nothing at all.
    #
    # Noise is what makes band choice matter: it is the diagnostic absorption
    # features that survive it, and a subset lacking them has only continuum
    # shape to match on.
    for snr in ABLATION_SNRS_DB:
        table(
            f"Background-removed matching by band subset, SNR {snr:.0f} dB"
            "  [the §3 ablation]",
            f"band_subset_snr{snr:.0f}",
            [(b.name, {"subset": b}) for b in BAND_SUBSETS],
            lambda kw, a, _snr=snr: run(
                library_nm, library, materials, sensor_nm, background,
                kw["subset"], a, _snr, args.trials,
                remove_background=True, resampled=resampled, base_seed=args.seed),
        )
    for b in BAND_SUBSETS:
        report["band_subsets"][b.name]["n_bands"] = int(b.mask(sensor_nm).sum())

    # --- Noise sensitivity, full range, background removed ------------------
    table(
        "Background-removed matching versus SNR, full range",
        "snr",
        [("noiseless" if not np.isfinite(s) else f"{s:.0f} dB", {"snr": s})
         for s in SNRS_DB],
        lambda kw, a: run(library_nm, library, materials, sensor_nm, background,
                          full, a, kw["snr"], args.trials,
                          remove_background=True, resampled=resampled, base_seed=args.seed),
    )

    # --- Library size: how many candidate materials must be told apart -----
    print("\nBackground-removed matching versus REFERENCE LIBRARY SIZE, "
          "full range, SNR 20 dB")
    print("  " + " " * 12 + "".join(f"{a:>8.2f}" for a in ABUNDANCES)
          + "    <- target abundance")
    all_materials = sorted(set(materials))
    for size in LIBRARY_SIZES:
        if size > len(all_materials):
            continue
        keep_m = set(all_materials[:size])
        sel = np.array([m in keep_m for m in materials])
        row = []
        for a in ABUNDANCES:
            acc = run(library_nm, library[sel], materials[sel], sensor_nm, background,
                      full, a, 20.0, args.trials, remove_background=True,
                      resampled=resampled[sel], base_seed=args.seed)
            row.append(acc)
            report["accuracy"].setdefault("library_size", {})[f"{size}|a={a}"] = acc
        print(f"  {size:>2} materials" + "".join(f"{v:>8.3f}" for v in row)
              + f"   (chance {1/size:.3f})")

    # --- The realistic-library arm: measured, not extrapolated -------------
    full_lib = load_full_mineral_library() if FULL_MINERAL_LIBRARY else None
    if full_lib is not None:
        fl_nm, fl_spectra, fl_species, fl_families = full_lib
        n_species = len(set(fl_species))
        n_families = len(set(fl_families))
        fl_resampled = resample_library(fl_spectra, fl_nm, sensor_nm)
        fl_bg, _ = background_spectrum(sensor_nm, target_mean=float(fl_spectra.mean()))
        weights = snr_weights(sensor_nm)

        print(f"\nFULL MINERAL LIBRARY: {fl_spectra.shape[0]:,} spectra, "
              f"{n_species} species, {n_families} families "
              f"(ECOSTRESS 'Class': silicate, carbonate, sulfate, oxide, ...)")
        print("  " + " " * 26 + "".join(f"{a:>8.2f}" for a in ABUNDANCES)
              + "    <- target abundance")

        # The draw is balanced over SPECIES, so the family prior is whatever
        # that implies -- not uniform. Silicates hold 70 species, tungstate 1.
        fam_base = baselines(family_of_species(fl_species, fl_families))
        print(f"  family baselines under a species-balanced draw: "
              f"uniform 1/{fam_base['n_classes']} = {fam_base['uniform']:.4f}, "
              f"MAJORITY ('{fam_base['majority_label']}') = {fam_base['majority']:.4f}")
        print(f"  -> a family result must clear {fam_base['majority']:.4f}, not "
              f"{fam_base['uniform']:.4f}")

        arms = [
            # label,                  labels,      weights, chance to beat
            ("species, uniform SNR",  fl_species,  None,    1 / n_species),
            ("species, AVIRIS SNR",   fl_species,  weights, 1 / n_species),
            ("FAMILY, uniform SNR",   fl_families, None,    fam_base["majority"]),
            ("FAMILY, AVIRIS SNR",    fl_families, weights, fam_base["majority"]),
        ]
        for label, labels, w, chance in arms:
            row = []
            for a in ABUNDANCES:
                acc = run(fl_nm, fl_spectra, labels, sensor_nm, fl_bg, full, a,
                          20.0, args.trials, remove_background=True,
                          resampled=fl_resampled, base_seed=args.seed,
                          band_weights=w)
                row.append(acc)
                report["accuracy"].setdefault("full_library", {})[f"{label}|a={a}"] = acc
            beats = "".join("  " if v is None else ("+" if v > chance else "-")
                            for v in row)
            print(f"  {label:<24}" + "".join(f"{v:>8.3f}" for v in row)
                  + f"   (bar {chance:.4f})  {beats}")

        report["full_library_meta"] = {
            "n_spectra": int(fl_spectra.shape[0]),
            "n_species": int(n_species),
            "n_families": int(n_families),
            "chance_species": 1.0 / n_species,
            "family_baseline_uniform": fam_base["uniform"],
            "family_baseline_majority": fam_base["majority"],
            "family_majority_label": fam_base["majority_label"],
            "family_histogram": fam_base["histogram"],
            "family_source": "ECOSTRESS 'Class' field",
            "snr_profile": "AVIRIS-like: peak VNIR, declining SWIR, "
                           "troughs at 1400 and 1900 nm water vapour",
        }

    # --- What the band ablation looks like under a real SNR profile --------
    print("\nBand ablation under WAVELENGTH-DEPENDENT SNR (nominal 20 dB at peak)")
    print("  " + " " * 12 + "".join(f"{a:>8.2f}" for a in ABUNDANCES)
          + "    <- target abundance")
    weights_curated = snr_weights(sensor_nm)
    for b in BAND_SUBSETS:
        row = []
        for a in ABUNDANCES:
            acc = run(library_nm, library, materials, sensor_nm, background, b, a,
                      20.0, args.trials, remove_background=True,
                      resampled=resampled, base_seed=args.seed,
                      band_weights=weights_curated)
            row.append(acc)
            report["accuracy"].setdefault("band_subset_aviris_snr", {})[f"{b.name}|a={a}"] = acc
        print(f"  {b.name:<12}" + "".join(f"{v:>8.3f}" for v in row))

    n_materials = len(set(materials))
    counts = {m: int((materials == m).sum()) for m in set(materials)}
    majority = max(counts.values()) / len(materials)
    report["chance_balanced"] = 1.0 / n_materials
    report["majority_class_rate_unbalanced"] = majority
    report["material_counts"] = counts
    report["sampling"] = "material drawn uniformly, then a spectrum of that material"
    print(f"\nbaselines with {n_materials} materials:")
    print(f"  chance, balanced draw (this experiment)   {1 / n_materials:.3f}")
    print(f"  majority class, had the draw been over    {majority:.3f}")
    print(f"    spectra instead (feldspar is 23% of them)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
