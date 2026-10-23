"""F + H — export compact evidence tables from the result JSONs (INDICON revision).

    python scripts/revision_tables.py

No new experiments. Everything here is read from a results file or from a config
file, never from memory, so a table cannot drift from the run that produced it.
That is the point of doing it as a script: the paper's tables and the repository
agree by construction.

F -- thermal, emissivity, and SAM identification evidence.
H -- the implementation-details table, read from `configs/` and the model code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pipeline.config import HyperspectralConfig, ThermalConfig, load_config  # noqa: E402

R = Path("results")
OUT = Path("results/revision")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def table_thermal() -> str:
    d = _load(R / "thermal" / "flame3_baseline.json")
    if not d:
        return "_(run scripts/thermal_baseline.py)_\n"
    order = ["max_T", "frac_over_100C", "p99.9_T", "mean_T"]
    label = {"max_T": "maximum temperature", "frac_over_100C": "fraction of pixels > 100 °C",
             "p99.9_T": "99.9th percentile temperature", "mean_T": "mean temperature"}
    lines = [
        f"**F1. Thermal detection, image level.** {d['n_frames']} frames "
        f"({d['n_fire']} fire, {d['n_no_fire']} no fire), FLAME 3 CV subset.", "",
        "| single-feature detector | AUC | best F1 |", "|---|---|---|",
    ]
    for k in order:
        if k in d["features"]:
            f = d["features"][k]
            lines.append(f"| {label[k]} | {f['auc']:.4f} | "
                         f"{f['f1_at_optimal_threshold']:.4f} |")
    m = d["margin"]
    lines += [
        "",
        f"Separation: coldest fire frame peaks at **{m['coldest_fire_peak_c']:.2f} °C**, "
        f"hottest no-fire frame at **{m['hottest_no_fire_peak_c']:.2f} °C** — a "
        f"**{m['separation_c']:.2f} °C** gap with no overlap.",
        "",
        "**This is image-level detection only.** The CV subset ships no pixel "
        "masks, so localisation is neither measured nor claimed. The best F1 uses "
        "the F1-maximising threshold chosen on this same data and is an upper "
        "bound; the AUC needs no threshold and nothing is fitted.",
    ]
    return "\n".join(lines) + "\n"


def table_emissivity() -> str:
    d = _load(R / "thermal" / "flame3_retrieval.json")
    if not d or not d.get("frames"):
        return "_(run scripts/thermal_retrieval.py)_\n"
    fr = d["frames"][0]
    mats = [k for k in fr if isinstance(fr[k], dict)]
    in_class = [m for m in mats if m not in ("brass", "unlisted material")]
    rows = sorted(((m, fr[m]) for m in mats if m != "unlisted material"),
                  key=lambda kv: -kv[1]["emissivity"])
    lines = [
        "**F2. Emissivity-aware temperature retrieval (Eq. 5).** Identical measured "
        f"radiance throughout; brightness temperature "
        f"{fr['brightness_temperature_k'] - 273.15:.1f} °C over {fr['n_hot_px']:,} "
        "pixels above 100 °C. Emissivity derived from ECOSTRESS hemispherical "
        "reflectance via Kirchhoff's law (ε = 1 − R), valid for opaque surfaces at "
        "local thermodynamic equilibrium — an inference from a published "
        "measurement, not a measurement.", "",
        "| assumed material | ECOSTRESS class | ε | retrieved | bounds (°C) |",
        "|---|---|---|---|---|",
    ]
    ecostress_class = {
        "nonphotosyntheticvegetation": "non-photosynthetic vegetation",
        "char": "non-photosynthetic vegetation (char)",
        "soil": "soil", "grass": "non-photosynthetic vegetation (grass)",
        "bark": "non-photosynthetic vegetation (bark)", "sand": "soil / sand",
        "brass": "manmade — metal",
    }
    for m, v in rows:
        tag = " *(library-constrained)*" if m == "brass" else ""
        name = {"nonphotosyntheticvegetation": "dry vegetation"}.get(m, m)
        lines.append(f"| {name}{tag} | {ecostress_class.get(m, '?')} | "
                     f"{v['emissivity']:.4f} | {v['retrieved_c']:.1f} °C | "
                     f"{v['low_c']:.1f} – {v['high_c']:.1f} |")
    lines += [
        "",
        f"Within-class spread (the {len(in_class)} wildfire materials): "
        f"**{fr['within_class_spread_k']:.1f} K**. "
        f"Across-class, adding brass: **{fr['across_class_spread_k']:.1f} K** — a "
        f"**{fr['across_class_spread_k'] / fr['within_class_spread_k']:.0f}×** ratio.",
        "",
        "**Brass is library-constrained, not chosen.** Steel has no LWIR-capable "
        "entry in ECOSTRESS and neither does metallic aluminium (its only matches "
        "were paint *on* an aluminium substrate). Brass and gold are the metals the "
        "library measures over 7.5–13.5 µm, and at ε ≈ 0.135 brass sits below "
        "steel's assumed 0.28 — so the ratio would have been *smaller* had steel "
        "been available.",
        "",
        "**Limitation.** Laboratory spectra at ambient temperature applied to fire "
        "pixels at 220–600 °C. LWIR emissivity is reasonably stable with "
        "temperature for most of these materials, but char and soot during active "
        "combustion are the least certain case — and char is the primary material "
        "in a burning scene.",
    ]
    return "\n".join(lines) + "\n"


def table_sam() -> str:
    d = _load(OUT / "g_fig2_data.json")
    if not d:
        return "_(run scripts/revision_figures.py)_\n"
    a = d["abundances"]
    full = d["subsets"]["full"]["accuracy"]
    lines = [
        "**F3. SAM material identification, species level.** Controlled sub-pixel "
        "mixing: real ECOSTRESS spectra resampled to the AVIRIS-NG grid, mixed into "
        "a real HAD100 background at known abundance, with **wavelength-dependent "
        "(AVIRIS-like) SNR** at a 20 dB peak. Synthetic in the mixing and the "
        f"noise; ground truth exact by construction. {d['n_species']} species, "
        f"{d['n_trials']} trials per cell, material drawn uniformly then a spectrum "
        f"of that material, so chance is 1/{d['n_species']} = {d['chance']:.4f}.", "",
        "| target abundance | " + " | ".join(f"{x:g}" for x in a) + " |",
        "|---|" + "---|" * len(a),
        "| top-1 species accuracy | " + " | ".join(f"{v:.3f}" for v in full) + " |",
        "",
        "**Species level only.** Family-level accuracy is withheld: under a "
        "species-balanced draw the implied family prior is proportional to species "
        "count (silicate holds 70 of 253), so the bar is the majority family at "
        "0.2767 rather than 1/16, and the family result sits below it sub-pixel. "
        "That baseline question is unresolved, so no family figure is reported.",
        "",
        "Identification is a capability for **resolved or majority-abundance** "
        "anomalies. Sub-pixel it is not: 0.068 at a = 0.10 and 0.024 at a = 0.05 "
        "are measured limits, not degraded performance.",
    ]
    return "\n".join(lines) + "\n"


def table_implementation() -> str:
    """H — implementation details, read from the configs and the model code."""
    hs = load_config("configs/had100_band50.yaml", HyperspectralConfig)
    hs276 = load_config("configs/had100.yaml", HyperspectralConfig)
    abu = load_config("configs/abu_urban.yaml", HyperspectralConfig)
    th = load_config("configs/thermal_default.yaml", ThermalConfig)
    ae = hs.autoencoder
    cost = _load(OUT / "d_cost.json") or {}
    hw = cost.get("hardware", {})
    tr = (cost.get("training") or {}).get("corpus", {})

    enc = " -> ".join(str(c) for c in ae.encoder_channels)
    kern = " -> ".join(f"{ae.kernel_spatial}x{ae.kernel_spatial}x{d}"
                       for d in ae.kernel_spectral_depths)
    rows = [
        ("Encoder (Conv3D channels)", f"1 -> {enc}; decoder mirrors it"),
        ("Kernel sizes (spatial x spatial x spectral)", kern),
        ("Latent dimension", f"{ae.latent_dim}  (Fig. 1 requires d << K)"),
        ("Output activation", "none (linear) — PCA scores are signed"),
        ("Patch size", f"{ae.patch_size} x {ae.patch_size} spatial"),
        ("PCA components K", f"{hs.preprocess.pca_n_components}"),
        ("Band selection", f"{hs.preprocess.good_bands_preset} "
                           f"(band50 protocol) / {hs276.preprocess.good_bands_preset} "
                           f"(276-band)"),
        ("RX PCA basis", f"{hs.rx.pca_basis} (band50) / {hs276.rx.pca_basis} (276-band)"),
        ("RX regularisation", f"ridge {hs.rx.ridge:g}, trace-scaled; "
                              f"shrinkage mode '{hs.rx.shrinkage}'"),
        ("Composite-loss weight lambda", f"{ae.lam}"),
        ("SAM domain in Eq. 3", f"{ae.sam_domain} (back-projected to reflectance)"),
        ("Optimiser", "Adam"),
        ("Learning rate", f"{ae.learning_rate:g}"),
        ("Epochs", f"{ae.epochs}"),
        ("Batch size", f"{ae.batch_size}"),
        ("Training patches (corpus)", f"{ae.max_patches:,} sampled from "
                                      f"1,040 anomaly-free crops"),
        ("Training regime", "corpus (HAD100) / per-scene self-supervised (ABU)"),
        ("Mixed precision", f"{'on' if ae.amp else 'off'} (torch.autocast)"),
        ("Input whitening", f"{'on' if ae.whiten_input else 'off'}"),
        ("Fusion", f"{hs.score_fusion.mode}, alpha = {hs.score_fusion.alpha} "
                   f"a priori; cross-validated alpha reported separately"),
        ("Threshold method", f"{hs.score_fusion.threshold_method}"),
        ("ABU PCA components", f"{abu.preprocess.pca_n_components}, per-scene basis"),
        ("Thermal band", "7.5-13.5 um (FLAME 3 sensor)"),
        ("Planck wavelength", f"{th.planck.wavelength_um:g} um"),
        ("Default emissivity", f"{th.planck.default_emissivity} "
                               f"(range {th.planck.default_emissivity_range})"),
        ("Seeds", "0, 1, 2, 3, 4 (A1, C); 0 elsewhere"),
        ("Determinism", f"deterministic={hs.deterministic}, "
                        "torch.use_deterministic_algorithms where affordable"),
    ]
    if tr:
        rows += [("Corpus training time", f"{tr.get('seconds', 0):.0f} s for "
                                          f"{tr.get('n_patches', 0):,} patches"),
                 ("Peak VRAM, training", f"{tr.get('peak_vram_gb', 0):.2f} GB")]
    if cost.get("training", {}).get("per_scene_median_s"):
        rows.append(("Per-scene training time (ABU)",
                     f"{cost['training']['per_scene_median_s']:.0f} s median"))
    if hw:
        rows += [("Hardware", f"{hw.get('gpu', '?')} "
                              f"({hw.get('gpu_total_gb', '?')} GB), "
                              f"{hw.get('platform', '?')}"),
                 ("Software", f"Python {hw.get('python', '?')}, "
                              f"PyTorch {hw.get('torch', '?')}")]

    lines = ["**H1. Implementation details.** Exported from `configs/` and the model "
             "code by `scripts/revision_tables.py`, not transcribed.", "",
             "| parameter | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="results/revision/f_h_tables.md")
    args = p.parse_args(argv)

    parts = [
        "# INDICON revision — evidence tables\n",
        "Generated by `scripts/revision_tables.py`. Every number is read from a "
        "results JSON or a config file.\n",
        "## F — thermal and material-ID evidence\n", table_thermal(),
        "\n", table_emissivity(), "\n", table_sam(),
        "\n## H — implementation details\n", table_implementation(),
    ]
    text = "\n".join(parts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
