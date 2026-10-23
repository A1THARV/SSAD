"""G — camera-ready figures (INDICON revision).

    python scripts/revision_figures.py

Reviewer 1 found Figs. 2-3 unreadable; Reviewer 3 raised presentation. Both are
replaced with vector PDFs sized for a 3.5 in IEEE column, every text element at
8 pt or larger, in the Okabe-Ito colour-blind-safe palette.

* **Fig. 2** replaces the SAM infographic: top-1 species identification accuracy
  against target abundance, one line per band subset, at realistic
  wavelength-dependent SNR against the full 253-species library, with chance
  marked. The band x full-library x AVIRIS-SNR combination did not exist and is
  computed here -- the earlier ablation used the 15-material subset.
* **Fig. 3** replaces the thermal infographic: per-frame maximum temperature for
  fire against no-fire, with the separating gap shaded.
* **Fig. 4** is deleted, not redrawn.
* **Fig. 1** is being redrawn in TikZ elsewhere and is deliberately not
  generated here.

Figures are also written as PNG at 600 dpi for drafts; the PDF is what goes in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.runtime import configure_openmp  # noqa: E402

configure_openmp()

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sam_mixing_experiment import (  # noqa: E402
    ABUNDANCES,
    BAND_SUBSETS,
    background_spectrum,
    load_full_mineral_library,
    resample_library,
    run,
    sensor_wavelengths_nm,
    snr_weights,
)

#: Okabe-Ito, safe for deuteranopia/protanopia and distinguishable in greyscale
#: by the line styles paired with them below.
OKABE = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
         "orange": "#E69F00", "purple": "#CC79A7", "sky": "#56B4E9",
         "grey": "#555555"}

COL_IN = 3.5          # IEEE single column
FIG_SNR_DB = 20.0     # nominal, i.e. 20 dB at the SNR profile's peak
N_TRIALS = 250

#: R3 asks for vector figures with every label legible at 3.5 in. 8 pt is the
#: floor IEEE accepts, Type-42 keeps the text selectable and searchable rather
#: than outlined, and savefig.bbox="tight" stops LaTeX having to crop whitespace.
RC_PARAMS = {
    "pdf.fonttype": 42, "ps.fonttype": 42,     # embed as TrueType, not Type-3
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.3, "lines.markersize": 3.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}


def configure_matplotlib() -> None:
    """Select the headless backend and the paper's rcParams.

    Called from main() rather than at import, so that importing this module to
    reuse fig2_data() does not reach into another caller's matplotlib state.
    """
    matplotlib.use("Agg")
    matplotlib.rcParams.update(RC_PARAMS)


def save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, **kw)
        print(f"  wrote {path}")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. 2 data -- band subsets against the FULL library at realistic SNR
# --------------------------------------------------------------------------


def fig2_data(cache: Path) -> dict:
    if cache.is_file():
        print(f"  reusing {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))

    lib = load_full_mineral_library()
    if lib is None:
        raise SystemExit("ECOSTRESS library absent; run scripts/build_speclib.py")
    lib_nm, spectra, species, _ = lib
    sensor_nm = sensor_wavelengths_nm()
    bg, real_bg = background_spectrum(sensor_nm, target_mean=float(spectra.mean()))
    resampled = resample_library(spectra, lib_nm, sensor_nm)
    weights = snr_weights(sensor_nm)
    n_species = len(set(species))

    print(f"  {spectra.shape[0]:,} spectra, {n_species} species, "
          f"real background={real_bg}, SNR profile=AVIRIS-like at {FIG_SNR_DB:g} dB peak")
    data = {"abundances": list(ABUNDANCES), "n_species": n_species,
            "chance": 1.0 / n_species, "snr_db_peak": FIG_SNR_DB,
            "n_trials": N_TRIALS, "real_background": bool(real_bg), "subsets": {}}
    for subset in BAND_SUBSETS:
        n_bands = int(subset.mask(sensor_nm).sum())
        row = []
        for a in ABUNDANCES:
            row.append(run(lib_nm, spectra, species, sensor_nm, bg, subset, a,
                           FIG_SNR_DB, N_TRIALS, remove_background=True,
                           resampled=resampled, base_seed=0, band_weights=weights))
        data["subsets"][subset.name] = {"n_bands": n_bands, "accuracy": row}
        print(f"    {subset.name:<8} ({n_bands:>3} bands) " +
              " ".join(f"{v:.3f}" for v in row))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def draw_fig2(data: dict, out_dir: Path) -> None:
    style = {
        "swir":    (OKABE["vermillion"], "-",  "o", "SWIR (1500-2450 nm)"),
        "full":    (OKABE["blue"],       "--", "s", "all bands (455-2410 nm)"),
        "vnir":    (OKABE["green"],      "-.", "^", "VNIR (400-1000 nm)"),
        "visible": (OKABE["orange"],     ":",  "D", "visible (400-700 nm)"),
    }
    fig, ax = plt.subplots(figsize=(COL_IN, 2.5))
    a = np.array(data["abundances"])
    for name in ("swir", "full", "vnir", "visible"):
        if name not in data["subsets"]:
            continue
        colour, ls, mk, label = style[name]
        acc = np.array(data["subsets"][name]["accuracy"], dtype=float)
        n = data["subsets"][name]["n_bands"]
        ax.plot(a, acc, color=colour, linestyle=ls, marker=mk,
                label=f"{label}, {n} bands")

    ax.axhline(data["chance"], color=OKABE["grey"], linewidth=0.8, linestyle=(0, (1, 2)))
    ax.annotate(f"chance (1/{data['n_species']})",
                xy=(a.max(), data["chance"]), xytext=(-2, 3),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=7, color=OKABE["grey"])

    ax.set_xlabel("target abundance in the pixel")
    ax.set_ylabel("top-1 species accuracy")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.3", "0.5", "0.75", "1.0"])
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(0.045, 1.12)
    ax.legend(loc="upper left", handlelength=2.4, borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir, "fig2_identification_vs_abundance")


# --------------------------------------------------------------------------
# Fig. 3 -- per-frame maximum temperature, fire against no-fire
# --------------------------------------------------------------------------


def fig3_data(cache: Path) -> dict:
    if cache.is_file():
        print(f"  reusing {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))
    from PIL import Image
    root = Path("data/thermal/flame3_cv_subset")
    out = {}
    for split in ("Fire", "No Fire"):
        d = root / split / "Thermal" / "Celsius TIFF"
        vals = [float(np.array(Image.open(p)).max()) for p in sorted(d.glob("*.TIFF"))]
        out[split] = vals
        print(f"    {split:<8} {len(vals):>4} frames  "
              f"max-T {min(vals):7.2f} .. {max(vals):7.2f} C")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def draw_fig3(data: dict, out_dir: Path) -> None:
    fire = np.array(data["Fire"])
    calm = np.array(data["No Fire"])
    gap_lo, gap_hi = calm.max(), fire.min()

    fig, ax = plt.subplots(figsize=(COL_IN, 2.5))
    bins = np.linspace(0, max(fire.max(), calm.max()) * 1.02, 46)
    ax.hist(calm, bins=bins, color=OKABE["sky"], edgecolor="white",
            linewidth=0.3, label=f"no fire (n={calm.size})")
    ax.hist(fire, bins=bins, color=OKABE["vermillion"], edgecolor="white",
            linewidth=0.3, alpha=0.85, label=f"fire (n={fire.size})")

    ax.axvspan(gap_lo, gap_hi, color=OKABE["grey"], alpha=0.16, linewidth=0)
    ax.annotate(f"{gap_hi - gap_lo:.1f} $\\degree$C gap\nno overlap",
                xy=((gap_lo + gap_hi) / 2, ax.get_ylim()[1] * 0.62),
                ha="center", va="center", fontsize=7, color="#222222")
    for x, lab, ha in ((gap_lo, f"{gap_lo:.2f}", "right"), (gap_hi, f"{gap_hi:.2f}", "left")):
        ax.axvline(x, color=OKABE["grey"], linewidth=0.7, linestyle=(0, (2, 2)))
        ax.annotate(lab, xy=(x, ax.get_ylim()[1] * 0.97), xytext=(3 if ha == "left" else -3, 0),
                    textcoords="offset points", ha=ha, va="top", fontsize=7,
                    color="#222222")

    ax.set_xlabel("per-frame maximum temperature ($\\degree$C)")
    ax.set_ylabel("frames")
    ax.set_yscale("log")
    ax.legend(loc="upper right", borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir, "fig3_thermal_max_temperature")


# --------------------------------------------------------------------------
# Optional -- per-scene fused against RX
# --------------------------------------------------------------------------


def draw_scatter(stats_path: Path, out_dir: Path) -> None:
    if not stats_path.is_file():
        print("  (skipped: run scripts/revision_stats.py first)")
        return
    import collections
    import re

    from PIL import Image
    from pipeline.hyperspectral.score_fusion import fuse_scores
    from sklearn.metrics import roc_auc_score

    st = json.loads(stats_path.read_text(encoding="utf-8"))
    test = Path("data/hyperspectral/had100/HAD100/HAD100Dataset/test/aviris_ng")
    gtd = Path("data/hyperspectral/had100/HAD100/HAD100Dataset/ground_truth/aviris_ng")
    names = sorted(p.stem for p in test.glob("*.npy"))
    nameset = set(names)

    def src(n):
        m = re.match(r"^(.*)_([12])$", n)
        if m and f"{m.group(1)}_1" in nameset and f"{m.group(1)}_2" in nameset:
            return m.group(1)
        return n

    def maps(path):
        d = np.load(path)
        return [d[k] for k in sorted((k for k in d.files if k.startswith("m")),
                                     key=lambda k: int(k[1:]))]

    rx, ae = maps(Path(st["rx_maps"])), maps(Path(st["ae_maps"]))
    truths = []
    for n in names:
        arr = np.array(Image.open(gtd / f"{n}.png"))
        truths.append(((arr[:, :, 1] if arr.ndim == 3 else arr) > 0).ravel())

    acc = collections.defaultdict(lambda: ([], []))
    for n, r, a, g in zip(names, rx, ae, truths):
        if not (0 < g.sum() < g.size):
            continue
        f = fuse_scores(r, a, alpha=0.5)
        acc[src(n)][0].append(roc_auc_score(g, r.ravel()))
        acc[src(n)][1].append(roc_auc_score(g, f.ravel()))
    xs = np.array([np.mean(v[0]) for v in acc.values()])
    ys = np.array([np.mean(v[1]) for v in acc.values()])

    fig, ax = plt.subplots(figsize=(COL_IN, 2.6))
    lo = min(xs.min(), ys.min()) - 0.01
    ax.plot([lo, 1.002], [lo, 1.002], color=OKABE["grey"], linewidth=0.8,
            linestyle=(0, (3, 3)), zorder=1)
    better = ys > xs
    ax.scatter(xs[better], ys[better], s=11, color=OKABE["blue"],
               label=f"fusion better ({int(better.sum())})", zorder=3,
               edgecolor="white", linewidth=0.25)
    ax.scatter(xs[~better], ys[~better], s=11, color=OKABE["vermillion"],
               marker="^", label=f"RX better ({int((~better).sum())})", zorder=3,
               edgecolor="white", linewidth=0.25)
    ax.set_xlabel("RX per-scene AUC")
    ax.set_ylabel("fused per-scene AUC ($\\alpha$ = 0.5)")
    ax.set_xlim(lo, 1.003); ax.set_ylim(lo, 1.003)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", borderaxespad=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out_dir, "fig_scatter_fused_vs_rx")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="figures")
    p.add_argument("--data", default="results/revision")
    args = p.parse_args(argv)
    out_dir, data_dir = Path(args.out), Path(args.data)
    configure_matplotlib()

    print("Fig. 2 -- identification accuracy vs abundance")
    draw_fig2(fig2_data(data_dir / "g_fig2_data.json"), out_dir)
    print("\nFig. 3 -- thermal max-temperature histogram")
    draw_fig3(fig3_data(data_dir / "g_fig3_data.json"), out_dir)
    print("\nOptional -- fused vs RX per-scene scatter")
    draw_scatter(data_dir / "a2b_stats.json", out_dir)
    print("\nFig. 1 is redrawn in TikZ elsewhere and is not generated here.")
    print("Fig. 4 is deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
