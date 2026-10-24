"""E — the 20-row benchmark ranking with method categories (INDICON revision).

    python scripts/revision_benchmark_table.py

Reviewer 3 asks that performance claims accurately reflect the benchmark. Two
things were wrong, and both are corrected here from the benchmark paper itself
rather than inferred from acronyms.

1. THE CITATION. RESULTS.md, the README and two scripts attributed DOI
   10.1109/TGRS.2023.3258067 to "Li et al., *Hyperspectral Anomaly Detection: A
   Survey and Benchmark*". The DOI is correct; **the title is not**. It resolves
   to Li, Wang, Xiao, Ling, Lin and An, *You Only Train Once: Learning a General
   Anomaly Enhancement Network With Random Masks for Hyperspectral Anomaly
   Detection*, IEEE TGRS 61, 2023, pp. 1-18 (arXiv:2303.18001). HAD100 is
   introduced in that paper, not in a survey.

2. "AHEAD OF EVERY DEEP-LEARNING ENTRY" IS FALSE. The paper's own Fig. 1 splits
   the field into "Non-deep-learning-based" and "Deep-learning-based", and the
   top two entries -- AET and AET-KCSR, labelled "AETNet (Ours)" and
   "AETNet-KCSR (Ours)" -- are the benchmark authors' own Vision Transformer
   method. Seven of the nineteen baselines are deep-learning-based, and this
   work is behind four of them.

Categories are the paper's own words, quoted in CATEGORY_SOURCE below.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: The benchmark paper, resolved from its DOI rather than from memory.
BENCHMARK_PAPER = {
    "authors": "Z. Li, Y. Wang, C. Xiao, Q. Ling, Z. Lin, W. An",
    "title": ("You Only Train Once: Learning a General Anomaly Enhancement "
              "Network With Random Masks for Hyperspectral Anomaly Detection"),
    "venue": "IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-18",
    "year": 2023,
    "doi": "10.1109/TGRS.2023.3258067",
    "arxiv": "2303.18001",
    "code": "https://github.com/ZhaoxuLi123/AETNet",
    "note": ("HAD100 is introduced in this paper. An earlier draft of RESULTS.md "
             "attached this DOI to the title 'Hyperspectral Anomaly Detection: A "
             "Survey and Benchmark', which does not exist -- same authors, venue "
             "and DOI, fabricated title."),
}

#: Verbatim from the paper, used to justify every row below.
CATEGORY_SOURCE = {
    "fig1": ("Fig. 1 axis labels split the methods into 'Non-deep-learning-based' "
             "and 'Deep-learning-based', and mark 'AETNet (Ours) 28.6FPS, "
             "0.9925mAUC' and 'AETNet-KCSR (Ours) 0.15FPS, 0.9948mAUC'."),
    "comparative": (
        "Sec. 'Comparative Methods': \"GRX is the most classic AD method and is "
        "widely used as the AD baseline. LRX, KRX, QLRX, 2S-GLRT, CR, and KCSR are "
        "dual-window-based AD methods. CR and KCSR are sparse-representation-based "
        "AD methods. PTA is one of the most advanced tensor-approximation-based AD "
        "methods, and AE[D] introduces morphological image processing to the "
        "hyperspectral AD field. KRX, KSVDD, KCSR, and KIFD all use the kernel "
        "trick to process spectral samples. RGAE, LREN, WeaklyAD, and AutoAD are "
        "deep-learning-based methods, of which the first three are spectral-level "
        "self-supervised methods.\""),
    "weaklyad": ("Sec. I: \"WeaklyAD uses a spectral-constrained GAN to enhance the "
                 "discrimination between anomalies and background in a weakly "
                 "supervised way.\""),
    "aednote": ("The quoted sentence reads 'AET introduces morphological image "
                "processing', but AET is the paper's own Vision Transformer and AED "
                "[27] is the attribute/edge-preserving filter in the comparison "
                "list. Read as AED; flagged rather than silently corrected."),
    "bands": ("Sec. 'Settings of Comparative Experiments': \"we conducte three "
              "comparative experiments on the first 50 bands, the first 100 bands, "
              "and the first 200 bands\" -- which independently confirms the band "
              "protocol RESULTS.md Sec. 2 recovered by reproducing GRX."),
}

#: method -> (category, family detail, source key)
CATEGORIES = {
    "GRX":         ("statistical",          "classic RX baseline, global",                "comparative"),
    "LRX":         ("statistical",          "dual-window, local",                          "comparative"),
    "QLRX":        ("statistical",          "dual-window, quasi-local",                    "comparative"),
    "2S-GLRT":     ("statistical",          "dual-window, generalised likelihood ratio",   "comparative"),
    "KRX":         ("kernel",               "dual-window + kernel trick",                  "comparative"),
    "KSVDD":       ("kernel",               "support vector data description",             "comparative"),
    "KIFD":        ("kernel",               "kernel isolation forest",                     "comparative"),
    "CR":          ("representation-based", "dual-window, sparse/collaborative repr.",     "comparative"),
    "KCSR":        ("representation-based", "dual-window, sparse repr. + kernel",          "comparative"),
    "PTA":         ("representation-based", "prior-based tensor approximation",            "comparative"),
    "AED":         ("morphological",        "attribute and edge-preserving filtering",     "aednote"),
    "FrFE":        ("transform-domain",     "fractional Fourier entropy",                  "comparative"),
    "RGAE":        ("deep learning",        "robust graph autoencoder, self-supervised",   "comparative"),
    "LREN":        ("deep learning",        "low-rank embedded network, self-supervised",  "comparative"),
    "AutoAD":      ("deep learning",        "autoencoder, self-supervised per scene",      "comparative"),
    "WeaklyAD-A":  ("deep learning",        "spectral-constrained GAN, weakly supervised", "weaklyad"),
    "WeaklyAD-B":  ("deep learning",        "spectral-constrained GAN, weakly supervised", "weaklyad"),
    "AET":         ("deep learning",        "AETNet, Vision Transformer (paper's own)",    "fig1"),
    "AET-KCSR":    ("deep learning",        "AETNet + KCSR detector (paper's own)",        "fig1"),
}

#: This work's entry, read from the final HAD100 table rather than typed.
#:
#: The quoted figure is the crop-level mAUC over all 100 images at the
#: cross-validated alpha, averaged over seeds 0-4. Alpha is selected
#: out-of-fold, so no image contributes to the weight it is scored under --
#: which is what makes this comparable to the published rows at all.
OURS_LABEL = "this work -- fused, CV alpha"
OURS_SOURCE = Path("results/revision/b_seeds.json")


def ours_band50() -> tuple[float, list[float]]:
    if not OURS_SOURCE.is_file():
        raise SystemExit(f"{OURS_SOURCE} is missing; run revision_b_seeds.py first.")
    t = json.loads(OURS_SOURCE.read_text(encoding="utf-8"))["final_table"]
    crop = t["fused_cv_alpha"]["crop"]
    return float(crop["mean"]), [float(v) for v in crop["values"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baselines", default="results/had100_baselines.json")
    p.add_argument("--out", default="results/revision/e_benchmark_table.json")
    p.add_argument("--md", default="results/revision/e_benchmark_table.md")
    args = p.parse_args(argv)

    data = json.loads(Path(args.baselines).read_text(encoding="utf-8"))["table"]
    rows = [
        {"method": m, "band50": v.get("band50"), "band100": v.get("band100"),
         "band200": v.get("band200"),
         "category": CATEGORIES.get(m, ("?", "?", "?"))[0],
         "detail": CATEGORIES.get(m, ("?", "?", "?"))[1],
         "source": CATEGORIES.get(m, ("?", "?", "?"))[2],
         "is_ours": False}
        for m, v in data.items()
    ]
    uncategorised = [r["method"] for r in rows if r["category"] == "?"]
    ours_mean, ours_seeds = ours_band50()
    rows.append({"method": OURS_LABEL, "band50": ours_mean, "band100": None,
                 "band200": None, "category": "statistical + deep learning",
                 "detail": "RX (Eq. 2) fused with a 3D-conv autoencoder (Eq. 1, 3)",
                 "source": "this work", "is_ours": True})
    rows.sort(key=lambda r: -(r["band50"] or -1))

    lines = ["| Rank | Method | Category | band50 | band100 | band200 |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        f = lambda v: "--" if v is None else f"{v:.4f}"
        name = f"**{r['method']}**" if r["is_ours"] else r["method"]
        cat = f"**{r['category']}**" if r["category"] == "deep learning" else r["category"]
        lines.append(f"| {i} | {name} | {cat} | {f(r['band50'])} | "
                     f"{f(r['band100'])} | {f(r['band200'])} |")
    table_md = "\n".join(lines)
    print(table_md)

    deep = [r for r in rows if r["category"] == "deep learning"]
    ahead = [r for r in deep if (r["band50"] or 0) > ours_mean]
    behind = [r for r in deep if (r["band50"] or 0) < ours_mean]
    ours_rank = next(i for i, r in enumerate(rows, 1) if r["is_ours"])

    print(f"\n{'=' * 72}")
    print(f"this work ranks {ours_rank} of {len(rows)} on band50 at "
          f"{ours_mean:.4f} (mean over seeds {ours_seeds})")
    others = sorted((r for r in rows if not r["is_ours"]),
                    key=lambda r: -(r["band50"] or -1))
    per_seed_rank = [1 + sum(1 for r in others if (r["band50"] or 0) > v)
                     for v in ours_seeds]
    print(f"  rank per seed: {per_seed_rank} -- the mean rank is not "
          "attained by every seed, so the table reports the mean and "
          "this spread together.")
    print(f"\ndeep-learning entries: {len(deep)} of {len(rows) - 1} baselines")
    print(f"  AHEAD of this work ({len(ahead)}):")
    for r in ahead:
        print(f"    {r['method']:<12} {r['band50']:.4f}   {r['detail']}")
    print(f"  behind this work ({len(behind)}):")
    for r in behind:
        print(f"    {r['method']:<12} {r['band50']:.4f}   {r['detail']}")
    print(f"\n  => \"ahead of every deep-learning entry\" is FALSE.")
    print(f"     True only of: {', '.join(r['method'] for r in behind)}")
    if uncategorised:
        print(f"\n  WARNING uncategorised: {uncategorised}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark_paper": BENCHMARK_PAPER,
        "category_source_quotes": CATEGORY_SOURCE,
        "rows": rows,
        "ours_rank": ours_rank,
        "n_rows": len(rows),
        "deep_learning_ahead_of_ours": [r["method"] for r in ahead],
        "deep_learning_behind_ours": [r["method"] for r in behind],
        "claim_correction": (
            "'ahead of every deep-learning entry' is false. This work (0.9843 "
            f"band50) is behind {len(ahead)} of {len(deep)} deep-learning baselines "
            f"({', '.join(r['method'] for r in ahead)}) and ahead of "
            f"{', '.join(r['method'] for r in behind)} only."),
    }, indent=2), encoding="utf-8")
    Path(args.md).write_text(table_md + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
