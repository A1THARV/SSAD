# Results

Measured output of the prototype, intended to replace the paper's Section VII
placeholder. Every number here was produced by a command in this repository and
is reproducible from a config in `configs/`; run records live under `results/`
with the config hash that produced them.

**Hardware.** NVIDIA RTX 3050 Laptop (4 GB VRAM), Windows 11, PyTorch 2.5.1+cu121.
Peak VRAM across every run reported here was **0.31 GB**, well inside the 3.3 GB
budget — the 4 GB constraint shaped the architecture (patch-based training over
PCA-reduced cubes) rather than limiting the results.

---

## 1. Headline: HAD100

100 test images, 64×64, AVIRIS-NG, 276 bands after the benchmark's own band
selection. Metric is **mAUC** — per-image AUC-ROC averaged over the 100 images,
which is what HAD100's `eval_results.py` reports and therefore the only figure
comparable to its published table. A pooled AUC over all 409,600 pixels is a
different quantity and is not used.

| Branch | 276 bands | band50 protocol |
|---|---|---|
| RX (Eq. 2), per-scene PCA basis | 0.9749 | — |
| RX (Eq. 2), corpus PCA basis | 0.9806 | 0.9835 |
| Autoencoder (Eq. 1) alone | 0.9522 | 0.9619 |
| **Fused, α = 0.5** | 0.9778 | **0.9843** |

**0.9843 is the headline.** It is the full pipeline — Eq. 1 + Eq. 2 + Eq. 3 fused
under Fig. 1 stage 5 — at α = 0.5, the fusion weight fixed in the config *before*
any of the ablations below were run. Nothing in that number was selected on the
test set.

Against the published ranking it sits **10th of 20** in the band50 column,
0.0002 behind KSVDD and ahead of KIFD, QLRX, GRX and LRX. See §3.

That placement belongs to the **fused system**, not to the deep component. The
autoencoder alone scores 0.9619, which is *below* AutoAD's 0.9688 — so this work
does not beat the benchmark's deep-learning entries with a better autoencoder. It
beats them by fusing a weak autoencoder with a strong RX branch, and §7
quantifies how weak.

Fusion buys **+0.0008 over the best single branch** at band50 — measured against
corpus-basis RX on the same config, which is the honest comparison. (An earlier
draft quoted +0.0029; that was fusion measured against the *weaker* per-scene RX
on the 276-band config, and it overstated the gain. §7 has the corrected sweep.)

```
python -m pipeline.cli.run_hyperspectral --config configs/had100_band50.yaml
python -m pipeline.cli.run_hyperspectral --config configs/had100.yaml
```

Note the two columns are not directly comparable branch-for-branch: the 276-band
config carries `rx.pca_basis: per_scene` (its default when the run was made), so
its fused row is fused against the *weaker* RX. The band50 config uses the corpus
basis for both, which is why its fusion delta is the honest one.

---

## 2. Eq. 2 is verified against a published reference implementation

HAD100's baseline table has three columns — `band50`, `band100`, `band200` — and
its shipped code never defines them. Recovering the protocol turned into a
correctness check on our own detector.

`GRX` in that table is plain global RX, i.e. exactly Eq. 2. Running our
`rx_score` on the **first N** of the 276 bands reproduces it:

| N | ours | published GRX | Δ |
|---|---|---|---|
| 50 | 0.9799 | 0.9799 | 0.0000 |
| 100 | 0.9714 | 0.9714 | 0.0000 |
| 200 | 0.9653 | 0.9649 | 0.0004 |

Two exact matches to four decimal places. The competing hypotheses do not match:
uniform subsampling gives 0.9742 / 0.9710, strided selection 0.9727 / 0.9701.

So `band50` means "the first 50 bands", and our Eq. 2 implementation agrees with
an independent published implementation on a public benchmark. The 0.0004 gap at
N = 200 is consistent with a difference in covariance regularisation, which
matters most where the 4,096-pixel covariance is closest to ill-conditioned.

Recorded as presets `had100_band50` / `_band100` / `_band200`; see
`scripts/had100_baselines.py`.

---

## 3. HAD100's `band50` advantage is a wavelength effect, not a dimensionality one

Every one of the 19 published baselines scores best at `band50` and degrades
monotonically as bands are added. The obvious reading — fewer dimensions, better
covariance estimate — is wrong.

The first 50 selected bands are original AVIRIS-NG bands 15–65, i.e.
**455–700 nm: the visible spectrum**. Holding the band *count* at 50 and sliding
the window across the range:

| window | wavelengths | RX mAUC |
|---|---|---|
| bands 0–49 | 455–700 nm | **0.9799** |
| bands 50–99 | 705–995 nm | 0.8905 |
| bands 100–149 | 1000–1310 nm | 0.8423 |
| bands 150–199 | 1515–2030 nm | 0.9067 |
| bands 200–249 | 2035–2280 nm | 0.9162 |
| 50 drawn evenly across the whole range | 455–2410 nm | 0.9742 |

Identical machinery, identical dimensionality, a 0.14 mAUC spread. The
discriminative signal for HAD100's targets lives in the visible; the SWIR bands
are close to noise for this task and adding them actively hurts.

**Caveat, stated rather than buried.** This was measured on the labelled test
images, because HAD100 ships no labelled validation split. It is reported as a
diagnostic observation about the benchmark, not as a held-out selection, and no
hyperparameter was chosen from it.

### A prediction this forces about Objective 2, recorded before M3 runs

If HAD100's discriminative signal lives at 455–700 nm, then **HAD100 is a weak
benchmark for Objective 2** — SAM material matching against USGS/ECOSTRESS.

Eq. 4's discriminative power over a mineral or material library comes
overwhelmingly from **SWIR absorption features**: hydroxyl and carbonate bands
near 2200–2350 nm, water features around 1400 and 1900 nm, the Fe-oxide edge in
the near-IR. Those are the diagnostic shapes a spectral library encodes. They sit
in precisely the windows the table above shows are near-noise for this
benchmark's targets — 1000–1310 nm at 0.8423, 1515–2030 nm at 0.9067,
2035–2280 nm at 0.9162.

The two objectives therefore want different parts of the spectrum. Detection here
is a visible-range problem; identification is a SWIR problem. A benchmark tuned
to the first is not evidence about the second.

**So this is written down now, before M3 produces any number:** a poor SAM result
on HAD100 would not indict Eq. 4. It would indict the benchmark's fit to
Objective 2. Stated in advance it is a prediction; stated afterwards it would be
an excuse, and the difference matters. If M3 does produce weak HAD100 material
matching, the correct response is to evaluate Objective 2 on a scene with
SWIR-diagnostic targets rather than to conclude Eq. 4 does not work.

Note this also bounds §1's headline honestly: the band50 configuration that
produces 0.9843 **discards every SWIR band**, so that configuration cannot serve
Objective 2 at all. Detection and identification will need different band
selections from the same pipeline — which the `good_bands_preset` mechanism
already supports, and which the paper should state rather than leave implicit.

### The published ranking

All 19 baselines, recomputed from the benchmark's shipped result matrices using
*our* `mean_per_image_auc` — so a transcription error or a pooled-vs-per-image
metric mismatch cannot creep in.

| Rank | Method | band50 | band100 | band200 |
|---|---|---|---|---|
| 1 | AET-KCSR | 0.9948 | 0.9897 | 0.9869 |
| 2 | KCSR | 0.9939 | 0.9865 | 0.9860 |
| 3 | CR | 0.9929 | 0.9859 | 0.9849 |
| 4 | AET | 0.9925 | 0.9875 | 0.9818 |
| 5 | WeaklyAD-B | 0.9887 | 0.9838 | 0.9850 |
| 6 | WeaklyAD-A | 0.9862 | 0.9788 | 0.9773 |
| 7 | AED | 0.9859 | 0.9645 | 0.9625 |
| 8 | 2S-GLRT | 0.9853 | 0.9683 | 0.9310 |
| 9 | KSVDD | 0.9845 | 0.9737 | 0.9706 |
| **10** | **this work — fused, α = 0.5** | **0.9843** | — | — |
| 11 | KIFD | 0.9829 | 0.9822 | 0.9774 |
| 12 | QLRX | 0.9801 | 0.9715 | 0.9682 |
| 13 | GRX | 0.9799 | 0.9714 | 0.9649 |
| 14 | LRX | 0.9775 | 0.9692 | 0.9720 |
| 15 | AutoAD | 0.9688 | 0.9661 | 0.9642 |
| 16 | KRX | 0.9616 | 0.9329 | 0.9282 |
| 17 | LREN | 0.9583 | 0.9286 | 0.9309 |
| 18 | FrFE | 0.9476 | 0.9371 | 0.9164 |
| 19 | PTA | 0.9031 | 0.8976 | 0.8686 |
| 20 | RGAE | 0.9003 | 0.8876 | 0.8823 |

Tenth of twenty on the column where the baselines are strongest — 0.0002 behind
KSVDD and ahead of the classical family (KIFD, QLRX, GRX, LRX). Behind the kernel
and transform-based leaders; the four methods above 0.99 are the honest target
for further work.

**On the deep-learning comparison, stated precisely.** The fused system (0.9843)
is ahead of AutoAD (0.9688), LREN (0.9583) and RGAE (0.9003). The autoencoder
*alone* is **0.9619 — below AutoAD**. The paper must attribute this to the fused
pipeline and not let it read as a claim about the deep component, which §7 shows
is the weaker half of the fusion.

Source: Li et al., *Hyperspectral Anomaly Detection: A Survey and Benchmark*,
IEEE TGRS 61, 2023, doi:10.1109/TGRS.2023.3258067. Per change C3 the baselines
are **consumed, not re-implemented**.

---

## 4. Fig. 1 was right about the single PCA block

The paper's Fig. 1 draws one PCA block feeding both the autoencoder and the RX
branch. The implementation initially split them — a corpus-fitted basis for the
autoencoder (which it needs, since train and inference must share axes) and a
per-scene basis for RX (on the reasoning that Eq. 2 estimates Σ from the scene
anyway, so a scene-local basis is consistent with it).

Measured, that reasoning was wrong:

| RX basis | K = 8 | K = 15 |
|---|---|---|
| per-scene | 0.9786 | 0.9749 |
| **corpus** | 0.9782 | **0.9806** |

At K = 15 the corpus basis wins by +0.0057 and is the best RX result obtained
anywhere in this work — it also clears the published GRX `band50` figure of
0.9799. At K = 8 the two are within 0.0004, i.e. indistinguishable.

Why the gap opens with width: the corpus basis is fitted on 4.26 M pixels from
1,040 scenes, the per-scene basis on 4,096 pixels from one. At K = 8 both have
plenty of data for eight stable directions. By K = 15 the per-scene fit is
estimating directions it does not have the samples to resolve, and Eq. 2 then
inverts a covariance expressed in a partly arbitrary basis.

**Consequence for the paper: Fig. 1 needs no second PCA path.** That edit comes
off the list; `rx.pca_basis: corpus` is now the default in
`configs/had100_band50.yaml`.

---

## 5. Eq. 3's SAM term earns its place

Ablating λ and the domain the angle is evaluated in:

| variant | λ | SAM domain | mAUC | vs no-SAM |
|---|---|---|---|---|
| **baseline** | 0.1 | **reflectance** | **0.9522** | **+0.0065** |
| no SAM term | 0.0 | — | 0.9457 | — |
| angle taken on PCA scores | 0.1 | latent | 0.9167 | **−0.0290** |

Two findings, and the second is the more important one.

Eq. 3's SAM term earns its place: dropping it costs **0.0065 mAUC**, so the
composite loss is doing work rather than decorating an MSE.

But it earns its place **only when evaluated in the reflectance domain**.
Evaluated on the PCA scores — the obvious implementation, since those are the
tensors already in hand — the same term costs 0.0290 *relative to omitting it
entirely*. The naive version does not merely fail to help; it actively harms,
and it would have been reported as evidence against the paper's own Eq. 3.

### Why the domain matters

Eq. 4's physical justification is that the angle between two *reflectance*
spectra is invariant to illumination — a shadowed and a sunlit patch of one
material differ in magnitude, not direction. That argument does not survive the
move to PCA scores:

* PCA scores are mean-centred, so the origin is the **corpus mean spectrum**, not
  darkness. An unremarkable background pixel projects to near zero, where its
  direction is noise and its angle to anything is unstable.
* The angle between score vectors is not the angle between the spectra they
  encode.

So the default back-projects both the input and the reconstruction to reflectance
(`x = z·W + μ`, a fixed differentiable linear map) before taking the angle. On
matched spectra the two domains disagree by more than 5×, and the 0.0355 mAUC gap
between the two rows above is what that disagreement costs.

The mechanism is visible in the training traces. Both runs converge, but the
latent-domain SAM term starts at 0.3565 rad against the reflectance term's
0.0899 — four times larger for the same reconstructions, because near the origin
the angle between score vectors is dominated by noise. The optimiser spends
capacity driving down an angle that carries no information about how well the
spectrum was reconstructed.

---

## 6. Whitening the autoencoder input

Pairing whitening with PCA width, because a prior measurement said the two
interact. Principal angles between bases fitted on disjoint halves of the corpus:

| components | 1–6 | 7–10 | 11–15 |
|---|---|---|---|
| angle between halves | < 2.2° | 3.7–8.7° | 13.0–**87.0°** |
| λ_k / λ_1 | ≥ 1.6e-3 | 1.7e-4 – 1.0e-3 | ≤ 1.1e-4 |

Only the leading ~8 directions reproduce across disjoint samples. Beyond that the
eigenvectors rotate freely while the eigenvalues stay smooth — eigenvector
stability fails first.

Whitening divides each component by √λ, promoting the trailing components to
equal weight. At K = 15 those are exactly the non-reproducible ones, so whitening
should hurt there and be more defensible near K ≈ 8. Measured:

| K | latent | whiten off | whiten on | Δ |
|---|---|---|---|---|
| 15 | 8 | **0.9522** | 0.9472 | **−0.0050** |
| 8 | 4 | 0.9217 | 0.9426 | **+0.0209** |

### Why this one belongs in the paper as a worked example

The ordering matters more than the numbers. The stability spectrum was measured
first, for an unrelated reason — checking whether a corpus PCA fit needed
sampling. It produced a *falsifiable prediction* about a parameter nobody had
measured yet: whitening should help at K ≈ 8 and hurt above it, because the
denominator √λ promotes directions that do not reproduce.

The prediction was written into `ROADMAP.md` before the ablation was run, it
specified the direction of the effect at both widths, and it held at both.

That is worth more to the paper than the 0.0209 itself, because it demonstrates
the method has a *mechanism* rather than a tuning surface. The usual form of this
result in the literature is "we swept the hyperparameter and 8 worked best",
which is unfalsifiable and does not transfer to another sensor. A subspace
stability spectrum is measurable on any corpus, with no labels, before any model
is trained — so the rule it implies ("whiten only up to the stability limit")
transfers where a swept optimum does not.

It also names its own failure mode: run at a single fixed width, as originally
planned, this would have been recorded as "whitening is harmful" and the
mechanism missed entirely. The paired design is what made it visible.

**The sign flips with width, in the predicted direction.** The prediction was
made from the stability spectrum before the measurement was run, and it holds:

* At **K = 8** every retained direction reproduces across disjoint corpus halves,
  so whitening does only the thing it was supposed to do — remove PC1's
  22,076× dominance of the MSE — and buys **+0.0209**.
* At **K = 15** components 11–15 are sample noise. Whitening promotes them to
  equal weight, and the network spends capacity reconstructing directions that
  will not be there next time. It **costs 0.0050**.

So `whiten_input` is not a good or bad setting; it is a setting that is only
meaningful paired with width. Neither value should be treated as a default, which
is why the config carries both and the config comment names the pairing.

Note the confound in the cross-K rows: K = 8 forces `latent_dim` from 8 down to 4
(Fig. 1 requires d ≪ K), so the two rows vary two things at once. The **within-K**
comparison — each row read left to right — is the clean one, and it is where the
sign flip lives.

---

## 7. Full ablation table

All six autoencoder variants, 100 HAD100 test images, `results/m2/ablations.json`.
One-at-a-time from the baseline rather than a full factorial.

| variant | K | latent | whiten | λ | SAM domain | mAUC |
|---|---|---|---|---|---|---|
| **baseline** | 15 | 8 | off | 0.1 | reflectance | **0.9522** |
| whiten | 15 | 8 | **on** | 0.1 | reflectance | 0.9472 |
| no SAM | 15 | 8 | off | **0.0** | — | 0.9457 |
| SAM in latent space | 15 | 8 | off | 0.1 | **latent** | 0.9167 |
| K = 8 | **8** | **4** | off | 0.1 | reflectance | 0.9217 |
| K = 8 + whiten | **8** | **4** | **on** | 0.1 | reflectance | 0.9426 |

### Fusion weight sensitivity

α = 1.0 is RX alone, α = 0.0 the autoencoder alone. **α = 0.5 is the reported
operating point** — fixed in the config a priori, before any of this was
measured. The rest of each curve is post-hoc sensitivity, not a selection.

**The sweep is reported on both configs, because the first one was on the wrong
one.** The six-variant sweep below runs on `had100.yaml`, where α = 1.0 is
*per-scene* RX at 0.9749. The headline in §1 is the band50 config, where RX alone
is 0.9835. A fusion curve measured against a weaker RX branch overstates what
fusion adds, so the band50 sweep is the one the paper should quote.

| α | band50 (corpus RX) | 276-band (per-scene RX) |
|---|---|---|
| 0.00 | 0.9619 | 0.9522 |
| 0.25 | 0.9772 | 0.9683 |
| 0.40 | 0.9819 | 0.9752 |
| **0.50** | **0.9843** | **0.9778** |
| 0.60 | 0.9861 | 0.9796 |
| 0.75 | 0.9880 | 0.9814 |
| 0.90 | **0.9882** | 0.9808 |
| 1.00 | 0.9835 | 0.9749 |

Moving to the stronger RX branch does what it should: the optimum shifts toward
α = 1 (**0.75 → 0.90**) and the marginal gain at that optimum shrinks
(**+0.0064 → +0.0048**). Both are the predicted direction — a better RX branch
needs less help and tolerates less interference.

**The correction that matters to §1.** I previously reported fusion buying
+0.0029 at α = 0.5. That was measured against per-scene RX. On the config the
headline actually uses, fusion at the a priori α = 0.5 buys **+0.0008** over RX
alone. The fusion story is weaker at the reported operating point than I first
stated, and the earlier figure was flattered by its baseline.

**On the 0.9882.** That is the post-hoc ceiling at α = 0.90, and it is *not*
claimed as a result — α was chosen by reading this table, which was computed on
the 100 images being reported. Quoting it as the headline would be exactly the
test-set tuning this project refuses elsewhere. It is recorded because the shape
of the curve is informative and because hiding it would be worse; it would rank
6th of 20 if it were legitimate, and it is not.

### What the autoencoder is worth, per variant

| α | baseline | whiten | no SAM | SAM latent | K=8 | K=8+whiten |
|---|---|---|---|---|---|---|
| 0.00 | 0.9522 | 0.9472 | 0.9457 | 0.9167 | 0.9217 | 0.9426 |
| 0.25 | 0.9683 | 0.9620 | 0.9625 | 0.9572 | 0.9618 | 0.9609 |
| 0.50 | 0.9778 | 0.9722 | 0.9739 | 0.9730 | 0.9735 | 0.9719 |
| 0.75 | 0.9814 | 0.9787 | 0.9810 | 0.9814 | 0.9795 | 0.9789 |
| 0.90 | 0.9808 | 0.9790 | 0.9815 | 0.9818 | 0.9812 | 0.9810 |
| 1.00 | 0.9749 | 0.9749 | 0.9749 | 0.9749 | 0.9786 | 0.9786 |

**An unflattering reading, stated plainly.** The optimum sits at α ≈ 0.75–0.90 on
both configs, i.e. the autoencoder is worth 10–25% of the decision, not half of
it. And at α = 0.9 every variant lands within 0.001 of every other
(0.9808–0.9818) *including the worst branch in the table*. Once the autoencoder's
weight is low enough to be optimal, **which** autoencoder it is stops mattering —
which means most of §5–§7's careful ablation work buys almost nothing at the
operating point that actually maximises accuracy.

The autoencoder is a weak second opinion, not a co-equal detector: **+0.0048 at
the post-hoc optimum, +0.0008 at the reported one.** PRD §14 anticipated it
underperforming RX on scene-limited AVIRIS data; it does, and this is that
prediction quantified.

One genuine improvement is visible across configs, though: the autoencoder branch
is stronger at band50 (0.9619 vs 0.9522), gaining more from dropping the SWIR
bands than RX does (+0.0097 against RX's +0.0029). The reconstruction objective
appears to suffer more from noise-dominated bands than the Mahalanobis distance
does — consistent with §3, and worth one sentence in the paper.

---

## 8. Edge case: a scene with no anomalies at all

Salinas, 512×217×204, run under `label_convention: zero_anomaly` so the ground
truth is all-background and every detection is by construction a false alarm
(PRD §11 edge case 3, change C5). AUC is undefined here and is reported as
`n/a (single-class GT)` rather than substituted — that is the correct output, not
a failure.

### Variance-retained PCA selection is actively harmful

The false-alarm rate is a steep function of PCA width, and the conventional
"retain 99% of the variance" rule picks the **worst** width available:

| PCA width | Otsu | percentile p99 | μ + 3σ |
|---|---|---|---|
| **K = 3** — what `pca_variance: 0.99` selects | **14.62%** | 1.00% | 1.07% |
| K = 5 | 11.00% | 1.00% | 2.12% |
| K = 10 | 0.31% | 1.00% | 0.79% |
| K = 15 | 0.22% | 1.00% | 0.57% |
| K = 30 | 0.00% | 1.00% | 0.23% |

Salinas is extremely low-rank — 99% of its variance sits in three components — so
the variance rule selects K = 3 and flags **14.6% of an anomaly-free scene**.

The mechanism: anomalies are low-variance by construction, so "retain 99% of the
variance" discards precisely the directions that separate them. Once those are
gone the *background itself* is heterogeneous in what remains, Otsu finds a
genuine split in it, and a seventh of the scene is flagged.

`pca_n_components` is therefore the operative setting in every config;
`pca_variance` is retained only as a documented fallback.

### And it corrects an earlier claim about thresholding

An earlier note in `ROADMAP.md` recorded "Otsu 0.22% vs percentile 1.00%" and
concluded Otsu is the better default for zero-anomaly scenes. The table above
shows that holds **only when the background is well modelled**. At K ≤ 5 Otsu is
an order of magnitude *worse* than a fixed percentile, because it adapts to the
histogram — a virtue when the histogram is unimodal-plus-tail, a liability when
the retained subspace has made the background multimodal. A percentile is
insensitive to this by construction: it flags 1.00% at every width, right or
wrong.

Both numbers in that earlier note were correct; they were measured at different
K, which was not recorded at the time.

### At K = 15 the flagged pixels are the right ones

Per land-cover class, at the Otsu threshold:

| class | pixels | flagged |
|---|---|---|
| 0 (unlabelled/edges) | 56,975 | 0.42% |
| 5 | 2,678 | 0.04% |
| all 15 other crop classes | 48,452 | 0.00% |

Every one of the 16 actual crop classes is essentially untouched; the residual
false alarms are confined to unlabelled and field-boundary pixels. RX's
unimodal-Gaussian background assumption survives a 16-class agricultural scene
better than expected — provided the PCA width leaves it enough directions to
model the background in.

---

## 9. Reproducibility

* Seeds fixed (`seed: 0`, `deterministic: true`); repeated training runs agree to
  1e-9 in per-epoch loss.
* Every run writes `metrics.json`, `config_used.yaml`, `config_hash.txt` and
  `command.txt`; the hash is taken over the *resolved* config, so a changed
  default is detected even when no file changed.
* The corpus PCA basis is fitted **exactly**, not sampled: a two-pass streaming
  covariance over 1,040 memory-mapped crops, 4,259,840 pixels, 85 s, holding
  0.61 MB of covariance against the 4.70 GB an eager load would take.
* **333 unit tests**, all under `tests/unit/`, no skips on this machine. There is
  no separate integration suite — the end-to-end paths are exercised by the runs
  in §11 rather than by tests, which is a gap worth naming rather than papering
  over.

### An environment hazard worth recording

Anaconda's MKL and PyTorch ship different builds of `libiomp5md.dll`. Importing
torch and then calling any MKL-backed NumPy routine — `a.T @ a`, `np.linalg.eigh`,
i.e. every line of the PCA path — **aborts the interpreter** with `OMP: Error #15`.

The standard workaround (`KMP_DUPLICATE_LIB_OK=TRUE`) is documented by Intel as
unsupported and warned to "may silently produce incorrect results", which is the
one failure mode this project cannot tolerate. It is therefore verified rather
than trusted: `tests/unit/test_runtime.py` runs the corpus-covariance kernel in
two subprocesses — one with torch loaded under the flag, one with NumPy alone —
and asserts the covariance and its eigenvalues are **bit-identical**. They are.
The workaround stays only as long as that test passes.

---

## 10. Change C2's premise is right about the conclusion and wrong about the reason

Change C2 replaced single-scene self-supervised training with training on a
separate anomaly-free corpus, arguing that Eq. 1's premise — "anomalies
reconstruct poorly" — cannot hold for a background model fitted on data
containing the anomalies it must detect.

Tested directly. Everything held fixed except which pixels the network saw: same
architecture, same 20 epochs, same λ, same corpus-fitted PCA basis.

| training data | patches | models | mAUC | wall clock |
|---|---|---|---|---|
| **the test scene itself** (Auto-AD style) | 4,096 per scene | 100 | **0.9499** | 42 min |
| corpus, **matched** budget | 4,096 | 1 | 0.9418 | 24 s |
| corpus, full budget | 40,000 | 1 | **0.9522** | 232 s |

**At matched data volume, training on the test scene wins by +0.0080.** The
contamination argument does not survive contact with the measurement. The
anomalies in HAD100 are a handful of pixels in a 64×64 crop — well under 1% of
the training patches — so the model barely sees them, while it learns *that
scene's* exact background rather than an average of 1,040 others. Scene-specific
knowledge outweighs the contamination.

What actually makes corpus training better is **data volume**, not purity. Given
ten times the patches, one corpus model beats 100 per-scene models by +0.0023 —
and does it in 232 seconds instead of 42 minutes, a ~10× cost difference that
grows linearly with the size of the test set.

**So change C2 stands, on different grounds than it was argued from.** It should
be justified in the paper by efficiency and by the data volume an anomaly-free
corpus makes available — not by the claim that a contaminated background model
cannot support Eq. 1. On this benchmark, at equal data, it demonstrably can.

The original argument would become correct in a regime this benchmark does not
test: larger anomalies occupying a meaningful fraction of the scene. That is a
stated limitation, not a defence.

---

## 11. Image-level fire *presence* is saturated by a one-line baseline

M4 was specified as image-level fire detection with a CNN. Before building one,
the trivial baseline was measured — and it leaves nothing for a model to do.

**What was measured, stated narrowly.** This is image-level fire *presence*: a
single binary flag per frame. It is **not** localisation. Objective 3 asks for
per-frame anomaly heatmaps, and that task is neither measured nor claimed here,
because FLAME 3's CV subset ships no ground-truth masks to score a heatmap
against. The two tasks are different problems and only the first is saturated.

**A single scalar, the maximum pixel temperature, separates all 738 frames
perfectly.**

| single feature | AUC | best F1 |
|---|---|---|
| **max T** | **1.0000** | **1.0000** |
| fraction of pixels > 100 °C | 1.0000 | 1.0000 |
| 99.9th percentile T | 0.9999 | 0.9992 |
| 99th percentile T | 0.9900 | 0.9773 |
| std T | 0.9784 | 0.9761 |
| mean T | 0.7437 | 0.9147 |

The margin is not narrow. The coldest fire frame peaks at **108.25 °C**; the
hottest no-fire frame peaks at **47.19 °C**. A 61 °C gap, zero overlap, 622
fire and 116 no-fire frames.

### There is no harder subset hiding inside it

Restricting the fire class to the smallest fires by burning area, keeping every
no-fire frame:

| fire subset | n fire | AUC (max T) | coldest fire peak |
|---|---|---|---|
| all | 622 | 1.0000 | 108.25 °C |
| smallest 50 % by area | 312 | 1.0000 | 108.25 °C |
| smallest 25 % | 156 | 1.0000 | 108.25 °C |
| smallest 10 % | 63 | 1.0000 | 108.25 °C |
| smallest 5 % | 32 | 1.0000 | 108.25 °C |

The smallest fire in the dataset has essentially **no** pixels above 100 °C
(0.000 % after rounding) and still peaks at 108.2 °C. The task is saturated at
every difficulty level the data contains.

### What follows

A CNN cannot demonstrate value here. It can at best tie a threshold, and
reporting "CNN AUC ≈ 1.0" beside "max(T) AUC = 1.0" would be padding — the same
error as quoting a fused score against a baseline that was never measured.

This is not a defect in FLAME 3. The dataset was built for computer vision on
**RGB**, where fire detection is genuinely hard; the radiometric thermal channel
is supplementary, and in that channel wildfire is trivially separable because
fire is, physically, the hot thing. The task is easy for the same reason the
data is good.

### The two sentences this reduces to

For the paper, the result is exactly this pair, and the second is not optional:

> Image-level fire presence on radiometric thermal is separable by a physical
> threshold: **AUC 1.0000** on maximum pixel temperature, a **61 °C** margin and
> zero overlap across 738 frames, holding to the smallest 5 % of fires by
> burning area.
>
> **Pixel-level localisation was not evaluated**; the CV subset provides no
> ground-truth masks.

Without the second sentence, the first invites precisely the question it cannot
answer — a reviewer reading "AUC 1.0000" against a paper whose Objective 3 is
localisation will assume the heatmap was scored, and nothing else in the
document would correct them.

**Consequence for the paper.** M4's image-level result is the baseline itself,
reported as such: fire *presence* on radiometric thermal is solved by a physical
threshold, and the contribution is the measurement, not a model. The thermal
branch's substantive work is Eq. 5 temperature retrieval (M5) — a threshold can
say fire is present, but it cannot say whether an anomaly is steel or concrete,
and emissivity retrieval needs the radiometry in a way detection demonstrably
does not. "M4 is trivial" is therefore also the argument for M5.

Measured over all 738 frames; no train/test split is involved because no model
is fitted — a threshold on absolute temperature has no parameters learned from
these labels.

**Future work, deliberately not started.** The temporal task that is *not*
saturated is fire-spread forecasting on FLAME 3's NADIR plot sequences, where
the question is where the fire goes next rather than whether it is present. That
is 32.59 GB and a different paper; recorded here so the gap is visible rather
than silently skipped.

---

## 12. One failure mode, three times: preprocessing that normalises away the physics

Three findings in this document look unrelated — they sit in different sections,
two different modalities, and involve different mathematics. They are the same
mistake.

| # | The step | What it normalises away | Cost |
|---|---|---|---|
| §8 | Retain 99 % of variance in PCA | **Low-variance directions** — which is what an anomaly *is* | 14.6 % of an anomaly-free scene flagged |
| §5 | Take Eq. 4's angle on PCA scores | **The physical origin** — mean-centring moves it from darkness to the corpus mean, so the angle stops measuring spectral shape | −0.0290 mAUC, worse than omitting the term |
| §11 | Per-frame min-max on thermal frames | **Absolute temperature** — every frame's hottest pixel maps to 1.0 | Erases the only feature separating fire from field |

Each step is standard. Each is defensible in isolation, and two of them are what
a textbook recommends. Each destroys the exact physical quantity the downstream
method depends on.

**What makes this class dangerous is not that the steps are wrong — it is that
none of them fail loudly.** Every one produces a plausible array of the right
shape and dtype. The PCA still explains its variance; the angles are still in
[0, π]; the normalised frames still look like thermal images. Models train
without complaint. The accuracy loss arrives with no signature pointing back at
its cause, and lands instead on the architecture, the hyperparameters, or the
dataset — which is how a preprocessing bug becomes a permanent conclusion about
a method.

**The check that catches all three is one question, asked before the step and
not after:** *what physical quantity does the downstream method depend on, and
does this transformation preserve it?*

* RX depends on low-variance structure → a variance-retention rule discards it.
* SAM depends on the direction of a spectrum from the origin → mean-centring
  moves the origin.
* Fire detection depends on absolute temperature → min-max removes the scale.

None of that needs an experiment. All three were caught here by measurement,
which worked but is the expensive path: each cost a run, and the first two were
caught only because an ablation happened to be scheduled. The question above is
free, and it is the generalisable result — it does not depend on hyperspectral
imaging, on wildfire, or on anything else specific to this work.

This belongs in the paper's Discussion as a single paragraph with all three as
evidence, not as three scattered warnings in three method sections. Three
independent instances across two modalities is the claim; any one of them alone
is an anecdote.

---

## 13. M5: the emissivity assumption dominates the retrieved temperature

Eq. 5 inverted on real FLAME 3 fire pixels at 10 µm. **The sensor data is
identical in every row** — only the assumed material changes. Every ε below is
derived from measured ECOSTRESS hemispherical reflectance, not hand-picked.

Frame `00001`, 9,124 pixels above 100 °C, brightness temperature **213.4 °C**:

| assumed material | ε | n spectra | retrieved | bounds |
|---|---|---|---|---|
| sand | 0.9388 | 49 | 223.5 °C | 216.9 – 240.0 |
| bark | 0.9483 | 18 | 222.2 °C | 217.0 – 237.5 |
| char | 0.9493 | 36 | 221.7 °C | 216.0 – 229.9 |
| dry vegetation | 0.9525 | 62 | 221.4 °C | 215.8 – 237.5 |
| grass | 0.9638 | 10 | 219.2 °C | 215.9 – 223.7 |
| soil | 0.9650 | 69 | 219.0 °C | 215.8 – 229.5 |
| *brass (out of class)* | *0.1352* | *16* | ***875.3 °C*** | *661.9 – 1155.8* |

**Brass is what the library has, not what was picked.** Steel — the natural
counterexample, and the one an earlier draft used at an assumed ε = 0.28 — has
no LWIR-capable entry here, and neither does metallic aluminium. The metals
ECOSTRESS actually measures over 7.5–13.5 µm are brass and gold, at ε ≈ 0.10–0.14.
Absent that constraint the obvious objection stands: selecting the lowest-ε
material available would inflate the contrast on purpose. The ratio below is
what the library permitted, and it would have been *smaller* had steel been
present.

| | ε range | retrieved spread |
|---|---|---|
| **within class** — the six wildfire materials | 0.939 – 0.965 | **4.4 K** |
| **across class** — adding brass | 0.135 – 0.965 | **656.3 K** |

A **149× ratio**. So the requirement on ε is not a per-pixel material identity;
it is that ε be constrained to the **right material class**. Getting the class
right costs ~4 K. Getting it wrong costs ~650 K.

### The vegetation prior is scene-appropriate, and that turned out not to matter

A burning scene is dry, senescent material, so the prior resolves to ECOSTRESS's
`nonphotosyntheticvegetation.*` branch rather than to every entry mentioning
vegetation. Measured, the correction is negligible:

| | n | mean ε |
|---|---|---|
| photosynthetic (green canopy) | 487 | 0.9537 |
| **non-photosynthetic (dry, dead)** | **62** | **0.9525** |

A difference of **0.0012**, and the within-class spread does not move — it is set
by sand against soil, not by vegetation.

The non-effect is itself worth reporting. Green-versus-dry is a **VNIR
reflectance** phenomenon — chlorophyll absorption and the red edge — while LWIR
emissivity is dominated by water and cellulose vibrational features and by
surface structure, which green and senescent vegetation largely share. **The
distinction that dominates Objective 2 barely registers for Objective 4.** The
selection is still made explicitly, because a prior that is right by coincidence
is not provenance.

### These numbers were re-measured, not carried over

An earlier draft reported 10.2 K within-class and 307.0 K across-class, using
hand-picked textbook emissivities with steel (ε = 0.28) as the counterexample.
Both figures changed once the library replaced the constants, and in opposite
directions:

* **Within-class fell from 10.2 K to 4.4 K.** The measured values cluster far
  more tightly (0.939–0.965) than the textbook set did (0.92–0.98).
* **Across-class rose from 307 K to 656 K.** Steel is absent from this library
  at LWIR coverage, and so is metallic aluminium — its only matches were *black
  gloss paint on an aluminium substrate*. Brass is present and measured, and at
  ε ≈ 0.135 it is well below steel's 0.28, so the error it produces is larger.

Substituting the counterexample while inheriting the old figure would have
understated the effect by a factor of two.

### Sufficient for what, exactly

"~4 K" means nothing without the task it is measured against:

| framing | value |
|---|---|
| spread, absolute | 4.4 K |
| relative, in kelvin — the physical figure | 0.90 % |
| propagated to **radiative power** (∝ T⁴) | **3.6 %** |

* **Sufficient for combustion-regime discrimination.** Smouldering (~300 °C) and
  flaming (~800 °C) sit 18× and 132× the spread away from this retrieval.
* **Small but not free for fire radiative power.** FRP goes as T⁴, so 0.90 % in
  temperature becomes 3.6 % in radiated power — better than the 8.5 % the
  hand-picked constants implied, and still worth stating for anyone taking these
  temperatures downstream.

### Where ε comes from, stated precisely

**Emissivity is derived from measured hemispherical reflectance via Kirchhoff's
law, ε(λ) = 1 − R(λ), valid for opaque surfaces at local thermodynamic
equilibrium.** That is how ASTER and the LST&E literature derive emissivity
routinely, so it is sound rather than a compromise — but it is *an inference
from a published measurement*, not a measurement, and is labelled that way
everywhere it surfaces. The library contains **no emissivity spectra at all**:
all 3,451 entries are reflectance (3,323) or transmittance (128).

Two filters the relation requires, both enforced at build time:

* **Hemispherical geometry only.** Kirchhoff relates directional emissivity to
  directional absorptivity, so the 929 bidirectional entries do not qualify.
* **Opaque samples only.** For a transmitting sample ε = 1 − R − T, so the 129
  transmittance entries are excluded rather than run through 1 − R.

### The conversion is verified on this library, not assumed

Derived ε for materials whose LWIR emissivity is well established:

| material | derived ε | n | expected | |
|---|---|---|---|---|
| water | 0.984 | 9 | 0.96 – 1.00 | ✓ |
| vegetation | 0.954 | 549 | 0.94 – 1.00 | ✓ |
| concrete | 0.944 | 6 | 0.85 – 0.97 | ✓ |
| brass | 0.137 | 15 | 0.02 – 0.30 | ✓ |

**This check earned its place immediately.** The first run used aluminium as the
metal and returned ε = 0.937 — wildly wrong, and flagged out of range. The cause
was that its only two matches were *black gloss paint on an aluminium substrate*:
the word appeared in the free-text `Description` field, describing what the
sample was mounted on rather than what was measured. Matching now uses typed
fields and the filename taxonomy only, and `Description` is excluded. Without
the validation, a paint spectrum would have entered the library as a metal.

**Three of the four bugs found while building this library shared one shape:** a
present thing reported absent, or an absent thing reported present, with no error
raised anywhere.

| bug | shape |
|---|---|
| `Type:` matched for measurement kind | every entry reported absent → two empty subsets, reading as a library deficiency |
| `Description` supplying keyword matches | paint reported present as aluminium |
| generic keyword tested before specific | 18 bark entries reported absent, having been absorbed into vegetation |

None raised an exception; each produced a well-formed file. This is the same
family as §12's preprocessing failures — the output is structurally valid and
semantically wrong — and it is why the countermeasure has to be an *external*
check on a value whose correct range is known independently, rather than any
amount of internal consistency.

### Limitations

* **Temperature.** These are laboratory spectra at **ambient temperature**,
  applied to fire pixels at 220–600 °C. LWIR emissivity is reasonably stable
  with temperature for most of these materials, but **char and soot during
  active combustion are the least certain case — and char is the primary
  material in a burning scene.** The ε used for the hottest pixels is the one
  with the weakest claim to apply at their temperature.
* **Single band.** Separating temperature from emissivity properly needs
  multiple thermal bands (TES). FLAME 3's radiometric TIFF is single-channel, so
  the assumption is unavoidable and is named in the output as
  `assumed-emissivity (single-band)`.
* **Ash is absent** from the library at LWIR coverage. Acceptable here: FLAME 3
  is *active* wildfire — char, flame and unburned vegetation — whereas ash
  dominates post-burn scenes, which this work does not score against. If ash is
  ever needed, cite a published value explicitly rather than substituting
  vegetation.

### What is refused

* **Unlisted materials get a range, never a point estimate** (§11 edge case 11),
  and the fallback is the **in-class** mean — computed with the counterexample
  excluded, so an unidentified patch of a wildfire scene is not handed an
  emissivity that no part of that scene has.
* **Bounds are not inverted.** Higher ε means less correction and therefore a
  *lower* temperature; swapping that would leave the central estimate correct
  while reversing the error bars, which no downstream check would catch.
* **Non-radiometric input is refused end to end**, exercised on the 8-bit
  `thermal_images_equip` fixture rather than only in unit tests.

### Eq. 5 is validated against an independent implementation

Self-consistency cannot catch a wrong physical constant or a units error — both
produce smooth, positive, plausible output. So `planck_radiance` is checked
three independent ways:

| check | what it validates | agreement |
|---|---|---|
| `astropy` BlackBody, 5 (λ, T) points from 0.5 µm / 5800 K to 100 µm / 50 K | the curve, pointwise | rel. 1e-9 |
| Wien displacement law | where its peak sits | rel. 1e-3 |
| Stefan–Boltzmann by integration | the area under it | rel. 1e-8 |

The Stefan–Boltzmann check needed log-spaced sampling. Linear sampling with
2,000,000 points still truncated enough long-wavelength tail to leave a 1.5e-4
error, where 200,000 log-spaced points give 4e-10 — loosening the tolerance
instead would have produced a test tuned to its own sampling error rather than
to the physics.

Reproduce with `python scripts/build_speclib.py && python scripts/thermal_retrieval.py
python scripts/sam_mixing_experiment.py`.

---

## 14. M3: identification needs SWIR — §3's prediction, measured

§3 measured that HAD100's *detection* signal lives at 455–700 nm, and predicted
that *identification* would need the SWIR absorption features a spectral library
encodes. That prediction is now tested — and it required a controlled experiment,
because no real scene can be re-flown with half its spectrum removed.

**Synthetic, and labelled so.** Real ECOSTRESS mineral spectra, resampled to the
AVIRIS-NG grid, mixed into a real HAD100 background at known sub-pixel abundance,
with Gaussian noise at known SNR. The spectra and the background are real; the
mixing and the noise are not. Ground truth is exact by construction, which is the
point: it is the one thing Cuprite cannot currently supply (§14.4).

### The result

Top-1 material identification accuracy, 15 materials (chance = 0.067), SNR 20 dB,
matching on the background-removed deviation:

| bands | | a = 1.0 | 0.75 | 0.50 | 0.30 | 0.20 | **0.10** | 0.05 |
|---|---|---|---|---|---|---|---|---|
| **swir** | 126 | 0.810 | 0.825 | 0.825 | 0.760 | 0.680 | **0.565** | 0.385 |
| full | 276 | 0.945 | 0.950 | 0.905 | 0.875 | 0.680 | 0.410 | 0.265 |
| vnir | 101 | 0.835 | 0.700 | 0.510 | 0.295 | 0.235 | 0.180 | 0.105 |
| **visible** | 50 | 0.730 | 0.545 | 0.420 | 0.325 | 0.205 | **0.185** | 0.150 |

**The prediction holds, and the symmetry with §3 is exact.**

| | best window | worst window |
|---|---|---|
| **Detection** (§3, RX mAUC) | visible 0.9799 | SWIR-ish 0.9162 |
| **Identification** (here, a = 0.10) | SWIR 0.565 | visible 0.185 |

The two objectives want opposite ends of the same spectrum, now measured on both
sides rather than argued on one. And the mirror runs deeper: just as adding SWIR
*hurt* detection, **adding the visible bands hurts identification** — at a ≤ 0.10,
SWIR alone (0.565) beats the full range (0.410), because the visible bands
contribute noise without contributing diagnostic features.

### Three construction errors, each of which produced a plausible table

The experiment is synthetic, so its conclusions are only as good as its
construction. Each of these was caught by a result that looked wrong:

**1. Mismatched units made the abundance axis a lie.** HAD100's crops are not
reflectance fractions — mean 2.51, against a library mean of 0.60. Mixed
unscaled, a nominal `a = 0.75` pixel draws 0.564 of its signal from the target
and **0.626 from the "25%" background**. Accuracy collapsed at every `a < 1` for
a reason with nothing to do with SAM. Caught because the collapse was a cliff
rather than a slope.

**2. A noiseless experiment is exactly solvable, and scores 1.000 everywhere.**
With exact linear mixing, `observed − bg = a·(target − bg)` is the reference
scaled by `a`, and SAM is magnitude-invariant, so the angle to the correct entry
is *identically zero*. The first band ablation duly reported perfect accuracy on
visible-only bands — which would have read as **refuting §3** while in fact
measuring nothing at all. Noise is what makes band choice matter: diagnostic
absorptions survive it and continuum shape does not.

**3. Matching the raw pixel answers a different question.** Against pure library
spectra with no background removal, accuracy falls to chance for any minority
target:

| | a = 1.0 | 0.75 | 0.50 | 0.10 |
|---|---|---|---|---|
| raw mixed pixel | 1.000 | 0.240 | 0.040 | 0.040 |
| background removed | 1.000 | 1.000 | 1.000 | 1.000 *(noiseless)* |

That is not a flaw in SAM. As abundance falls, every mixture converges on the
background *regardless of what was mixed in*, so the nearest library entry
becomes whichever sits closest to the background — the same one every trial. The
pipeline is not affected, because RX already defines an anomaly as a departure
from background; matching the deviation is what it has available. But reporting
only the second row would overstate SAM, and only the first would misdescribe
the pipeline, so both are reported.

### Noise sensitivity

Full range, background-removed:

| SNR | a = 1.0 | 0.50 | 0.20 | 0.10 | 0.05 |
|---|---|---|---|---|---|
| noiseless | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 40 dB | 1.000 | 0.995 | 0.985 | 0.975 | 0.880 |
| 30 dB | 0.995 | 0.995 | 0.945 | 0.855 | 0.575 |
| 20 dB | 0.930 | 0.920 | 0.675 | 0.395 | 0.275 |
| 10 dB | 0.875 | 0.630 | 0.260 | 0.175 | 0.160 |

Reproduce with `python scripts/sam_mixing_experiment.py`.

### 14.4 Why not Cuprite — and a trap worth naming

Cuprite, Nevada is the canonical AVIRIS mineral-mapping site and the standard
answer for identification ground truth. Access was verified on 2026-09-15 and is
**partial in the way that matters**:

| | status |
|---|---|
| Tetracorder repo (GitHub, Zenodo) — the mineral map *as data* | ✅ 200 |
| AVIRIS data portal — raw radiance | ✅ 200 |
| `ftp://ftpext.usgs.gov/.../speclab/pub/cuprite` — the calibrated cube | ❌ connection fails |
| `speclab.cr.usgs.gov/cuprite95.tgif.2.2um_map.gif` | ❌ 301, file gone |

The mineral map is obtainable; the calibrated reflectance cube it corresponds to
is not. The Tetracorder repo omits the cube for size and points at the dead FTP.
Raw AVIRIS is still available, but atmospheric correction to apparent surface
reflectance is a substantial job and the result would no longer correspond
pixel-for-pixel to the published map — which is the entire reason for wanting
Cuprite. No mirror of `cuprite.95.cal.rtgc` was found.

**The trap, worth a footnote wherever Cuprite is cited:** "Cuprite has ground
truth" is true only of the **614 × 750 Tetracorder pairing**. The 250 × 191 crop
that circulates widely in the unmixing literature is catalogued as having **no
ground truth** — it is a different subset. A paper citing "Cuprite, which has
USGS ground truth" while using the common crop is claiming validation it does not
have.

---

## 15. Provenance

Every figure above traces to a run directory or a named script. `results/` is
gitignored (it holds score maps); the run records are reproducible from the
configs and commands listed here.

| Result | Produced by | config hash |
|---|---|---|
| §1 headline, band50 | `results/band50-full_20260823-165748` | `55e48abab601` |
| §1 headline, 276 bands | `results/had100-m2-baseline_20260823-160820` | `649c82d06c59` |
| §2 GRX reproduction, §3 wavelength windows | `scripts/had100_baselines.py` + the band-window sweep | — |
| §3 published ranking | `results/had100_baselines.json` | — |
| §4–§7 ablations, 276-band α sweep | `results/m2/ablations.json` via `scripts/run_ablations.py` | — |
| §7 band50 α sweep (the one to quote) | `results/m2-band50/ablations.json` | — |
| §11 FLAME 3 baseline saturation | `scripts/thermal_baseline.py` | — |
| §13 Eq. 5 retrieval | `results/thermal/flame3_retrieval.json` via `scripts/thermal_retrieval.py` | — |
| §13 emissivity library | `results/speclib_build.json` via `scripts/build_speclib.py` | — |
| §14 SAM mixing experiment | `results/sam_mixing_experiment.json` via `scripts/sam_mixing_experiment.py` | — |
| §10 train-on-test-scene | `results/m2/train_on_test_scene.json` via `scripts/train_on_test_scene.py` | — |
| §8 Salinas zero-anomaly | `results/salinas-zero-anomaly_20260823-163509` | `86f4b5c9576a` |
| §9 corpus PCA timing, stability spectrum | `BandReducer.fit_streaming` on the 1,040-crop corpus | — |

Note that `band50-full` and `band50-rx` share a config hash: `--rx-only` is a
command-line flag, not a config field, so the resolved config is genuinely
identical and only `command.txt` distinguishes them. That is the intended
behaviour — the hash identifies the configuration, not the invocation.

### Commands

```bash
# Headline runs
python -m pipeline.cli.run_hyperspectral --config configs/had100_band50.yaml
python -m pipeline.cli.run_hyperspectral --config configs/had100.yaml
python -m pipeline.cli.run_hyperspectral --config configs/salinas.yaml

# Ablations and baselines
python scripts/run_ablations.py --out results/m2
python scripts/run_ablations.py --config configs/had100_band50.yaml     --only baseline,rx_k15_corpus --out results/m2-band50
python scripts/had100_baselines.py
python scripts/train_on_test_scene.py
```
