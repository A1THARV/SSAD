# Build Roadmap — Spectral-Spatial Anomaly Detection Prototype

Implementation plan derived from `PRD_Anomaly_Detection_Prototype.md` (Draft v1.0).
Every task cites the PRD requirement it satisfies, so the PRD stays the spec and this stays the schedule.

---

## Decisions locked before build

| PRD §15 question | Decision | Consequence for this plan |
|---|---|---|
| Compute environment | **GPU available** — RTX 3050 Laptop, 4 GB VRAM, torch 2.5.1+cu121, Python 3.12.4 | Autoencoder can exceed toy scale, but 4 GB forces **patch-based** training, not whole-cube 3D conv. CPU fallback stays a hard requirement (PRD §10). |
| Thermal dataset | **FLAME 3, CV subset** (~6.23 GB) — C8 closed 2026-08-21 | Radiometric thermal TIFFs at 640×512, fire/no-fire masks, aerial UAV. Serves M4 *and* M5 from one dataset, so ECOSTRESS L2T_LSTE, HIT-UAV, M3OT and FIRMS all drop. NADIR set (32.59 GB) deferred until the temporal path needs it. |
| Demo requirement | **Streamlit dashboard is in scope** | Phase 8 is committed work, not a stretch goal. Results-folder schema (below) must be stable from Phase 2 so the dashboard has something to read. |
| Milestone priority | **All four objectives evenly** | Hyperspectral and thermal branches interleave rather than running to completion in sequence. Shared infrastructure is built once, up front. |

### PRD v2 change brief — applied 2026-08-21

| Change | Status | Effect on this plan |
|---|---|---|
| C1 — HAD100 + HyperAD in, San Diego out, Salinas demoted | **applied** | HAD100 is the primary benchmark: 3.75 GB fetched, checksum-pinned, canonical preprocessing run |
| C2 — autoencoder trains on anomaly-free corpus | **applied to config** | `training_corpus` and `train_on_test_scene` are typed config fields; the model itself is Phase 4 |
| C3 — M2 scored against HAD100's 17 published baselines | **applied** | Baselines bundle fetched; `band_presets.py` reproduces HAD100's band selection so the comparison is like-for-like |
| C4 — ECOSTRESS primary spectral library | **applied to config** | Source entry rewritten with the resampling requirement recorded; the curated subset is Phase 4 work |
| C5 — Salinas becomes the zero-anomaly fixture | **applied** | `label_convention: zero_anomaly`; its class map is deliberately unread |
| C6 — `errors.py` additions | **applied** | `IncompatibleInputError`, `MissingGroundTruthError`, structured `DataUnavailableError` |
| C7 — codify the `auc_roc` deviation | **applied** | PRD §9 updated; `report.py` renders `n/a (single-class GT)`, enforced by test |
| C8 — thermal branch replacement | **resolved: FLAME 3** | One aerial wildfire dataset serves Objectives 3 *and* 4. Supersedes the aerial/satellite split |

### Environment fact that must be fixed first

`SSAD/` is **not its own git repository** — the enclosing repo root is `C:\Users\ACER`, currently on branch `feature/redis-caching` with commits from an unrelated dashboard project. Committing prototype code from here would write into the wrong history. Task 0.1 is `git init` inside `SSAD/`.

---

## Deviations from the PRD's build order, and why

The PRD's milestone table (§13) is sound, but five dependencies in it resolve in the wrong order. Each is corrected in the phases below.

1. **Eq. 3 depends on Eq. 4.** The composite loss `L = MSE + λ·SAM` (FR-9.4, milestone M2) calls the SAM function that the PRD schedules in M3. `spectral_angle` moves ahead of the autoencoder — Phase 4.0.
2. **`metrics.py` is a leaf dependency of both branches**, but the PRD introduces it at FR-9.11 alongside the M6 reporting work. Without it, M1 has no way to state its headline number. It moves to Phase 0.
3. **Salinas has no anomaly ground truth.** PRD §3 DoD item 1 requires a real AUC-ROC for `--dataset salinas`, but §7.1 concedes Salinas ships land-cover labels, not an anomaly mask. A labeling convention (Salinas-A subscene vs. rare-class-as-anomaly) must be chosen and written down *before* any Salinas number is quoted, or that number means nothing. Decision point in Phase 2.
4. **Data acquisition is a blocking, latency-bound task**, not a coding task, yet the PRD bundles all of it into M0. The two sources that can require manual registration (San Diego bundle, thermal equipment set) are started on day one and run as a parallel track while coding proceeds against whichever dataset landed first.
5. **4 GB VRAM caps the autoencoder.** A 3D-conv AE over a 512×217×224 Salinas cube will not fit as a single tensor. Patch extraction is part of the model design from the start, not a later optimization.

---

## Cross-cutting contracts — settle in Phase 0, painful to retrofit

These four decisions leak into every module. Fixing them once, first, is what keeps later phases from thrashing.

**Results folder schema.** Every CLI writes `results/<run_id>/` with a fixed layout: `metrics.json`, `config_used.yaml`, `config_hash.txt`, `command.txt`, `anomaly_map.png`, `roc_curve.png`, plus branch-specific extras. FR-9.11 traceability and the Phase 8 dashboard both read this contract; if it drifts, both break.

**Exception hierarchy.** PRD §10 forbids silent failures. Define `PipelineError` and typed subclasses (`DataUnavailableError`, `DegenerateInputError`, `NonRadiometricInputError`, `UnregisteredInputError`) in Phase 0. Every "must raise a clear error" requirement (edge cases 9, 14; FR-9.7) then resolves to raising an existing type rather than inventing one at the call site.

**Seed control.** One `set_seeds(cfg.seed)` touching `random`, `numpy`, `torch`, plus `torch.use_deterministic_algorithms` where it doesn't cost too much runtime. Edge case 15 (re-run reproducibility) is free if this exists from the start and unwinnable if bolted on after training code lands.

**Docstring citation convention.** Every function implementing a paper equation opens its docstring with `Implements Eq. N of the paper: <formula>`. PRD §10 traceability and DoD item 4 are then checkable with a grep, and Appendix A becomes a generated table rather than a hand-maintained one.

---

## Phase 0 — Foundation
**Satisfies:** M0 (partial) · **Effort:** ~1 session · **Blocks:** everything

| # | Task | Done when |
|---|---|---|
| 0.1 | `git init` in `SSAD/`; `.gitignore` for `data/`, `results/`, `.venv/`, `__pycache__/` | `git status` inside `SSAD/` shows a clean repo rooted here, not at `C:\Users\ACER` |
| 0.2 | `pyproject.toml`, `src/pipeline/` layout per PRD §6 | `pip install -e .` succeeds; `import pipeline` works from any cwd |
| 0.3 | `config.py` — YAML → dataclass loader, config hashing, `set_seeds()` | Round-trip test: load `salinas.yaml`, assert typed fields; same config → same hash |
| 0.4 | `errors.py` — the exception hierarchy above | — |
| 0.5 | `evaluation/metrics.py` — `DetectionMetrics`, AUC-ROC / precision / recall / F1 | Unit test on hand-computed 10-pixel case with known confusion matrix; a **zero-anomaly ground truth** returns defined values rather than a divide-by-zero (pre-empts edge case 3) |
| 0.6 | `results_io.py` — writes and reads the results-folder schema | Write-then-read round-trip test |

**Exit:** `pytest` green on config + metrics. Nothing is downloaded yet and nothing detects anything, but the spine every later phase hangs on is tested.

---

## Phase 1 — Data acquisition (parallel track, starts with Phase 0)
**Satisfies:** M0 (remainder), PRD §7 · **Effort:** ~1 session of work + unknown wall-clock on registrations

Start 1.1 and 1.4 **on day one** — they are the two that can stall on a human-in-the-loop registration, and everything downstream waits on them.

| # | Source | Method | Risk |
|---|---|---|---|
| 1.1 | San Diego Airport (scidb.cn bundle) | HTTP, may need account | **High** — primary integration-test dataset (PRD §12), smallest scene, only one with a true binary anomaly mask |
| 1.2 | ABU series | `git clone` from GitHub | Low |
| 1.3 | Salinas | ehu.eus, Zenodo mirror as fallback | Low |
| 1.4 | Thermal — electrical equipment (scidb.cn) | HTTP, may need account | **High** — sole source for M4/M5 |
| 1.5 | Curated spectral library subset | Parse ~20–30 spectra from USGS splib07 + ECOSTRESS; commit as a small `.npz` + provenance JSON | Medium — parsing effort, mitigated by PRD §14 (subset only) |

### Acquisition results (2026-08-20, revised 2026-08-21 under change C1)

| Source | Status | What arrived |
|---|---|---|
| **HAD100 — data** | **verified** | 3.75 GB, checksum-pinned. 94 annotated AVIRIS-NG source scenes plus 522 anomaly-free scenes (260 AVIRIS-NG at 100×100×425, 262 AVIRIS-Classic at 81×81×224) |
| **HAD100 — baselines** | **verified** | 166 MB, checksum-pinned. 17 published methods + 2 variants, with evaluation code |
| **HAD100 — canonical preprocessing** | **run** | Its own `main.py` produced 1,040 AVIRIS-NG training crops (64×64×276), 1,048 AVIRIS-Classic crops (64×64×162), 100 test images + 100 masks |
| **HyperAD / ERX** | **verified** | 79 MB cloned. Paper reference [6]'s own repository and `.npy` datasets |
| ABU series | **verified** | All 13 scenes, 81 MB. Binary masks, 0.08%–2.72% anomaly fraction, band counts 102/188/191/193/204/205/207 across three dtypes |
| Salinas | **verified** | 512×217×204 int16, checksum-pinned, from the Zenodo mirror (`ehu.eus` 403s every scripted request). Now the zero-anomaly fixture per C5 |
| `thermal-images-equip` | **retained as negative fixture only** | 646 images, verified 320×240 8-bit RGB false-colour — non-radiometric, which is precisely its role under C8 |
| ECOSTRESS / USGS splib07 | manual, deferred | Not needed before Phase 4 |
| ~~San Diego~~, ~~scidb thermal~~, ~~Salinas-A~~ | **dropped** | Recorded in `download_data.py`'s `REJECTED` table so they are not re-suggested |

**Findings that change later phases:**

1. **"HAD100" means 100 test *images*, from 94 source scenes.** Six scenes contribute two 64×64 crops each. Table III must say 100 images / 94 scenes rather than "100 scenes".
2. **The two HAD100 sensors are not band-compatible.** AVIRIS-NG survives band selection at 276 bands, AVIRIS-Classic at 162. So C2's "train on 522 anomaly-free scenes" cannot mean one model over both: `configs/had100.yaml` trains on the 1,040 AVIRIS-NG crops only, which is also the corpus matching the test set (every test image is AVIRIS-NG). The AVIRIS-Classic corpus is a cross-sensor extension requiring resampling — the `IncompatibleInputError` case from C6.
3. **Band selection is part of the benchmark, not a preprocessing choice.** HAD100's 17 baselines were all scored after *its* band selection, so `band_presets.py` transcribes it verbatim and refuses to apply one sensor's indices to the other's cube — every index would still be in range, so nothing downstream would catch the error.
4. **ABU keeps its Phase 2 role.** Seven band counts across three dtypes make edge case 5 testable on real data; `abu-beach-1` (19 anomaly pixels) and `abu-beach-3` (11) are real sub-pixel cases for edge case 4.

**The manual drop-in contract** (mitigates the registration risk, and edge case 14): `download_data.py` never fabricates data. For each source it (a) attempts an automated resumable fetch, (b) verifies size and checksum against a committed manifest, and (c) on failure prints the dataset page URL, the exact expected file path under `data/raw/<name>/`, and the expected checksum — then exits non-zero. A manually downloaded file dropped at that path passes verification on the next run and the pipeline proceeds identically. This makes a registration wall a five-minute detour instead of a blocked milestone.

**Exit (M0):** `python scripts/download_data.py` either populates `data/` with all checksums verified, or fails with an actionable per-source message. Salinas' 224 bands and ABU's 189–205 must both load — no hardcoded band count anywhere (edge case 5).

---

## Phase 2 — Hyperspectral statistical baseline
**Satisfies:** M1, FR-9.1/9.2/9.3, DoD 1 · **Effort:** ~2 sessions

| # | Task | PRD ref |
|---|---|---|
| 2.1 | `hyperspectral/loader.py` → `HSICube` dataclass; band-count-agnostic | FR-9.1, edge 5 |
| 2.2 | `preprocess.py` — water-absorption band mask (~1350–1450 nm, ~1800–1950 nm), SNR threshold mask, saturated/dead-pixel clipping | FR-9.1, edges 1 & 6 |
| 2.3 | PCA wrapper — fit once on background-representative pixels, `transform` many; refuses to transform before fit | FR-9.3 |
| 2.4 | `rx_detector.py` — Eq. 2 with ridge-regularized covariance inverse | FR-9.2, edge 2 |
| 2.5 | `cli/run_hyperspectral.py` → results folder, anomaly map, ROC curve | FR-9.12 |

**Tests that matter here:** RX against a hand-computed 3-pixel toy scene (exact value); RX on a synthetic Gaussian background with injected outliers scores outliers strictly higher (the statistical property, not just "it ran"); a deliberately zeroed band is excluded by `good_bands`; a homogeneous scene raises no `LinAlgError`.

**~~Decision point — Salinas anomaly labels.~~ Resolved by change C5.** No anomaly labels are derived from the land-cover map; Salinas runs as the zero-anomaly fixture instead. HAD100 and ABU ship real binary masks and need no convention.

**Exit (M1):** real AUC-ROC / precision / recall / F1 across HAD100's 100 test images and all 13 ABU scenes, plus a false-positive rate on the Salinas zero-anomaly run. First numbers that can go into the paper.

### M1 headline (2026-08-21, RX at PCA-15)

mAUC = mean per-image AUC-ROC, the same quantity HAD100's `eval_results.py` reports, so these sit on the published table's axis.

| Run | mAUC at PCA-15 | per-dataset RX ceiling |
|---|---|---|
| **HAD100, 100 test images** | **0.9749** | 0.9786 (PCA-8) |
| **ABU, 13 scenes** | **0.9627** | 0.9636 (PCA-50) |

**M2 must beat 0.9786** on HAD100 — RX's *tuned* ceiling, not its score at a convenient setting. Gap to the benchmark leader AET-KCSR (0.9948) is 0.0162.

> **✅ Cleared at M2 (2026-08-23): 0.9843** — the full fused pipeline under the benchmark's own band50 protocol, at the a-priori α = 0.5. That is +0.0057 over the bar above, and 10th of 20 in HAD100's published band50 ranking. Remaining gap to AET-KCSR: 0.0105. Full tables in `RESULTS.md`.

### PCA dimensionality ablation — and the regularisation artefact underneath it

The first sweep, run at `ridge=1e-6`, produced a U-shaped curve on HAD100: a peak at 8, a trough at 50, a partial recovery by 120. **That U-shape is an artefact of under-regularisation, not a property of dimensionality.**

RX weights every direction by `1/λᵢ`, so trailing low-variance PCA components contribute enormous noise-dominated terms unless damped. Under-regularising therefore looks like a penalty for using more components. Re-running as a 2-D grid over ridge × width shows it directly:

**HAD100 mAUC, ridge × PCA width**

| ridge | K=15 | K=30 | K=50 | K=80 | K=120 | K=200 |
|---|---|---|---|---|---|---|
| 1e-6 | 0.9749 | 0.9671 | 0.9643 | 0.9684 | 0.9720 | — |
| 3e-3 | 0.9789 | 0.9789 | 0.9783 | 0.9784 | 0.9787 | 0.9777 |
| 1e-2 | 0.9768 | 0.9786 | 0.9796 | 0.9805 | **0.9809** | 0.9804 |
| 3e-2 | 0.9720 | 0.9757 | 0.9777 | 0.9794 | 0.9805 | **0.9813** |

At `ridge ≤ 1e-6` (including exactly 0) the trough is stable and reproducible. By `ridge=1e-2` it has **vanished and inverted** — accuracy now rises monotonically with width. ABU shows monotone improvement in width at every ridge tested. So the honest conclusion is the opposite of the first reading: **more components are better on both datasets, once regularised properly.**

The optimal ridge grows with width, which is the signature of covariance-estimation error — at K=200 the detector estimates a 200×200 covariance from 4,096 pixels. That is shrinkage territory, so `rx_score` now offers `shrinkage="ledoit_wolf"`, an analytically-derived alternative with **no tunable parameter**: 0.9635 on ABU (matching its tuned optimum exactly) and 0.9783 on HAD100 (within 0.003 of tuned). Since the tuned ridge does *not* transfer between the two datasets, Ledoit-Wolf is the defensible default when one setting must serve both.

**Reported honestly:** PCA-15 was chosen with both curves already visible. It is not a clean held-out selection, and should not be described as one. Its strongest justification is architectural and independent of the data — Fig. 1's first encoder kernel has spectral depth 7, so `K` must comfortably exceed 7, which rules out HAD100's nominal 8-component peak on grounds that owe nothing to the test set.

HAD100 does not document how `band50/100/200` were produced, so PCA-*n* is not necessarily their band-*n* — state results as "PCA-*n*".

### Discussion thread for the paper: HAD hyperparameters do not transfer across scenes

Three independent demonstrations of one claim, which is worth writing up as a single argument rather than three scattered observations. Each is specific, reproducible from this repo, and unflattering — which is what makes the thread credible.

**1. Optimal dimensionality is scene-dependent.** At a fixed regularisation, HAD100 and ABU disagree about how many PCA components to keep, and the disagreement is not marginal. Even after the regularisation artefact above is corrected, the *ridge* itself does not transfer: HAD100 is best at 1e-2–3e-2, ABU at ≤1e-3, and each dataset's optimum is materially worse on the other.

**2. Fixed-percentile thresholding measures scene composition, not detector quality.** Across ABU at p99, F1 spans 0.09–0.67 while true anomaly fractions span 0.08%–2.72%. A percentile flags 1% of pixels regardless, so F1 largely reports the mismatch between that fixed 1% and the scene's actual anomaly rate.

**3. The zero-anomaly case proves it.** On Salinas — a scene with **no anomalies at all** — p99 flagged **1,112 pixels, 1.00% of the scene, every one a false positive**. Otsu flagged 0.22%, 4.5× fewer, because it responds to the histogram rather than guaranteeing a quota. This closes §11 edge case 3 on real data.

Taken together these argue for adaptive, scene-responsive settings over globally-tuned constants — which is what the paper's Fig. 1 (`T = μ + γσ`) and §V (F1-optimal on a held-out split) already specify, and what `shrinkage="ledoit_wolf"` provides on the regularisation side.

**Default corrected:** percentile is the *worst* option for the zero-anomaly case, not the most honest one. Otsu and `mean_std` both respond to the score distribution; percentile cannot. `abu_urban.yaml` and `had100.yaml` now default to `otsu`.

### Paper edits logged (not code changes)

| Item | Action |
|---|---|
| Fig. 1 box 4c draws `d_RX = √((x−μ)ᵀΣ⁻¹(x−μ))`; Eq. 2 has no square root | **Change the figure.** Eq. 2 matches Reed & Yu's original squared form. AUC is unaffected (monotone), but Fig. 1 stage 5 normalises by max before fusing, and `√S/max(√S) = √(S/max S)` compresses mid-range values upward — so the fusion balance and any distribution-based threshold genuinely shift. `rx_score` stays as Eq. 2. |
| ~~Fig. 1 shows one PCA block feeding both the autoencoder and the RX branch~~ | **CLOSED — no edit needed. The figure was right.** Measured at M2: corpus-basis RX beats per-scene RX by **+0.0057 mAUC** at K=15 (0.9806 vs 0.9749) and is the best RX result anywhere in this work. The "keep them separate" reasoning below was wrong and is struck. `rx.pca_basis: corpus` is now the default. |
| Table III wording | "100 test images from 94 source scenes". |
| **Fig. 3 draws CNN -> LSTM -> anomaly heatmap** | **Redraw or caption it.** Same class of defect as Fig. 1's spurious square root: the figure depicts a pipeline that was not built and that the measurement says is unnecessary on this data. M4 ships as a physical threshold (AUC 1.0000, 61 C margin), so the drawn CNN->LSTM stack has no counterpart in the code. Two honest options: redraw Fig. 3 as the threshold detector actually used, or keep it and caption it as the architecture intended for fire-spread forecasting on the NADIR sequences (future work, not evaluated here). Do NOT leave it uncaptioned -- a reader matching figure to results finds a model that does not exist. |
| **Discussion paragraph** | Add the consolidated preprocessing finding (RESULTS.md §12): three independent instances, across two modalities, of a standard preprocessing step normalising away the physically meaningful quantity. Write once with all three as evidence -- variance-retained PCA, SAM on PCA scores, per-frame thermal normalisation. This is the most generalisable result in the work and does not depend on hyperspectral imaging at all. |
| §VII, M4 claim | Must read as image-level fire **presence**, with the explicit second sentence that pixel-level localisation was not evaluated because the CV subset ships no masks. Without it the AUC 1.0000 reads as a localisation claim. |
| §V, deep-learning claim | The fused system (0.9843) beats AutoAD / LREN / RGAE. The **autoencoder alone is 0.9619, below AutoAD's 0.9688** — attribute the win to the fused pipeline, never to the deep component. |
| §III / §VI, Objective 2 | State *before* M3 runs: HAD100's signal is at 455–700 nm, while SAM library matching depends on SWIR absorption features (2200–2350, 1400, 1900 nm) — the windows measured at 0.84–0.92 here. A weak SAM result on HAD100 indicts the benchmark's fit to Objective 2, not Eq. 4. Said in advance it is a prediction; said afterwards it is an excuse. Note also that the band50 config discards every SWIR band, so it cannot serve Objective 2 at all. |
| §IV, whitening | Promote out of the ablation table into a **worked example**: the stability spectrum was measured first for an unrelated reason, produced a falsifiable prediction about a parameter nobody had measured, and held at both widths. That demonstrates a mechanism rather than a tuning surface, and it transfers to other sensors where a swept optimum does not. |
| §V comparison table | Report against HAD100's **band50** column — that is where all 19 baselines are strongest, so it is the hardest bar. State that band50 = the first 50 of the 276 selected bands = 455–700 nm; the benchmark never defines it and we recovered it by reproducing GRX to 4 dp. |
| §IV / §V, band protocol | Add: our RX reproduces the benchmark's own GRX baseline exactly (0.9799 / 0.9714 at band50 / band100), which verifies Eq. 2 against a published reference implementation. Worth a sentence — it is the strongest correctness evidence in the paper. |

### Corpus PCA is fitted exactly, not sampled — and the stability spectrum explains the width ablation

`BandReducer.fit_streaming` accumulates the `B×B` covariance in two passes over memory-mapped crops. PCA never needs the pixel matrix, only the covariance: at 276 bands that is **0.61 MB resident**, so the 1,040-crop corpus is fitted exactly rather than sampled. Measured on the real corpus — **84 s, 4,259,840 pixels, RSS +0.02 GB** against the 4.70 GB an eager load would take. There is no sampling fraction to justify.

Two passes deliberately: the one-pass identity `Cov = E[xxᵀ] − μμᵀ` cancels catastrophically when the mean dwarfs the variance, which is where reflectance data sits.

**The finding that matters.** Fitting two disjoint halves of the corpus and measuring principal angles between the resulting 15-dimensional bases:

| components | 1–6 | 7–10 | 11–15 |
|---|---|---|---|
| principal angle | < 2.2° | 3.7–8.7° | 13.0–87.0° |
| λ_k / λ_1 | ≥ 1.6e-3 | 1.7e-4 – 1.0e-3 | ≤ 1.1e-4 |

Only the leading ~8 directions are reproducible. Beyond that the "components" are different directions in different halves — sample noise, not scene structure.

This gives the width ablation a mechanism rather than leaving it an empirical curiosity. HAD100's RX accuracy peaked at **k ≈ 8–12**, and subspace stability collapses at **k ≈ 8–10**. The two coincide because components past that point carry no reproducible signal. It is also why eigenvalue agreement is the wrong sufficiency check and subspace angle is the right one: the eigenvalues stay smooth while the eigenvectors rotate freely.

**Consequence for whitening.** `λ₁/λ₁₅ = 22,076×` on this corpus, so an MSE-trained autoencoder on raw scores would be almost entirely driven by PC1 — the concern is real. But whitening divides by `√λ`, promoting components 11–15 to equal weight, and those are precisely the directions that do not reproduce across halves. So whitening at K=15 would amplify sample noise. The `whiten_input` ablation at M2 should therefore be run **paired with width**, not at fixed K=15: whitening is most defensible near the stability limit (K ≈ 8), least so above it.

> **✅ Prediction confirmed at M2 (2026-08-23).** Run paired with width as specified, the sign flips exactly where the stability spectrum said it would:
>
> | K | latent | whiten off | whiten on | Δ |
> |---|---|---|---|---|
> | 15 | 8 | **0.9522** | 0.9472 | **−0.0050** |
> | 8 | 4 | 0.9217 | 0.9426 | **+0.0209** |
>
> At K=8 every retained direction reproduces, so whitening does only what it was meant to — remove PC1's dominance of the MSE — and gains 0.0209. At K=15 it promotes sample noise to equal weight and loses 0.0050. Had this been run at fixed K=15, as originally planned, whitening would have been recorded as simply harmful and the mechanism missed entirely.
>
> Cross-K rows carry a confound (K=8 forces `latent_dim` 8→4 to keep `d ≪ K`); the **within-K** comparison is the clean one, and that is where the sign flip lives.

### PCA basis: settled at M2 — one path, and Fig. 1 was right

**Autoencoder — corpus-fit, mandatory.** Not a preference: train and inference must share fixed input axes, or the encoder meets a different basis at test time.

**RX — corpus-fit too, on the evidence.** The prediction below was that per-scene axes would suit RX, because RX is by construction a per-scene detector and corpus axes would lead on *inter*-scene variance rather than the intra-scene structure that makes a target stand out. Measured on all 100 HAD100 test images:

| RX basis | K = 8 | K = 15 |
|---|---|---|
| per-scene | 0.9786 | 0.9749 |
| **corpus** | 0.9782 | **0.9806** |

At K = 8 the two are indistinguishable (0.0004). At K = 15 the corpus basis wins by **+0.0057** and produces the best RX number obtained anywhere in this work — it also clears the published GRX band50 figure of 0.9799.

**Why the argument failed.** It confused *estimating Σ* with *choosing the axes Σ is expressed in*. RX does need per-scene background statistics, and it still gets them — `background_statistics` is computed on the scene either way. What the basis controls is only which 15 directions survive truncation, and there a corpus fit on 4.26 M pixels from 1,040 scenes simply has more data than a scene fit on 4,096 pixels. The gap opens with width because by K = 15 the per-scene fit is resolving directions it lacks the samples for, and Eq. 2 then inverts a covariance expressed in a partly arbitrary basis. At K = 8 both bases have enough data, which is exactly why the gap closes there.

So the branches unify, Fig. 1 is true as drawn, and the second-PCA-path edit comes off the paper list.

**Correction to an earlier note in this file:** separate bases were said to "complicate stage-5 fusion". That was overstated. Fusion is score-level — `E_rec/max(E_rec)` and `d_RX/max(d_RX)` are per-pixel scalars on the same spatial grid, and they combine correctly regardless of the bases that produced them.

**Correction to an earlier note in this file:** separate bases were said to "complicate stage-5 fusion". That was overstated. Fusion is score-level — `E_rec/max(E_rec)` and `d_RX/max(d_RX)` are per-pixel scalars on the same spatial grid, and they combine correctly regardless of the bases that produced them. The only casualty is the figure, not the mathematics.

---

## Phase 3 — Thermal baseline ✅ SHIPPED AS A MEASUREMENT
**Satisfies:** M4, FR-9.7/9.8 · **Effort:** ~1 session (spent) · **Unblocked:** FLAME 3 verified 2026-09-04

**Status 2026-09-14.** 3.1 (loader + radiometric classification) and 3.2
(calibration, clip reporting, normalisation) are built and tested. **3.3-3.5 are
deliberately not built**, on evidence rather than for time: image-level fire
presence is saturated by a one-line physical baseline -- max pixel temperature
gives AUC 1.0000 over all 738 frames with a 61 C margin, holding to the smallest
5% of fires by burning area. A CNN can at best tie that, and an LSTM has no
sequences to consume in the CV subset. M4 therefore ships as the measurement
(RESULTS.md §11), and the effort moves to M5 where radiometry does something a
threshold cannot.

Task 3.6's synthetic generator stays cancelled, and 3.5's temporal path is now
**future work**: fire-spread forecasting on the NADIR plot sequences is the
version of the temporal task that is not saturated, but it is 32.59 GB and a
different paper.

The original hold notice follows, retained for the record.

**Do not implement.** PRD §7.3 is withdrawn: its datasets are ground-level equipment inspection, which is not remote sensing, and the paper's title claims remote sensing. The replacement is an owner decision between aerial thermal (serves Objective 3, likely non-radiometric so M5 stays blocked) and satellite thermal (serves Objective 4 and Eq. 5 directly, but at 70 m–375 m rather than drone scale). The decision edits the paper, not only the code.

The task table below is retained because it survives the dataset choice — only the loader specifics change.

| # | Task | PRD ref |
|---|---|---|
| 3.1 | `thermal/loader.py` — reads the dataset; **classifies each input at load time as radiometric temperature map vs. 8-bit visualization-only** and carries that flag on the data object | FR-9.7, edge 9 |
| 3.2 | `preprocess.py` — DN→temperature per documented gain/offset, frame registration for multi-frame input | FR-9.7 |
| 3.3 | CNN spatial backbone, per-frame anomaly score | FR-9.8 |
| 3.4 | `cli/run_thermal.py` → same results-folder contract | FR-9.12 |
| 3.5 | LSTM/GRU temporal head + graceful degradation to CNN-only below `min_seq_len` (default 3), with a log line stating it degraded; frame-dropout gap handling | FR-9.8, edges 8 & 10 |
| 3.6 | ~~Synthetic heating-sequence generator~~ — **likely unnecessary**: the transformer load ladder in `thermal-images-equip` is a real heating progression. Keep the generator only as a fallback if the ladder proves unusable | §7.3 |

**The honesty requirement is a code requirement, not a writing one.** Per PRD §7.3, synthetic sequences must be tagged as synthetic in the data object itself and that tag must propagate into `metrics.json` and `RESULTS.md`. Real single-frame CNN results and synthetic-sequence LSTM results are reported as separate rows, never averaged into one number.

The radiometric flag from 3.1 is what makes edge case 9 enforceable: Phase 5's Planck module reads the flag and raises `NonRadiometricInputError` rather than computing a temperature from 8-bit display values.

**Exit (M4):** real AUC-ROC / F1 for fault vs. normal on the electrical-equipment dataset, reported as a single-frame CNN result.

---

## Phase 4 — Hyperspectral deep pipeline
**Satisfies:** M2, FR-9.4/9.5 · **Effort:** ~2–3 sessions

| # | Task | PRD ref |
|---|---|---|
| 4.0 | ✅ **Done** — `sam.py`: Eq. 4 plus resampling and `match_material` | FR-9.6 — **pulled ahead**, the composite loss depends on it |
| 4.1 | ✅ **Done** — `reconstruction_error`, Eq. 1, pure NumPy and torch-free | FR-9.4 |
| 4.2 | ✅ **Done** — `composite_loss`, Eq. 3, reusing 4.0's atan2 form in torch | FR-9.4 |
| 4.3 | ✅ **Done** — Fig. 1's 3D-conv encoder/decoder, patch-based; geometry validated against the figure before any VRAM is spent | FR-9.4, §14 risk |
| 4.4 | ✅ **Done** — `score_fusion.py`, weighted/max over normalised branches | FR-9.5 |
| 4.5 | ✅ **Done** — trained on the 1,040-crop anomaly-free corpus | change C2 |
| 4.6 | ✅ **Done** — and it partly refutes C2's *reasoning*, see below | change C2 |
| 4.7 | ✅ **Done** — all 19 baselines recomputed from the benchmark's shipped result matrices with **our** metric code, not transcribed | change C3 |

**Status: M2 met.** See `RESULTS.md` for the full tables. Headline on HAD100 (100 test images, mAUC):

| Branch | 276 bands | band50 protocol |
|---|---|---|
| RX, per-scene basis | 0.9749 | — |
| RX, corpus basis | 0.9806 | 0.9835 |
| Autoencoder alone | 0.9522 | 0.9619 |
| **Fused (α = 0.5)** | 0.9778 | **0.9843** |

α = 0.5 was fixed in config before any ablation ran, so nothing in the headline was selected on the test set. The α sweep is reported as post-hoc sensitivity only.

**Correction to an earlier entry in this file.** The first α sweep ran on `had100.yaml`, where α = 1.0 is *per-scene* RX at 0.9749 — a weaker branch than the headline config uses. Re-run on `had100_band50.yaml` (corpus RX, 0.9835) the optimum shifts **0.75 → 0.90** and the gain at it shrinks **+0.0064 → +0.0048**, both as predicted: a stronger RX branch needs less help and tolerates less interference. More importantly, fusion at the *reported* α = 0.5 buys **+0.0008, not the +0.0029 first reported** — that figure was flattered by its baseline. The post-hoc ceiling is 0.9882 at α = 0.90 and is **not claimed**, since α was read off the reported test images.

**The strongest result is not the headline number.** Our RX reproduces HAD100's own GRX baseline to four decimal places (0.9799 / 0.9714 at band50 / band100 against their 0.9799 / 0.9714), which verifies Eq. 2 against an independent published implementation on a public benchmark. It also recovered what the benchmark's undocumented `band50` / `band100` / `band200` columns actually mean — the **first N** bands, which nothing in the shipped code states.

**Change C2 survives, but not on the argument it was made with.** Task 4.6 held everything fixed except which pixels the network saw:

| training data | patches | models | mAUC | wall clock |
|---|---|---|---|---|
| the test scene itself (Auto-AD style) | 4,096 × 100 | 100 | 0.9499 | 42 min |
| corpus, **matched** budget | 4,096 | 1 | 0.9418 | 24 s |
| corpus, full budget | 40,000 | 1 | **0.9522** | 232 s |

At matched data volume **training on the test scene wins by +0.0080**. The contamination argument does not hold here: HAD100's anomalies are under 1% of the training patches, so the model barely sees them while learning that scene's exact background. What makes corpus training better is *data volume*, not purity — and a ~10× cost advantage that grows with the test set. The paper should justify C2 on those grounds instead, and state the limitation: the original argument would hold where anomalies occupy a meaningful fraction of the scene, which this benchmark does not test.

**And band50 turns out to be a wavelength effect, not a dimensionality one.** All 19 baselines score best there; the first 50 bands are 455–700 nm, the visible. Holding the count at 50 and sliding the window gives 0.9799 (visible) against 0.8423–0.9162 elsewhere — a 0.14 mAUC spread on identical machinery. Measured on the labelled test images (HAD100 ships no validation split) and reported with that caveat, not as a held-out selection.

**SAM built ahead of the autoencoder (task 4.0, complete).** Eq. 4 is computed as `2·atan2(‖u−v‖, ‖u+v‖)` on unit-normalised spectra, not as `arccos(u·v)`. The two agree in exact arithmetic, but arccos has infinite derivative at zero angle, so a cosine good to 1e-16 yields an angle good only to ~1.5e-8. Eq. 3's composite loss operates precisely there — a well-reconstructed patch has near-zero angle — so the arccos form would feed numerical noise into the training signal. With atan2 an identical spectrum returns exactly 0.0.

Two forms exist for two callers: `spectral_angle_paired` is O(N·B) for the per-pixel loss, `spectral_angle_batch` is the O(N·M·B) cross-product for library matching. `resample_to_grid` handles change C4's real work and **refuses to extrapolate** — edge-clamping would invent flat spectral shape outside the library's measured range, which SAM, being a shape comparison, would read as genuine similarity.

**Magnitude invariance is load-bearing and tested as such.** The atan2 identity holds *only* for unit vectors — on raw spectra it silently returns a wrong angle rather than failing, e.g. `θ((1,0),(2,0))` would give 0.6435 instead of 0. Since §III-B justifies SAM precisely by its insensitivity to illumination and topographic shading, a missing normalisation would make that claim false in the code. Verified: normalisation is applied on all three entry points, and invariance now has tests at **non-zero** angles (`θ(u,v) = θ(u,7v)`), not only the zero-angle case, which can pass by accident.

Validated on real ABU spectra: anomaly mean vs background mean 0.0848 rad (4.86°), bit-for-bit identical whether or not the inputs are pre-normalised — so it measures shape, not brightness. The anomaly is 1.26× brighter than background, which unnormalised would have inflated the angle to 0.2429 rad. Background self-angle exactly 0.0; per-pixel separation 1.88×.

**Runtime budget check:** PRD §10 caps the full 3-scene hyperspectral run at 20 minutes CPU-only. Patch size, latent dimension, and epoch count are the levers. Measure CPU-only runtime as soon as 4.3 trains at all — if it blows the budget, shrink the model then, not after tuning.

**Report the truth.** PRD §14 anticipates the autoencoder underperforming RX on scene-limited AVIRIS data. If the AUC-ROC delta is negative, that is the finding and it goes in `RESULTS.md` with the comparison intact. FR-9.5's acceptance test asserts fusion doesn't silently regress below the RX baseline — a regression there is a reportable result, not a test to loosen.

**Exit (M2), rewritten by change C3:** the pipeline positioned within HAD100's published 17-baseline ranking, plus the background-only versus train-on-test-scene comparison. This upgrades Table III from "ours vs. RX vs. one DL method" to "ours vs. 17 published methods on a standard benchmark" without re-implementing a single competitor.

---

## Phase 5 — Material characterization, both branches
**Satisfies:** M3 + M5, FR-9.6/9.9 · **Effort:** ~2 sessions

| # | Task | PRD ref |
|---|---|---|
| 5.1 | `match_material` — top-3 by spectral angle over the Phase 1.5 curated library, always returning angles alongside names | FR-9.6 |
| 5.2 | Low-confidence flag above a configurable angle threshold | Edge 7 |
| 5.3 | `planck_radiance` — Eq. 5's `B(λ,T)` with `h`, `c`, `k` as named constants, units in the docstring | FR-9.9 |
| 5.4 | `retrieve_temperature` — inverts Eq. 5 given assumed `ε`; output names which mode was used (assumed-ε vs. TES/split-window approximation) | FR-9.9 |
| 5.5 | Emissivity reference table + documented default range with stated uncertainty for unlisted materials | Edge 11 |
| 5.6 | Wire both into their branch CLIs: SAM on hyperspectral anomaly mean spectra, Planck on thermal anomaly regions | M3, M5 |

**The validation that anchors this phase:** at `ε = 1`, `planck_radiance` must match `astropy.modeling.physical_models.BlackBody` at 2–3 known (λ, T) points within numerical tolerance. Self-consistency tests can't catch a wrong constant or a units error; an external reference can.

SAM's own sanity tests: a library spectrum matched against itself returns θ ≈ 0; water vs. metal returns a large angle.

**Exit:** M3 — top-3 material report for ≥5 detected anomalies across the 3 hyperspectral scenes. M5 — blackbody test passing, one worked material-ID example on real thermal data.

---

## Phase 6 — Fusion and generated reporting
**Satisfies:** M6, FR-9.10/9.11, DoD 3 & 6 · **Effort:** ~1–2 sessions

| # | Task | PRD ref |
|---|---|---|
| 6.1 | `fusion/fuse.py` — **independent mode as default**: both branches side by side in one document, explicitly labeled *not spatially fused* | FR-9.10, edge 12 |
| 6.2 | Co-registered mode — pixel-level score fusion, available when the user supplies aligned imagery; raises `UnregisteredInputError` if asked to fuse unaligned inputs at pixel level | FR-9.10 |
| 6.3 | Single-modality degradation — clearly labeled single-branch report | Edge 13 |
| 6.4 | `evaluation/report.py` → generates `RESULTS.md`: one table per dataset, formatted as a drop-in replacement for the paper's Table III | FR-9.11 |
| 6.5 | Every number carries its run: command line + config hash embedded per table | FR-9.11 |

`RESULTS.md` is generated, never hand-edited — DoD item 6. Add it to `.gitignore`? No: commit it, so the paper's numbers are diffable across runs, but never touch it by hand.

**Exit (M6):** one end-to-end fused report from a real hyperspectral run and a real thermal run, in independent mode.

---

## Phase 7 — Edge-case hardening
**Satisfies:** M7, PRD §11 · **Effort:** ~1–2 sessions

Building the phases above in the stated order means most of §11 is already handled by construction. This phase writes the automated test for each row — PRD §11 requires a test, not just correct behavior — and closes the remainder.

| Already handled by construction | Needs new work here |
|---|---|
| 1 band masking (2.2) · 2 singular covariance (2.4) · 3 zero-anomaly GT (0.5) · 5 band count (2.1) · 6 dead pixels (2.2) · 7 SAM low confidence (5.2) · 8 short sequence (3.5) · 9 non-radiometric (3.1 + 5.3) · 10 frame dropout (3.5) · 11 unknown emissivity (5.5) · 12 non-co-registered (6.1) · 13 single modality (6.3) · 14 missing data (Phase 1 contract) | **4** — sub-pixel / 1–2 px anomalies: report small-object and large-object metrics as **separate rows**, never averaged. Needs a size-stratified evaluation path in `metrics.py`. **15** — reproducibility: run the same config twice, assert metrics match within tolerance. Free if Phase 0.3 seeding holds; this test proves it. |

`scripts/generate_edge_cases.py` programmatically corrupts real data (zero a band, saturate pixels, drop a frame, flatten a scene to near-singular covariance) rather than shipping hand-crafted fixtures that drift.

**Exit (M7):** `pytest tests/` green including `tests/edge_cases/`; coverage on `src/pipeline/` tracked against the 80% target.

---

## Phase 8 — Dashboard and reproduction polish
**Satisfies:** M8, FR-9.12 · **Effort:** ~1–2 sessions

| # | Task |
|---|---|
| 8.1 | Streamlit app — pick a `results/<run_id>/`, render anomaly map, ROC curve, material-ID table, and for thermal runs the radiometric/synthetic labels |
| 8.2 | `README.md` — clone → `download_data.py` → three CLI commands → same headline numbers |
| 8.3 | Clean-machine rehearsal: fresh venv, fresh `data/`, follow the README literally, confirm the numbers reproduce |

The dashboard only reads the results-folder schema from Phase 0.6 — it holds no analysis logic, so a broken dashboard can never invalidate a result.

**Exit (M8, and Definition of Done):** a third party clones, downloads, runs three commands, and gets the same numbers.

---

## Critical path and risk

```
Phase 0 ──┬── Phase 2 (HS baseline) ── Phase 4 (HS deep) ──┐
          │                                                ├── Phase 6 ── 7 ── 8
Phase 1 ──┴── Phase 3 (thermal) ────── Phase 5 (materials)─┘
   ▲
   └─ start day one; 1.1 and 1.4 can stall on registration
```

Total: roughly **13–17 focused sessions**, assuming no dataset stalls.

| Risk | Trigger to watch for | Response |
|---|---|---|
| ~~scidb registration blocks San Diego or thermal data~~ — **occurred** | Confirmed in the 2026-08-20 acquisition run | Hyperspectral: ABU is the integration dataset, Phase 2 unblocked. Thermal: `thermal-images-equip` covers M4. Only M5's quantitative emissivity retrieval still needs the scidb set, because no verified-radiometric substitute exists |
| Autoencoder blows the 20-minute CPU budget | First timed CPU run in Phase 4.3 | Shrink patch size / latent dim / epochs via config — the levers exist because 4.3 makes them config values |
| AE underperforms RX | Phase 4 exit | Report it. PRD §14 names this a legitimate finding; the comparison stays in `RESULTS.md` |
| No real thermal sequences with temporal anomaly onset | Known now, confirmed in Phase 3 | Single-frame CNN is the headline real-data result; LSTM-on-synthetic is a separate, labeled secondary row |
| Salinas labeling convention chosen late or informally | Phase 2 decision point | Lock it in config before the first Salinas number is quoted |

---

## Follow-up outside this codebase

Once `RESULTS.md` exists, Table III in the conference paper is updated from it — a paper task, not a code task (PRD §15.4). The §7.3 data-gap caveat and any negative AE-vs-RX delta belong in the paper's revised Section VII as well.
