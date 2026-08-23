# Technical PRD — Spectral-Spatial Anomaly Detection Prototype
### AI/ML-Based Anomaly Detection in Hyperspectral and Thermal (IR) Remote Sensing Datasets

| Field | Value |
|---|---|
| Document type | Product/Technical Requirements Document (PRD) for a research prototype |
| Companion artifact | *AI/ML Based Spectral-Spatial Anomaly Detection in Hyperspectral and Thermal (IR) Remote Sensing Datasets* (conference paper, same project) |
| Intended executor | Claude Code (or any coding agent/engineer) working from this spec |
| Status | Draft v1.0 — ready for implementation planning |
| Owner | Atharv Donadkar + team (JD College of Engineering & Management) |

---

## 1. Purpose

The conference paper proposes a four-objective pipeline (hyperspectral anomaly detection, spectral signature matching, thermal anomaly detection, and thermal emissivity-based material ID) but contains no executed experiments — Section VII ("Results and Discussion") is explicitly a placeholder. This PRD defines a **runnable, testable prototype** that implements the paper's stated algorithms against real public datasets, produces real metrics (AUC-ROC, precision, recall, F1) to replace the paper's placeholder tables, and is robust to a defined set of edge cases.

This is a **research prototype**, not a production system. The goal is to prove the pipeline works end-to-end on real data, at small-to-medium scale, with defensible quantitative results — something that can be demoed, screenshotted into the paper's results section, and defended in a viva/presentation.

## 2. Goals and Non-Goals

**Goals**
- Implement every equation in the paper (reconstruction error, RX detector, composite loss, SAM, Planck's law/emissivity) as real, tested code — not pseudocode.
- Run the hyperspectral branch on real AVIRIS scenes (Salinas, San Diego, ABU series) with ground-truth anomaly masks, and report real AUC-ROC/precision/recall/F1.
- Run the thermal branch on a real thermal-imagery dataset with labeled hotspots/faults, and report the same metrics.
- Demonstrate the fusion + material-characterization output on at least one worked example end-to-end.
- Handle a defined list of edge cases gracefully (Section 10) rather than crashing or silently producing garbage.
- Be reproducible: one command reproduces each headline number.

**Non-Goals (out of scope for this prototype)**
- Production-grade deployment, authentication, multi-tenant serving, or cloud infrastructure.
- Real-time / onboard-edge-hardware inference (the paper cites ERX for that; this prototype targets a laptop/workstation with optional GPU, not a Jetson).
- Training on proprietary/classified defense datasets — everything must run on publicly downloadable data.
- Building a labeled thermal dataset from scratch (drone flights, licensing) — the prototype uses an existing public thermal dataset (Section 8.2), with a clear "how to swap in your own drone data later" extension point.
- A polished consumer UI. A CLI + optional Streamlit dashboard is sufficient.

## 3. Success Criteria (Definition of Done)

The prototype is "done" when all of the following are true:

1. `python -m pipeline.run_hyperspectral --dataset salinas` (and `san_diego`, `abu_urban`) produces an anomaly map, an AUC-ROC/PR/F1 report, and a saved figure — using real downloaded data, not synthetic stand-ins.
2. `python -m pipeline.run_thermal --dataset <thermal_ds>` produces a per-frame anomaly heatmap, a classification of the anomaly's likely material via the Planck/emissivity module, and a metrics report.
3. `python -m pipeline.run_fusion --hs <scene> --thermal <sequence>` produces one fused report combining both branches (even when the two inputs are not spatially co-registered — see FR-9.10).
4. All five paper equations have a corresponding, unit-tested function in the codebase, each docstring-linked to its equation number in the paper.
5. The edge-case test suite (Section 11) passes — each case has an automated test, not just a manual check.
6. A `RESULTS.md` is generated (not hand-written) containing the real metrics tables that will replace Table III (currently "TBD") in the paper.
7. `pytest` passes in CI (or locally) with no network access required beyond a one-time `scripts/download_data.py` step.

## 4. High-Level Architecture

```
                        ┌────────────────────────┐
                        │   scripts/download_data │  (one-time, cached)
                        └────────────┬────────────┘
                                     │
        ┌────────────────────────────┴────────────────────────────┐
        │                                                          │
┌───────▼────────┐                                        ┌────────▼────────┐
│ HYPERSPECTRAL   │                                        │ THERMAL BRANCH  │
│ BRANCH          │                                        │                 │
│ ─────────────── │                                        │ ─────────────── │
│ 1. Loader       │                                        │ 1. Loader       │
│ 2. Preprocess   │                                        │ 2. Preprocess   │
│    (band mask,  │                                        │    (radiometric │
│    PCA)         │                                        │    calib, reg)  │
│ 3. RX detector  │                                        │ 3. CNN spatial  │
│    (statistical │                                        │    features     │
│    baseline)    │                                        │ 4. LSTM temporal│
│ 4. Spectral-    │                                        │    model        │
│    spatial      │                                        │ 5. Anomaly      │
│    autoencoder  │                                        │    heatmap      │
│ 5. Score fusion │                                        │ 6. Planck's law │
│    (RX + AE)    │                                        │    + emissivity │
│ 6. Threshold →  │                                        │    retrieval    │
│    anomaly mask │                                        │ 7. Material     │
│ 7. SAM material │                                        │    match        │
│    ID (USGS/    │                                        │                 │
│    ECOSTRESS)   │                                        │                 │
└───────┬─────────┘                                        └────────┬────────┘
        │                                                            │
        └─────────────────────────┬──────────────────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │  FUSION MODULE  │
                          │  (decision-level│
                          │  score align +  │
                          │  weighted fuse) │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  EVALUATION +   │
                          │  REPORTING      │
                          │  (metrics.py →  │
                          │  RESULTS.md)    │
                          └─────────────────┘
```

This maps directly onto Figures 1–4 of the paper: the hyperspectral branch is Fig. 1 + Fig. 2 (SAM), the thermal branch is Fig. 3, and the fusion module is Fig. 4.

## 5. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Entire hyperspectral/DL ecosystem is Python; matches every cited paper's likely implementation language |
| Deep learning | PyTorch 2.x | Standard for research-grade autoencoders/CNN-LSTM; easy CPU fallback (`device="cpu"`) |
| Hyperspectral I/O | `spectral` (SPy), `scipy.io` (for `.mat` scene files), `h5py` | SPy is the de facto Python hyperspectral library; Salinas/Pavia/San Diego ship as `.mat` |
| Thermal I/O | `opencv-python`, `Pillow`, `flirpy` (if radiometric FLIR files are used) | flirpy extracts real temperature arrays from radiometric FLIR JPEGs/SEQs, not just 8-bit visuals |
| Numerics/classic ML | `numpy`, `scipy`, `scikit-learn` (PCA, `roc_auc_score`, `precision_recall_fscore_support`) | Avoid reinventing PCA/metrics |
| Config | `pyyaml` + `dataclasses` (no Hydra — keep it simple for a prototype) | One YAML per dataset/experiment, loaded into typed config objects |
| Visualization | `matplotlib` | Anomaly maps, ROC curves, emissivity spectra plots |
| Demo/report | CLI first; optional `streamlit` dashboard (Phase 6) | CLI is testable and CI-friendly; Streamlit is a nice-to-have for live demos |
| Testing | `pytest`, `pytest-cov` | Every module and every edge case gets a test |
| Packaging | `pyproject.toml`, `src/` layout | Standard, importable, installable with `pip install -e .` |

No cloud services, no databases, no auth — this is a local research codebase.

## 6. Repository Structure

```
anomaly-prototype/
├── pyproject.toml
├── README.md
├── RESULTS.md                      # generated, not hand-written
├── configs/
│   ├── salinas.yaml
│   ├── san_diego.yaml
│   ├── abu_urban.yaml
│   ├── thermal_default.yaml
│   └── fusion_default.yaml
├── scripts/
│   ├── download_data.py            # pulls all datasets + spectral libraries, caches under data/
│   └── generate_edge_cases.py      # synthetic corruption of real data for edge-case tests
├── data/                            # gitignored; populated by download_data.py
│   ├── hyperspectral/{salinas,san_diego,abu}/
│   ├── thermal/<dataset_name>/
│   └── speclib/{usgs_splib07,ecostress}/
├── src/pipeline/
│   ├── __init__.py
│   ├── config.py                   # dataclasses + YAML loader
│   ├── errors.py                   # typed exception hierarchy — NFR "no silent failures"
│   ├── results_io.py               # results/<run_id>/ schema, shared by CLIs + dashboard
│   ├── hyperspectral/
│   │   ├── loader.py                # FR-9.1
│   │   ├── preprocess.py            # band masking, PCA — FR-9.1, eq. context
│   │   ├── rx_detector.py           # Eq. 2 — FR-9.2
│   │   ├── autoencoder.py           # model + Eq. 1, Eq. 3 loss — FR-9.4
│   │   ├── score_fusion.py          # FR-9.5
│   │   └── sam.py                   # Eq. 4 — FR-9.6
│   ├── thermal/
│   │   ├── loader.py                # FR-9.7
│   │   ├── preprocess.py            # radiometric calibration, registration — FR-9.7
│   │   ├── cnn_lstm.py               # FR-9.8
│   │   └── planck_emissivity.py      # Eq. 5 — FR-9.9
│   ├── fusion/
│   │   └── fuse.py                  # FR-9.10
│   ├── evaluation/
│   │   ├── metrics.py                # AUC-ROC, precision, recall, F1
│   │   └── report.py                 # writes RESULTS.md
│   └── cli/
│       ├── run_hyperspectral.py
│       ├── run_thermal.py
│       └── run_fusion.py
├── tests/
│   ├── unit/                        # one file per module above
│   ├── integration/                 # end-to-end pipeline runs on real (small) data
│   └── edge_cases/                  # Section 11, one test per row
└── notebooks/
    └── exploration.ipynb            # optional, for visual debugging only — no logic lives here
```

## 7. Data Sources

All sources below are real, currently accessible, and free. `scripts/download_data.py` must implement each of these as a discrete, resumable step, with a checksum/size sanity check, and must **fail loudly with an actionable message** if a source is unreachable (do not silently skip).

### 7.1 Hyperspectral scenes (ground-truth labeled)

| Dataset | Source | Notes |
|---|---|---|
| Salinas | ehu.eus "Hyperspectral Remote Sensing Scenes" page ([ehu.eus/ccwintco](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)); mirrored on Zenodo | 512×217, 224 bands, AVIRIS, ~3.7 m/px. Has a full land-cover ground truth, not an anomaly mask — for anomaly-detection framing, treat rare classes (e.g., isolated structures) as anomalies, or use the Salinas-A subscene as commonly done in HAD papers |
| San Diego Airport | Distributed as part of standard HAD benchmark collections (e.g., the anomaly-detection dataset bundle on ScienceDB: [scidb.cn](https://www.scidb.cn/en/detail?dataSetId=5badf8a68a70450c9fe60b0accb6db27)) | 100×100 crop, 224→186 bands after low-SNR band removal, AVIRIS, has a true binary anomaly ground-truth mask (aircraft on tarmac) — this is the most directly usable "real anomaly" scene |
| ABU series (Airport-Beach-Urban) | [github.com/sxt1996/Airport-Beach-Urban-ABU](https://github.com/sxt1996/Airport-Beach-Urban-ABU) | 13 scenes (4 airport / 4 beach / 5 urban), AVIRIS + one ROSIS scene, each with a binary anomaly ground truth — use for cross-scene generalization testing |

### 7.2 Spectral reference libraries (for SAM, Objective 2)

| Library | Source | Notes |
|---|---|---|
| USGS Spectral Library v7 | [USGS ScienceBase, DOI 10.5066/F7RR1WDJ](https://www.sciencebase.gov/catalog/item/5807a2a2e4b0841e59e3a18d) | Minerals, vegetation, man-made materials, 0.2–200 µm |
| ECOSTRESS Spectral Library v1.0 | [speclib.jpl.nasa.gov](https://speclib.jpl.nasa.gov) | ~3,400 spectra, strong on man-made/urban materials and emissivity — most relevant for the thermal branch's material-matching step too |

Only a curated subset (~20–30 representative spectra covering vegetation, soil, water, concrete, asphalt, and 2–3 metals) needs to be bundled/cached — not the full libraries.

### 7.3 Thermal imagery (labeled, for Objectives 3–4)

| Dataset | Source | Notes |
|---|---|---|
| Infrared Thermal Image Dataset of High-Voltage Electrical Power Equipment | [scidb.cn](https://www.scidb.cn/en/detail?dataSetId=e416c488169f484485ad7575dcfc43ce) | Radiometric-adjacent thermal images of circuit breakers, transformers, surge arresters — hotspot = fault, closest real-world analogue to "thermal anomaly = material of interest" |
| `thermal-images-equip` (induction motors & transformers) | [github.com/mohnaj-nit/thermal-images-equip](https://github.com/mohnaj-nit/thermal-images-equip) | Alternative/supplementary source, same domain |
| FLIR ADAS free dataset | Teledyne FLIR public release | Larger, more diverse thermal scenes; useful for stress-testing the CNN-LSTM's generalization even though it's not a fault-detection dataset — optional, Phase 5+ |

**Known gap, stated honestly:** none of these are *sequences with a clean temporal anomaly-onset label* the way the paper's CNN-LSTM design assumes (a hotspot appearing/growing over time). The prototype should:
1. Use single-frame or short-burst thermal images for the CNN spatial-feature + Planck/emissivity path (this is fully supported by real data).
2. For the LSTM temporal path, either (a) synthesize short sequences by simulating gradual heating on a real background frame (documented as synthetic, not claimed as real), or (b) fall back to a single-frame CNN-only anomaly score when no real sequence data is available, and clearly label results accordingly. **Do not silently present synthetic sequence results as if they were the FLIR/electrical-equipment dataset's real temporal data.**

## 8. Functional Requirements (module by module)

Each FR includes: purpose, inputs/outputs, the paper equation it implements (if any), and a concrete acceptance test.

### FR-9.1 Hyperspectral Loader + Preprocessing
- **Input:** raw scene file (`.mat`), band-wavelength table.
- **Output:** `HSICube` object — `numpy.ndarray` of shape `(H, W, B)`, plus a boolean `good_bands` mask.
- **Requirements:** mask known water-absorption bands (~1350–1450 nm, ~1800–1950 nm) and any band with SNR below a configurable threshold; apply PCA (configurable `n_components`, default retaining 99% variance) for the reduced cube fed to the autoencoder.
- **Acceptance test:** loading Salinas returns a cube with the documented shape and band count; a deliberately corrupted band (all zeros) is correctly excluded by `good_bands`.

### FR-9.2 RX Detector (statistical baseline) — Eq. 2
- Implements `RX(x) = (x-μ)ᵀ Σ⁻¹ (x-μ)` exactly, with a numerically stable covariance inverse (regularize `Σ` with a small ridge term if near-singular — this **is** an edge case, see Section 11).
- **Acceptance test:** on a synthetic scene with known Gaussian background and injected outliers, RX assigns strictly higher scores to outliers than background (statistical property check, not just "runs without error").

### FR-9.3 PCA / Band Selection
- Wraps `sklearn.decomposition.PCA`; must be fit only on background-representative pixels when available (or the full scene when no separate background sample exists), and must be re-usable (fit once, `transform` many times) so training/inference don't refit inconsistently.

### FR-9.4 Spectral-Spatial Autoencoder — Eq. 1 (reconstruction error) + Eq. 3 (loss)
- Architecture: 3D-conv encoder → latent vector → 3D-conv decoder (matches Fig. 1's block diagram). Kernel/filter sizes are a tunable config, not hardcoded.
- Reconstruction error: `S(x) = ||x - x̂||²`, implemented as a standalone function (not just inlined in the training loop) so it's independently testable.
- Loss: `L = (1/N) Σ ||xᵢ - x̂ᵢ||² + λ · SAM(xᵢ, x̂ᵢ)` — the SAM term reuses FR-9.6's function; `λ` is a config value.
- Training: unsupervised, trained on the scene itself (or a background-only crop) per standard HAD practice; must support CPU-only training on a single scene in under ~10 minutes on a laptop (this bounds model size — this is a prototype, not a paper-scale model).
- **Acceptance test:** reconstruction error is measurably higher (statistically, over a held-out anomaly mask) on ground-truth anomaly pixels than background pixels, on at least the San Diego scene.

### FR-9.5 Score Fusion + Thresholding
- Combines normalized RX score and normalized reconstruction error via a configurable weighted sum (`α` config, default 0.5) or max-fusion (config toggle).
- Thresholding: percentile-based (default 99th percentile of fused score) or Otsu, configurable; outputs a binary mask.
- **Acceptance test:** on Salinas/San Diego/ABU, fused-score AUC-ROC ≥ RX-only AUC-ROC minus a small tolerance (i.e., fusion should not silently regress vs. the baseline — if it does, that's a real, reportable finding, not a bug to hide).

### FR-9.6 SAM Material Matching — Eq. 4
- `θ(x, r) = cos⁻¹((x·r) / (‖x‖‖r‖))`, vectorized over the curated USGS/ECOSTRESS subset (Section 7.2).
- Given a detected anomaly's mean spectrum, return the top-3 closest reference materials by angle, plus the angle values (do not just return a single hard label with no confidence indication).
- **Acceptance test:** feeding a reference spectrum from the library back into SAM against itself returns θ ≈ 0; feeding two maximally dissimilar reference spectra (e.g., water vs. metal) returns a large angle.

### FR-9.7 Thermal Loader + Preprocessing
- Reads radiometric thermal images/sequences (via `flirpy` if the source format supports it, else calibrated-DN-to-temperature conversion per the dataset's documented gain/offset) and applies frame registration for any multi-frame sequence.
- **Requirement:** must explicitly detect and label whether a given input is (a) a real radiometric temperature map, or (b) an 8-bit visualization-only thermal JPEG with no true temperature values — the Planck/emissivity module (FR-9.9) must refuse (with a clear error, not a silent wrong answer) to run quantitative emissivity retrieval on (b).

### FR-9.8 CNN Spatial Features + LSTM Temporal Model
- CNN backbone extracts per-frame spatial features (hot-spot boundaries/gradients); LSTM (or GRU as a lighter fallback) models the sequence.
- **Requirement:** must degrade gracefully to CNN-only single-frame scoring when sequence length < a configurable minimum (default 3 frames) — see Section 11 edge cases.
- **Acceptance test:** on the electrical-equipment thermal dataset, per-image anomaly score is higher for images labeled as containing a fault than for normal-operation images (AUC-ROC reported, not asserted as perfect).

### FR-9.9 Planck's Law + Emissivity Retrieval — Eq. 5
- `L(λ,T) = ε(λ)·B(λ,T)`, `B(λ,T) = (2hc²/λ⁵) · 1/(e^(hc/λkT) − 1)`, implemented with real physical constants (`h`, `c`, `k` as named constants with units documented in the docstring, not magic numbers).
- Given a measured radiance/temperature and an assumed or retrieved emissivity, this module must support both directions: (a) retrieve `T` given assumed `ε` (the common case when only one unknown), and (b) flag when both are unknown and a split-window/TES approximation is being used instead of a closed-form solution (be explicit about which mode was used in the output, per FR-9.7's honesty requirement).
- **Acceptance test:** at `ε=1` (blackbody), the module's `B(λ,T)` output matches a reference blackbody calculator (e.g., `astropy.modeling.physical_models.BlackBody` or a hand-computed table) to within numerical tolerance at 2–3 known (λ, T) points.

### FR-9.10 Fusion Module
- Combines hyperspectral anomaly output + thermal anomaly output into one report.
- **Requirement (important, and currently glossed over in the paper):** hyperspectral and thermal inputs will almost never be pixel-co-registered in this prototype (different sensors, different flights/datasets). The fusion module must therefore support two modes:
  1. **Co-registered mode** (if/when the user supplies aligned imagery): pixel-level score fusion as described in the paper.
  2. **Independent mode** (default, given Section 7.3's data reality): report both branches' findings side by side in one document, explicitly labeled as *not spatially fused*, rather than fabricating a fused pixel map from unaligned data.

### FR-9.11 Evaluation + Reporting
- Computes AUC-ROC, precision, recall, F1 against ground truth for every dataset run.
- Writes `RESULTS.md` — a generated Markdown file with one table per dataset, formatted to be a drop-in replacement for the paper's placeholder Table III.
- **Requirement:** every number in `RESULTS.md` must be traceable to a specific script invocation and config file (embed the command + config hash in the report) — no hand-edited numbers.

### FR-9.12 CLI + Optional Dashboard
- Three entry points minimum: `run_hyperspectral`, `run_thermal`, `run_fusion`, each accepting `--dataset` and `--config` and writing outputs to `results/<run_id>/`.
- Streamlit dashboard (Phase 6, optional): loads a `results/<run_id>/` folder and renders the anomaly map, ROC curve, and material-ID table interactively. Not required for Definition of Done.

## 9. Interface Contracts (so modules compose cleanly)

```python
# src/pipeline/hyperspectral/loader.py
@dataclass
class HSICube:
    data: np.ndarray          # (H, W, B), float32, reflectance or radiance
    wavelengths: np.ndarray   # (B,) in nm
    good_bands: np.ndarray    # (B,) bool
    ground_truth: np.ndarray | None  # (H, W) bool, if available

# src/pipeline/hyperspectral/rx_detector.py
def rx_score(cube: np.ndarray, ridge: float = 1e-6) -> np.ndarray: ...  # -> (H, W)

# src/pipeline/hyperspectral/autoencoder.py
def reconstruction_error(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray: ...  # Eq. 1
def composite_loss(x, x_hat, lam: float) -> torch.Tensor: ...                  # Eq. 3

# src/pipeline/hyperspectral/sam.py
def spectral_angle(x: np.ndarray, r: np.ndarray) -> float: ...  # Eq. 4, radians
def match_material(spectrum: np.ndarray, library: dict[str, np.ndarray], top_k=3) -> list[tuple[str, float]]: ...

# src/pipeline/thermal/planck_emissivity.py
def planck_radiance(wavelength_m: np.ndarray, temperature_k: float) -> np.ndarray: ...  # B(λ,T)
def retrieve_temperature(radiance, wavelength_m, emissivity: float) -> float: ...        # invert Eq. 5

# src/pipeline/evaluation/metrics.py
@dataclass
class DetectionMetrics:
    auc_roc: float | None     # None when ground truth holds a single class — see below
    precision: float
    recall: float
    f1: float
    threshold_used: float
```

**`auc_roc` is `float | None`, revised from `float` in v1.0** (PRD v2 change C7). AUC-ROC is genuinely undefined when the ground truth contains only one class, and `None` forces callers to handle that rather than letting a stand-in value flow into a report. This is not a hypothetical: under change C5 the Salinas run is the zero-anomaly edge-case fixture, so its AUC cell is undefined *by design*. The `RESULTS.md` generator renders it as `n/a (single-class GT)` — never blank, never `0.0`, since `0.0` would read as a detector that failed completely.

The implementation also carries `false_positive_rate`, pixel/detection counts, and a `notes` tuple explaining any metric that could not be computed. These are additions, not changes: the five fields above keep their names and meanings.

Every function above needs a unit test before it is wired into a CLI script. This ordering (leaf functions → tests → composition) is the intended build order for Claude Code — see Section 13.

## 10. Non-Functional Requirements

- **Reproducibility:** global seed control (`numpy`, `torch`, `random`) via config; identical config + data → identical metrics (±floating-point tolerance).
- **Runtime budget:** full hyperspectral pipeline (all 3 scenes) completes in under 20 minutes on a CPU-only laptop; thermal pipeline under 10 minutes. GPU is an optional speed-up, never a hard requirement.
- **No silent failures:** any module that cannot produce a valid result (missing data, wrong format, degenerate input) must raise a clear, typed exception — never return zeros/NaNs disguised as a valid score.
- **Traceability to the paper:** every module's docstring cites the paper's equation/figure number it implements.

## 11. Edge Cases (each row = one required automated test)

| # | Category | Edge case | Required behavior |
|---|---|---|---|
| 1 | Hyperspectral | Water-absorption / low-SNR bands present | Automatically masked before PCA/RX/AE; test asserts masked bands are excluded |
| 2 | Hyperspectral | Near-singular background covariance (homogeneous scene) | Ridge-regularized inverse; no `LinAlgError` |
| 3 | Hyperspectral | Scene with zero true anomalies | Pipeline runs; reports near-zero false-positive rate at the chosen threshold rather than crashing on an all-background ground truth |
| 4 | Hyperspectral | Sub-pixel / 1–2 pixel anomaly | Documented expected degradation (report metrics separately for small- vs. large-object anomalies, don't average them away silently) |
| 5 | Hyperspectral | Different band count across scenes (Salinas 224 vs. ABU 189–205) | Loader/PCA pipeline is band-count-agnostic; no hardcoded `224` anywhere in code |
| 6 | Hyperspectral | Saturated / dead pixels (extreme outlier values) | Clipped/flagged in preprocessing before they dominate PCA |
| 7 | SAM | Query spectrum matches no library entry well (all angles large) | Returns top-3 with angles, plus an explicit "low confidence" flag above a configurable angle threshold — never asserts a match it isn't confident about |
| 8 | Thermal | Sequence shorter than LSTM's minimum length | Falls back to CNN-only single-frame scoring, logs that it did so |
| 9 | Thermal | Non-radiometric (visualization-only) thermal image fed to Planck module | Raises a clear error rather than computing a fabricated temperature |
| 10 | Thermal | Frame dropout mid-sequence | Interpolation or explicit gap-handling in the LSTM input, not a silent index-shift bug |
| 11 | Thermal | Unknown/unlisted material (emissivity not in reference table) | Falls back to a documented default emissivity range with the uncertainty stated in the output, not a false-precision point estimate |
| 12 | Fusion | Hyperspectral and thermal inputs not co-registered (the default real-world case here) | Runs in "independent mode" (FR-9.10) rather than fabricating pixel alignment |
| 13 | Fusion | Only one modality available | Fusion module degrades to single-branch report, clearly labeled |
| 14 | System | Missing/corrupted downloaded data file | `download_data.py` re-download with checksum retry; pipeline scripts fail fast with an actionable message, not a stack trace from deep inside `scipy.io.loadmat` |
| 15 | System | Re-running with the same config | Bit-for-bit (or metric-for-metric within tolerance) identical results — reproducibility check |

## 12. Testing Strategy

- **Unit tests** (`tests/unit/`): one file per leaf function in Section 9 — RX, reconstruction error, composite loss, SAM, Planck/emissivity. These use small synthetic arrays with known analytic answers (e.g., hand-computed RX score for a 3-pixel toy scene).
- **Integration tests** (`tests/integration/`): run each CLI entry point against a *small real* dataset (San Diego, being the smallest, is the primary integration-test dataset) and assert the output files exist, are well-formed, and metrics fall in a sane range (not exact-value assertions, since retraining introduces variance — assert `auc_roc > 0.5` at minimum, tightening as the model is tuned).
- **Edge-case tests** (`tests/edge_cases/`): one test per row in Section 11, using `scripts/generate_edge_cases.py` to programmatically corrupt real data (rather than hand-crafting fragile fixtures).
- **Coverage target:** ≥80% on `src/pipeline/`, enforced via `pytest-cov` in CI if CI is set up; not a hard gate for a research prototype, but tracked.

## 13. Milestones & Suggested Build Order (for Claude Code)

Each milestone is independently demoable and independently a legitimate "we made real progress" checkpoint — do not wait until the end to have something runnable.

| # | Milestone | Deliverable | Acceptance |
|---|---|---|---|
| M0 | Scaffold | Repo structure, `pyproject.toml`, config system, `download_data.py` for all Section 7 sources | `pip install -e .` works; `python scripts/download_data.py` populates `data/` with checksums verified |
| M1 | Hyperspectral statistical baseline | RX detector + evaluation harness, run on Salinas + San Diego + ABU | Real AUC-ROC numbers reported in `RESULTS.md` for all 3 datasets |
| M2 | Hyperspectral deep pipeline | Autoencoder (Eq. 1, 3) + score fusion (FR-9.5), compared against M1 | AUC-ROC delta vs. RX baseline reported and explained (better or worse — report the truth) |
| M3 | Material ID | SAM module (Eq. 4) against curated USGS/ECOSTRESS subset | Top-3 material match report for at least 5 detected anomalies across the 3 scenes |
| M4 | Thermal pipeline | Loader/preprocess + CNN(+LSTM) on the electrical-equipment thermal dataset | AUC-ROC/F1 on fault-vs-normal classification reported |
| M5 | Emissivity/material (thermal) | Planck's law module (Eq. 5) wired to thermal anomaly outputs | Blackbody sanity test passes (Section 8, FR-9.9); at least one worked material-ID example on real thermal data |
| M6 | Fusion + reporting | `run_fusion` CLI, `RESULTS.md` generator | One end-to-end fused report produced from real hyperspectral + real thermal runs (independent-mode fusion per FR-9.10) |
| M7 | Edge-case hardening | All 15 Section 11 tests passing | `pytest tests/` green, including `tests/edge_cases/` |
| M8 | Polish | README with reproduction instructions, optional Streamlit dashboard | A third party (or the professor) can clone, run `download_data.py`, run the 3 CLI commands, and get the same headline numbers |

**Recommended order of implementation within each milestone:** leaf/pure functions first (with unit tests) → wire into a module → wire into a CLI script → integration test → update `RESULTS.md` generator. This lets Claude Code work in small, independently-verifiable steps rather than one large untestable pass.

## 14. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| No real dataset perfectly matches the paper's "thermal sequence with temporal anomaly onset" framing | M4/M5 results look weaker than the paper's narrative implies | State this honestly in `RESULTS.md` and the paper's revised results section; report single-frame CNN results as the primary real-data result, LSTM-on-synthetic-sequences as a clearly labeled secondary demonstration |
| Autoencoder underperforms RX on small AVIRIS scenes (common in HAD literature when training data is scene-limited) | M2 shows no improvement over M1 | This is a legitimate research finding, not a bug — report it; the paper's Section VI already anticipates comparing against a baseline |
| USGS/ECOSTRESS full libraries are large and slow to parse | Slows down `download_data.py` | Only fetch/parse the curated ~20–30 spectrum subset (Section 7.2), documented explicitly with which entries were chosen and why |
| Hyperspectral vs. thermal co-registration is genuinely unavailable | Fusion module can't do real pixel-level fusion | FR-9.10's independent mode is the honest default — do not fabricate alignment |
| CPU-only training is slow for the 3D-conv autoencoder | Runtime budget (Section 10) blown | Keep the model small (prototype-scale, not paper-scale); make architecture depth/width a config value so it can be scaled down further if needed |

## 15. Open Questions (flagging for a decision before/while Claude Code builds)

1. **Compute environment:** confirm whether implementation will run on the team's own laptops (CPU-only assumption holds) or if any GPU access (e.g., Colab) is available — affects how ambitious M2's autoencoder architecture can be.
2. **Thermal dataset choice:** confirm which of the three Section 7.3 sources to standardize on as primary (electrical-equipment thermal dataset is recommended as the default).
3. **Demo requirement:** confirm whether a live Streamlit dashboard is actually needed for the showcase, or whether static `RESULTS.md` + saved figures are sufficient — this changes whether M8's dashboard work is in scope.
4. **Paper update:** once `RESULTS.md` exists, Table III in the conference paper should be updated with real numbers — flag this as a follow-up task, not part of this codebase.

## 16. Appendix A — Equation-to-Code Traceability

| Paper equation | Description | Module | Function |
|---|---|---|---|
| Eq. 1 | Reconstruction error `S(x) = ‖x − x̂‖²` | `hyperspectral/autoencoder.py` | `reconstruction_error` |
| Eq. 2 | RX detector `RX(x) = (x−μ)ᵀΣ⁻¹(x−μ)` | `hyperspectral/rx_detector.py` | `rx_score` |
| Eq. 3 | Composite training loss | `hyperspectral/autoencoder.py` | `composite_loss` |
| Eq. 4 | SAM `θ(x,r) = cos⁻¹(x·r / ‖x‖‖r‖)` | `hyperspectral/sam.py` | `spectral_angle` |
| Eq. 5 | Planck's law `L(λ,T) = ε(λ)·B(λ,T)` | `thermal/planck_emissivity.py` | `planck_radiance`, `retrieve_temperature` |

## 17. Appendix B — Dataset & Library Quick Reference

- Salinas / Pavia: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
- San Diego / Pavia anomaly-detection bundle: https://www.scidb.cn/en/detail?dataSetId=5badf8a68a70450c9fe60b0accb6db27
- ABU series: https://github.com/sxt1996/Airport-Beach-Urban-ABU
- USGS Spectral Library v7: https://www.sciencebase.gov/catalog/item/5807a2a2e4b0841e59e3a18d
- ECOSTRESS Spectral Library: https://speclib.jpl.nasa.gov
- Thermal (electrical equipment): https://www.scidb.cn/en/detail?dataSetId=e416c488169f484485ad7575dcfc43ce
- Thermal (motors/transformers, alt.): https://github.com/mohnaj-nit/thermal-images-equip
- Reference real-time RX implementation (for algorithm cross-checking, not copying): https://github.com/WiseGamgee/HyperAD

---

*End of PRD. Next step: hand Sections 4–13 to Claude Code as the implementation brief, starting at Milestone M0.*
