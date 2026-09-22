# Spectral–Spatial Anomaly Detection in Hyperspectral and Thermal Remote Sensing

Research prototype backing a conference paper by Atharv Donadkar et al. (JD
College of Engineering & Management, Nagpur). It implements the paper's five
equations against public benchmarks and produces the measured results that
replace its Section VII.

**Status:** all five milestones have measured results. 494 unit tests passing.

---

## Results

**[`RESULTS.md`](RESULTS.md) is the substance of this repository.** Headline
figures, in brief:

| | result |
|---|---|
| **HAD100 detection** (Eq. 1 + 2 + 3 fused, α = 0.5) | **0.9843 mAUC** — 10th of 20 against the benchmark's published baselines |
| **Eq. 2 verification** | reproduces HAD100's own `GRX` baseline to **4 decimal places** (0.9799 / 0.9714) |
| **Eq. 5 validation** | agrees with `astropy` BlackBody to **1e-9**, Wien to 1e-3, Stefan–Boltzmann to 1e-8 |
| **Thermal detection (M4)** | image-level fire presence is **saturated** — a one-line threshold gives AUC 1.0000 with a 61 °C margin |
| **Temperature retrieval (M5)** | the emissivity assumption moves the answer by **307 K** across material classes, 4.4 K within |
| **Material identification (M3)** | works at resolved/majority abundance (0.904 / 0.576), **not a capability sub-pixel** (0.068 at a = 0.10) |

Two results generalise beyond this paper:

- **Which bands are diagnostic is task-dependent, and non-diagnostic bands
  actively degrade performance.** Measured in both directions on the same data:
  detection peaks in the visible and is hurt by SWIR; identification peaks in
  SWIR and is hurt by the visible. Below a target abundance of ~0.20, dropping
  150 of 276 bands nearly doubles identification accuracy.
- **A recurring failure mode**: transformations that remove the very thing being
  measured, producing a plausible number with no error raised. Four independent
  instances across two modalities — see `RESULTS.md` §12.

---

## Layout

```
src/pipeline/
  hyperspectral/    RX (Eq. 2), autoencoder (Eq. 1, 3), SAM (Eq. 4), PCA, fusion
  thermal/          radiometric loader, calibration, Planck (Eq. 5)
  evaluation/       metrics, thresholding, reporting
  cli/              end-to-end runs writing a fixed results-folder schema
scripts/            data acquisition, ablations, experiments
tests/unit/         494 tests
configs/            one YAML per dataset/protocol
```

`ROADMAP.md` is the living plan: locked decisions, phase status, and a running
log of findings that overturned earlier assumptions.

---

## Running it

```bash
pip install -e ".[test]"
python scripts/download_data.py --verify     # what is present, what is missing
python -m pytest -q
```

Datasets are **not** in this repository. `scripts/download_data.py` fetches what
it can and, for account-gated sources, prints the page, the exact path to drop
the file at, and what milestone it blocks. A hand-placed file passes verification
on the next run and the pipeline behaves identically.

```bash
python -m pipeline.cli.run_hyperspectral --config configs/had100_band50.yaml
python scripts/thermal_retrieval.py
python scripts/sam_mixing_experiment.py
```

### Required datasets

| source | size | needed for |
|---|---|---|
| HAD100 (AVIRIS-NG) | 3.75 GB | M1, M2 — primary benchmark |
| HAD100 baselines | 0.1 GB | M2 comparison |
| ABU, Salinas | small | cross-check, zero-anomaly fixture |
| FLAME 3 CV subset | 7.5 GB | M4, M5 — thermal branch |
| ECOSTRESS spectral library | 0.13 GB | M3, M5 — SAM and the ε prior |

---

## Conventions

Three rules the code is built around, each because breaking one produced a wrong
result that looked right:

1. **Never tune on the test set that is reported against.** Where an operating
   point was chosen post hoc it says so, beside the curve it was chosen from.
2. **Refuse rather than fabricate.** A non-radiometric image cannot reach
   Planck's law; a library spectrum is never extrapolated onto a sensor grid; a
   single-class ground truth reports `n/a` rather than a stand-in AUC.
3. **Verify against something external.** Self-consistency cannot catch a wrong
   physical constant or a units error — Eq. 2 is checked against a published
   implementation, Eq. 5 against `astropy`.

An environment note: Anaconda's MKL and PyTorch ship different OpenMP runtimes,
which aborts the interpreter on every line of the PCA path. `pipeline/runtime.py`
carries the workaround, and `tests/unit/test_runtime.py` verifies it is
bit-identical rather than trusting it.

---

## Citation

Benchmarks and libraries are consumed, never re-implemented:

- Li et al., *Hyperspectral Anomaly Detection: A Survey and Benchmark*, IEEE TGRS
  61, 2023. doi:10.1109/TGRS.2023.3258067
- Hopkins et al., *FLAME 3*, arXiv:2412.02831. doi:10.21227/w0mz-aq48
- Meerdink et al., *The ECOSTRESS Spectral Library v1.0*, RSE 230:111196, 2019
