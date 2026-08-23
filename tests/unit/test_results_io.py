"""The results/<run_id>/ contract (Roadmap 0.6, FR-9.11)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from pipeline.config import HyperspectralConfig, config_hash, load_config
from pipeline.errors import DataUnavailableError
from pipeline.evaluation.metrics import compute_detection_metrics
from pipeline.results_io import (
    CONFIG_FILE,
    COMMAND_FILE,
    HASH_FILE,
    METRICS_FILE,
    SCHEMA_VERSION,
    RunRecord,
    list_runs,
    make_record,
    new_run_id,
    read_run,
    write_run,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture()
def cfg() -> HyperspectralConfig:
    return load_config(CONFIG_DIR / "had100.yaml", HyperspectralConfig)


@pytest.fixture()
def metrics() -> dict:
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    gt = np.array([1, 0, 1, 0, 0], dtype=bool)
    return {"rx": compute_detection_metrics(scores, gt, threshold=0.65)}


def test_run_id_is_sortable_and_prefixed() -> None:
    run_id = new_run_id("had100")
    assert run_id.startswith("had100_")
    assert len(run_id.split("_")[-1]) == len("20260820-143012")


def test_run_id_sanitises_unsafe_characters() -> None:
    assert "/" not in new_run_id("abu/urban 1")


def test_write_creates_all_four_contract_files(tmp_path: Path, cfg, metrics) -> None:
    record = make_record("had100", cfg, metrics=metrics)
    run_dir = write_run(tmp_path, record, cfg)

    for name in (METRICS_FILE, CONFIG_FILE, HASH_FILE, COMMAND_FILE):
        assert (run_dir / name).is_file(), f"{name} missing from the run directory"


def test_round_trip_preserves_metrics(tmp_path: Path, cfg, metrics) -> None:
    record = make_record("had100", cfg, metrics=metrics)
    run_dir = write_run(tmp_path, record, cfg)

    restored = read_run(run_dir)
    assert restored.run_id == record.run_id
    assert restored.config_hash == record.config_hash
    assert restored.metrics["rx"] == metrics["rx"]


def test_written_hash_matches_the_config(tmp_path: Path, cfg, metrics) -> None:
    """FR-9.11 -- the recorded hash must actually describe the recorded config."""
    record = make_record("had100", cfg, metrics=metrics)
    run_dir = write_run(tmp_path, record, cfg)

    on_disk_hash = (run_dir / HASH_FILE).read_text(encoding="utf-8").strip()
    reloaded_cfg = yaml.safe_load((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))

    assert on_disk_hash == config_hash(cfg)
    assert on_disk_hash == config_hash(reloaded_cfg)


def test_saved_config_includes_unstated_defaults(tmp_path: Path, cfg, metrics) -> None:
    """config_used.yaml is the *resolved* config, so a run is reproducible from it."""
    record = make_record("had100", cfg, metrics=metrics)
    run_dir = write_run(tmp_path, record, cfg)
    saved = yaml.safe_load((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))

    # kernel_spatial is never mentioned in had100.yaml but must be recorded.
    assert saved["autoencoder"]["kernel_spatial"] == 3


def test_provenance_labels_survive_the_round_trip(tmp_path: Path, cfg) -> None:
    """PRD §7.3 -- a synthetic-sequence caveat must reach the report mechanically.

    Tagging the run is what stops synthetic LSTM results being written up as if
    they came from the real dataset; the label has to survive serialization for
    that to work.
    """
    record = make_record(
        "thermal",
        cfg,
        labels={"sequence_source": "synthetic", "radiometric": "false"},
        notes=["LSTM trained on simulated heating over a real background frame."],
    )
    run_dir = write_run(tmp_path, record, cfg)
    restored = read_run(run_dir)

    assert restored.labels["sequence_source"] == "synthetic"
    assert restored.labels["radiometric"] == "false"
    assert restored.notes == record.notes


def test_metrics_json_is_human_readable(tmp_path: Path, cfg, metrics) -> None:
    record = make_record("had100", cfg, metrics=metrics)
    run_dir = write_run(tmp_path, record, cfg)
    payload = json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8"))

    assert payload["schema_version"] == SCHEMA_VERSION
    # Positives are 0.9 and 0.7; the negative 0.8 outranks one of them, so 5 of
    # the 6 (positive, negative) pairs are concordant.
    assert payload["metrics"]["rx"]["auc_roc"] == pytest.approx(5 / 6)
    assert "command" in payload and "created_at" in payload


def test_several_metric_sets_in_one_run(tmp_path: Path, cfg) -> None:
    """M2 reports an autoencoder-vs-baseline delta from a single invocation."""
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    gt = np.array([1, 0, 1, 0, 0], dtype=bool)
    record = make_record(
        "had100",
        cfg,
        metrics={
            "rx": compute_detection_metrics(scores, gt, threshold=0.65),
            "autoencoder": compute_detection_metrics(scores[::-1], gt, threshold=0.65),
            "fused": compute_detection_metrics(scores, gt, threshold=0.5),
        },
    )
    restored = read_run(write_run(tmp_path, record, cfg))
    assert set(restored.metrics) == {"rx", "autoencoder", "fused"}


# --------------------------------------------------------------------------
# §11 edge case 14 -- actionable errors, never a raw stack trace
# --------------------------------------------------------------------------


def test_missing_run_directory_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(DataUnavailableError, match="not found"):
        read_run(tmp_path / "nope")


def test_incomplete_run_directory_is_actionable(tmp_path: Path) -> None:
    (tmp_path / "half_run").mkdir()
    with pytest.raises(DataUnavailableError, match="not a complete run"):
        read_run(tmp_path / "half_run")


def test_corrupt_metrics_json_is_actionable(tmp_path: Path) -> None:
    run_dir = tmp_path / "broken"
    run_dir.mkdir()
    (run_dir / METRICS_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(DataUnavailableError, match="not valid JSON"):
        read_run(run_dir)


def test_schema_version_mismatch_is_actionable(tmp_path: Path, cfg, metrics) -> None:
    record = make_record("had100", cfg, metrics=metrics)
    run_dir = write_run(tmp_path, record, cfg)

    payload = json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8"))
    payload["schema_version"] = SCHEMA_VERSION + 1
    (run_dir / METRICS_FILE).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataUnavailableError, match="schema version"):
        read_run(run_dir)


# --------------------------------------------------------------------------
# Listing -- what the dashboard enumerates
# --------------------------------------------------------------------------


def test_list_runs_is_newest_first(tmp_path: Path, cfg, metrics) -> None:
    for run_id in ("had100_20260101-000000", "had100_20260601-000000"):
        record = RunRecord(run_id=run_id, command="x", config_hash="h", metrics=metrics)
        write_run(tmp_path, record, cfg)

    names = [p.name for p in list_runs(tmp_path)]
    assert names == ["had100_20260601-000000", "had100_20260101-000000"]


def test_list_runs_skips_incomplete_directories(tmp_path: Path, cfg, metrics) -> None:
    write_run(tmp_path, make_record("good", cfg, metrics=metrics), cfg)
    (tmp_path / "crashed_run").mkdir()

    assert [p.name for p in list_runs(tmp_path)] == [
        p.name for p in list_runs(tmp_path) if p.name.startswith("good")
    ]


def test_list_runs_on_missing_root_is_empty(tmp_path: Path) -> None:
    assert list_runs(tmp_path / "no_results_yet") == []
