"""RESULTS.md formatting (PRD v2 change C7)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.evaluation.metrics import compute_detection_metrics
from pipeline.evaluation.report import (
    AUC_UNDEFINED,
    format_auc,
    format_score,
    metrics_table,
)


def test_defined_auc_renders_as_a_number() -> None:
    assert format_auc(0.94123456) == "0.9412"


def test_undefined_auc_is_never_blank_or_zero() -> None:
    """C7 -- the distinction must survive the report boundary."""
    rendered = format_auc(None)
    assert rendered == AUC_UNDEFINED
    assert rendered.strip() != ""
    assert "0.0" not in rendered
    assert "None" not in rendered


def test_undefined_auc_says_why() -> None:
    """A paper table is read out of context, so the cell explains itself."""
    assert "single-class" in format_auc(None).lower()


def test_zero_auc_is_not_confused_with_undefined() -> None:
    """A genuinely zero AUC is a real result and must render as a number."""
    assert format_auc(0.0) == "0.0000"
    assert format_auc(0.0) != format_auc(None)


def test_format_score_handles_none_without_printing_none() -> None:
    assert format_score(None) == "n/a"
    assert format_score(0.5) == "0.5000"


def test_salinas_style_zero_anomaly_run_renders_cleanly() -> None:
    """The end-to-end C5 case: no anomalies present, so AUC is undefined."""
    scores = np.linspace(0, 1, 200)
    gt = np.zeros(200, dtype=bool)
    m = compute_detection_metrics(scores, gt, threshold_percentile=99.0)

    table = metrics_table({"rx": m})

    assert AUC_UNDEFINED in table
    assert m.auc_roc is None
    # The false-positive rate is the meaningful number in this run, so it must
    # still appear as a real value.
    assert f"{m.false_positive_rate:.4f}" in table


def test_table_has_one_row_per_method() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    gt = np.array([1, 0, 1, 0, 0], dtype=bool)
    m = compute_detection_metrics(scores, gt, threshold=0.65)

    table = metrics_table({"rx": m, "autoencoder": m, "fused": m})
    body = [ln for ln in table.splitlines() if ln.startswith("| ")]

    assert len(body) == 4  # header + 3 methods
    assert "| rx |" in table
    assert "| autoencoder |" in table


def test_table_is_valid_markdown() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    gt = np.array([1, 0, 1, 0, 0], dtype=bool)
    table = metrics_table({"rx": compute_detection_metrics(scores, gt, threshold=0.65)})
    lines = table.splitlines()

    widths = {ln.count("|") for ln in lines}
    assert len(widths) == 1, "every row must have the same column count"
    assert set(lines[1]) <= set("|-")


def test_empty_metrics_is_explicit() -> None:
    assert "No metrics" in metrics_table({})
