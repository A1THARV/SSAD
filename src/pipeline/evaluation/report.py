"""Rendering metrics into ``RESULTS.md`` (FR-9.11).

The full generator is Phase 6 work. What lives here now is the formatting
layer, written early and deliberately: PRD v2 change C7 requires an undefined
AUC-ROC to reach the page as ``n/a (single-class GT)`` -- not blank, and not
``0.0``. Left as a note it would be lost; as a tested function it cannot be.

That distinction is not cosmetic. Salinas is now the §11 edge case 3 fixture
(change C5) -- a scene with no anomalies, whose whole point is that AUC-ROC is
undefined there. A ``0.0`` in that cell would read as a detector that failed
completely, which is the opposite of what the run shows.
"""

from __future__ import annotations

from ..evaluation.metrics import DetectionMetrics

__all__ = [
    "AUC_UNDEFINED",
    "format_auc",
    "format_score",
    "metrics_table",
]

#: What an undefined AUC-ROC renders as. Names *why* it is undefined, so the
#: cell is self-explanatory in a paper table read out of context.
AUC_UNDEFINED = "n/a (single-class GT)"

_COLUMNS = ("AUC-ROC", "Precision", "Recall", "F1", "Threshold", "FPR", "Anomaly px")


def format_auc(value: float | None, places: int = 4) -> str:
    """Render an AUC-ROC, including the undefined case.

    Args:
        value: The metric, or ``None`` when the ground truth held one class.
        places: Decimal places.

    Returns:
        A fixed-point string, or :data:`AUC_UNDEFINED`.
    """
    if value is None:
        return AUC_UNDEFINED
    return f"{value:.{places}f}"


def format_score(value: float | None, places: int = 4) -> str:
    """Render precision/recall/F1-style values.

    ``None`` is rendered as a plain ``n/a``: unlike AUC-ROC these are reported
    as ``0.0`` with an explanatory note rather than as ``None``, so this branch
    exists only to keep an unexpected ``None`` from printing as the string
    ``"None"`` in a table.
    """
    if value is None:
        return "n/a"
    return f"{value:.{places}f}"


def metrics_table(metrics: dict[str, DetectionMetrics]) -> str:
    """Render named metric sets as one GitHub-flavoured Markdown table.

    Args:
        metrics: Named results, e.g. ``{"rx": ..., "fused": ...}``. Order is
            preserved, so callers control the row order.

    Returns:
        The table, without a trailing newline. Empty input yields an explicit
        placeholder line rather than a header with no rows.
    """
    if not metrics:
        return "_No metrics recorded for this run._"

    header = "| Method | " + " | ".join(_COLUMNS) + " |"
    rule = "|---" * (len(_COLUMNS) + 1) + "|"
    lines = [header, rule]

    for name, m in metrics.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    format_auc(m.auc_roc),
                    format_score(m.precision),
                    format_score(m.recall),
                    format_score(m.f1),
                    f"{m.threshold_used:.4g}",
                    format_score(m.false_positive_rate),
                    f"{m.n_anomaly_pixels:,}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)
