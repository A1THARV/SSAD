"""The ``results/<run_id>/`` contract shared by every CLI, report and the dashboard.

One schema, written by all three CLI entry points (FR-9.12) and read by the
report generator (FR-9.11) and the Streamlit dashboard (M8). Fixing it in Phase
0 is what keeps the dashboard from holding any analysis logic of its own -- it
only renders what a run already recorded, so a broken dashboard can never
invalidate a result.

Layout::

    results/<run_id>/
        metrics.json        run metadata + one entry per named metric set
        config_used.yaml    the fully resolved config, defaults included
        config_hash.txt     short hash of that resolved config
        command.txt         the exact command line that produced the run
        anomaly_map.png     branch-specific figures
        roc_curve.png

FR-9.11 requires every number in ``RESULTS.md`` to be traceable to a specific
invocation and config. ``command.txt`` plus ``config_hash.txt`` are what make
that check mechanical rather than a matter of trust.
"""

from __future__ import annotations

import json
import shlex
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .config import config_hash, to_dict
from .errors import DataUnavailableError
from .evaluation.metrics import DetectionMetrics

__all__ = [
    "SCHEMA_VERSION",
    "RunRecord",
    "new_run_id",
    "current_command",
    "write_run",
    "read_run",
    "list_runs",
]

SCHEMA_VERSION = 1

METRICS_FILE = "metrics.json"
CONFIG_FILE = "config_used.yaml"
HASH_FILE = "config_hash.txt"
COMMAND_FILE = "command.txt"


@dataclass
class RunRecord:
    """Everything one pipeline run recorded about itself.

    Attributes:
        run_id: Directory name under ``output_root``.
        created_at: UTC ISO-8601 timestamp.
        command: The command line that produced this run (FR-9.11).
        config_hash: Short hash of the resolved config (FR-9.11).
        metrics: Named metric sets, e.g. ``{"rx": ..., "fused": ...}``. Keeping
            several under one run is what lets M2 report an autoencoder-versus-
            baseline delta from a single invocation.
        labels: Provenance flags that must travel with the numbers. This is
            where PRD §7.3's honesty requirement is enforced mechanically:
            ``sequence_source: synthetic`` or ``radiometric: "false"`` set here
            propagates into ``metrics.json`` and then into ``RESULTS.md``,
            rather than depending on someone remembering the caveat at
            write-up time.
        notes: Free-text caveats attached to the run as a whole.
    """

    run_id: str
    command: str
    config_hash: str
    created_at: str = ""
    schema_version: int = SCHEMA_VERSION
    metrics: dict[str, DetectionMetrics] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "command": self.command,
            "config_hash": self.config_hash,
            "labels": dict(self.labels),
            "notes": list(self.notes),
            "metrics": {name: m.to_dict() for name, m in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        return cls(
            run_id=data["run_id"],
            command=data["command"],
            config_hash=data["config_hash"],
            created_at=data.get("created_at", ""),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            metrics={
                name: DetectionMetrics.from_dict(payload)
                for name, payload in data.get("metrics", {}).items()
            },
            labels=dict(data.get("labels", {})),
            notes=list(data.get("notes", [])),
        )


def new_run_id(prefix: str, when: datetime | None = None) -> str:
    """Build a sortable run id like ``salinas_20260820-143012``."""
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in prefix)
    return f"{safe}_{stamp}"


def current_command() -> str:
    """Reconstruct the command line that launched this process (FR-9.11)."""
    return shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]]) if sys.argv else ""


def write_run(
    output_root: str | Path,
    record: RunRecord,
    config: Any,
) -> Path:
    """Create ``<output_root>/<run_id>/`` and write the four contract files.

    Args:
        output_root: Usually ``results/``.
        record: The run's metadata and metrics.
        config: The resolved config dataclass (or a plain mapping).

    Returns:
        The created run directory, so callers can save figures alongside.
    """
    run_dir = Path(output_root) / record.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = to_dict(config) if hasattr(config, "__dataclass_fields__") else dict(config)

    (run_dir / METRICS_FILE).write_text(
        json.dumps(record.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (run_dir / CONFIG_FILE).write_text(
        yaml.safe_dump(payload, sort_keys=True, default_flow_style=False), encoding="utf-8"
    )
    (run_dir / HASH_FILE).write_text(record.config_hash + "\n", encoding="utf-8")
    (run_dir / COMMAND_FILE).write_text(record.command + "\n", encoding="utf-8")
    return run_dir


def read_run(run_dir: str | Path) -> RunRecord:
    """Read a run directory back into a :class:`RunRecord`.

    Raises:
        DataUnavailableError: the directory or its ``metrics.json`` is missing
            or malformed -- an actionable message rather than a ``JSONDecodeError``
            from somewhere inside the dashboard (§11 edge case 14).
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / METRICS_FILE

    if not run_dir.is_dir():
        raise DataUnavailableError(
            reason="Run directory not found",
            path=run_dir,
            remedy="Run one of the pipeline CLIs first, or check the run id spelling.",
        )
    if not metrics_path.is_file():
        raise DataUnavailableError(
            reason=f"Directory exists but has no {METRICS_FILE}, so it is not a complete run",
            path=metrics_path,
            remedy="A run probably crashed before writing its results; re-run it.",
        )
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataUnavailableError(
            reason=f"File is not valid JSON -- {exc}",
            path=metrics_path,
            remedy="Delete the run directory and re-run the pipeline to regenerate it.",
        ) from exc

    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise DataUnavailableError(
            reason=(
                f"File uses results schema version {version}, but this build expects "
                f"version {SCHEMA_VERSION}"
            ),
            path=metrics_path,
            remedy="Re-run the pipeline to regenerate the run under the current schema.",
        )
    return RunRecord.from_dict(data)


def list_runs(output_root: str | Path) -> list[Path]:
    """Every complete run directory under ``output_root``, newest first."""
    root = Path(output_root)
    if not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and (p / METRICS_FILE).is_file()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def make_record(
    run_id_prefix: str,
    config: Any,
    metrics: dict[str, DetectionMetrics] | None = None,
    labels: dict[str, str] | None = None,
    notes: list[str] | None = None,
) -> RunRecord:
    """Convenience constructor wiring together the run id, command and hash."""
    return RunRecord(
        run_id=new_run_id(run_id_prefix),
        command=current_command(),
        config_hash=config_hash(config),
        metrics=metrics or {},
        labels=labels or {},
        notes=notes or [],
    )
