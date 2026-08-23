"""The typed exception hierarchy (PRD v2 change C6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.errors import (
    ConfigError,
    DataUnavailableError,
    DegenerateInputError,
    IncompatibleInputError,
    MissingGroundTruthError,
    ModelNotFittedError,
    NonRadiometricInputError,
    PipelineError,
    UnregisteredInputError,
)

ALL_ERRORS = [
    ConfigError,
    DataUnavailableError,
    DegenerateInputError,
    IncompatibleInputError,
    MissingGroundTruthError,
    ModelNotFittedError,
    NonRadiometricInputError,
    UnregisteredInputError,
]


@pytest.mark.parametrize("cls", ALL_ERRORS)
def test_every_error_is_a_pipeline_error(cls: type) -> None:
    """Catching PipelineError must catch every deliberate failure."""
    assert issubclass(cls, PipelineError)


def test_pipeline_error_does_not_swallow_real_bugs() -> None:
    """A KeyError is a bug, not an expected failure, and must stay visible."""
    assert not issubclass(KeyError, PipelineError)
    assert not issubclass(IndexError, PipelineError)
    assert issubclass(PipelineError, Exception)


# --------------------------------------------------------------------------
# C6 -- DataUnavailableError carries its remedy in the type
# --------------------------------------------------------------------------


def test_data_unavailable_requires_path_and_remedy() -> None:
    """§11 edge case 14 is enforced by the signature, not by prose discipline."""
    with pytest.raises(TypeError):
        DataUnavailableError("something went wrong")  # type: ignore[call-arg]


def test_data_unavailable_exposes_structured_fields() -> None:
    err = DataUnavailableError(
        reason="Checksum mismatch",
        path="data/hyperspectral/abu/abu-urban-1.mat",
        remedy="Re-run scripts/download_data.py --only abu",
    )
    assert err.reason == "Checksum mismatch"
    assert err.path == Path("data/hyperspectral/abu/abu-urban-1.mat")
    assert "download_data" in err.remedy


def test_data_unavailable_message_names_path_and_fix() -> None:
    err = DataUnavailableError(
        reason="File is missing",
        path="data/speclib/ecostress",
        remedy="Fetch it from https://speclib.jpl.nasa.gov/download",
    )
    message = str(err)
    assert "File is missing" in message
    assert "data" in message and "ecostress" in message
    assert "speclib.jpl.nasa.gov" in message


# --------------------------------------------------------------------------
# C6 -- pairing failures are their own category
# --------------------------------------------------------------------------


def test_incompatible_is_distinct_from_degenerate() -> None:
    """Each input is fine; only the combination is not.

    Catching DegenerateInputError must not accidentally catch a wavelength-grid
    mismatch, because the two call for different fixes: one means the data is
    unusable, the other means it needs resampling first.
    """
    assert not issubclass(IncompatibleInputError, DegenerateInputError)
    assert not issubclass(DegenerateInputError, IncompatibleInputError)


def test_missing_ground_truth_is_separately_catchable() -> None:
    """An unlabelled scene should still produce an anomaly map, just no metrics.

    That recovery is only expressible if the error is narrower than
    DegenerateInputError -- catching the latter to allow it would swallow
    genuinely unusable input too.
    """
    assert not issubclass(MissingGroundTruthError, DegenerateInputError)

    try:
        raise MissingGroundTruthError("Scene carries no labels; metrics skipped.")
    except MissingGroundTruthError:
        recovered = True
    assert recovered
