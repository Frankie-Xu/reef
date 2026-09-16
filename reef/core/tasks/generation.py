"""Inputs and validation results for record-driven task generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reef.core.records_types import AgentRecord


@dataclass(frozen=True)
class TaskGenerationRequest:
    """One task's source records, requirements, and optional local assets.

    Assets are files or directories accessible to the generator, such as a
    repository snapshot or verifier fixtures. Construction does not read them.
    Method-specific settings belong to the processor's configuration.
    """

    source_records: tuple[AgentRecord, ...]
    description: str
    assets: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_records, tuple) or not self.source_records:
            raise ValueError("source_records must be a non-empty tuple of AgentRecord values")
        if any(not isinstance(record, AgentRecord) for record in self.source_records):
            raise TypeError("source_records must contain AgentRecord values")
        if len({record.scenario for record in self.source_records}) != 1:
            raise ValueError("source_records must belong to one scenario")
        record_ids = [record.agent_record_id for record in self.source_records]
        if any(not record_id for record_id in record_ids) or len(set(record_ids)) != len(record_ids):
            raise ValueError("source_records must have distinct non-empty record ids")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be non-empty text")
        if not isinstance(self.assets, tuple) or any(not isinstance(path, Path) for path in self.assets):
            raise TypeError("assets must be a tuple of pathlib.Path values")


@dataclass(frozen=True)
class TaskValidationResult:
    """A completed validation: no errors means the task passed all required checks.

    Errors describe task defects. Failures to execute the checks (for example,
    an unavailable container runtime) raise exceptions instead of rejecting the
    task as invalid.
    """

    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.errors, tuple):
            raise TypeError("errors must be a tuple of non-empty strings")
        if any(not isinstance(error, str) or not error.strip() for error in self.errors):
            raise ValueError("errors must contain non-empty strings")

    @property
    def is_valid(self) -> bool:
        """Whether every required check passed."""
        return not self.errors
