"""Optional candidate backend capabilities used by service routes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from reef.harness.tree.mutations import Mutation
from reef.train.cordis_backend.proposals import ProposalInbox


class StepRecords(ABC):
    """Read a scenario's retained training step files."""

    @abstractmethod
    def read_step_records(self, directory: str, relative: str | None) -> dict[str, Any]: ...


class ProposalGate(ABC):
    """Admit proposed mutations and expose their scenario inbox."""

    @property
    @abstractmethod
    def proposals(self) -> ProposalInbox | None: ...

    @abstractmethod
    def admit(
        self, entries: Sequence[Mapping[str, Any]], mutations: Sequence[Mutation]
    ) -> tuple[list[dict[str, Any]], str | None]: ...
