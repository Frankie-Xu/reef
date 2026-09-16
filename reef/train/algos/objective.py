"""Backend-neutral algorithm objectives, before optimizer or worker partitioning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.train.algos.signals import StepSignal
from reef.train.types import TrainingBatch


class TrainingObjective(ABC):
    """Own a method's batch preparation and choice of backend loss implementation.

    Declare ``name`` and ``loss_family`` and implement ``prepare``. Preparation
    receives the complete reserved batch, so group-relative statistics do not
    depend on optimizer or data-parallel partitioning. Model-dependent terms
    remain in the backend loss implementation. Return proposed algorithm state;
    the trainer owns its commit, including retries and recovery.

    Keep this module and method implementations independent of torch, Slime,
    and Tinker. A recipe's loss family must be inspectable before workers start.
    """

    name: str = ""
    loss_family: str = ""

    @abstractmethod
    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        """Prepare a full batch without mutating it or committing algorithm state."""
