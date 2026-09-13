"""Backend-facing operations and configuration for Reef training coordination.

Concrete integrations implement training or inference operations independently.
The contracts carry scheduling values and acknowledged backend operations;
publication ordering and recovery policy remain in the coordinator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

from reef.runtime.base import PreparedTrainingStep, TrainingJobResult
from reef.runtime.training_job.execution import PreparedTrainingJob
from reef.runtime.training_job.scenarios import ScenarioHistory
from reef.runtime.weights.version import RuntimeLoadId, new_runtime_load_id_incarnation


@dataclass(frozen=True)
class TrainingCoordinationConfig:
    """Deployment policy interpreted only by Reef's coordinator."""

    save_hf_template: str | None
    colocate: bool = False
    lora: bool = False
    adapter_capacity: int | None = None
    keep_lora_base_resident: bool = False


def _initial_runtime_load_id() -> str:
    return str(RuntimeLoadId(new_runtime_load_id_incarnation(), 0))


@dataclass
class TrainingContext:
    """Scheduling state available to backend preparation without control handles."""

    next_rollout_id: int = 0
    runtime_load_id: str = field(default_factory=_initial_runtime_load_id)
    history: ScenarioHistory | None = None


class TrainingOperations(Protocol):
    """Training-only preparation, checkpoint I/O and native weight sending.

    Native training backends implement this interface for Reef's coordinator.
    The recipe-facing candidate lifecycle is ``reef.train.backend.TrainingBackend``.
    Sender methods must never pause, resume, offload or restart inference.
    Reef supplies the exact identity for each transfer. The sender must echo
    that identity; Reef independently verifies every receiver before commit.
    """

    config: TrainingCoordinationConfig
    context: TrainingContext

    def start(self) -> None: ...

    def check_health(self) -> None: ...

    def prepare_training_step(
        self, batch: Any, step_preparer: str, algorithm_state: Mapping[str, Any]
    ) -> PreparedTrainingStep: ...

    def prepare(
        self, payload: Mapping[str, Any], *, job_id: str, rollout_id: int, prior_marker: Mapping[str, Any] | None
    ) -> AbstractContextManager[PreparedTrainingJob | TrainingJobResult]: ...

    def prepare_weights(self, runtime_load_id: str, *, force_full: bool) -> None:
        """Prepare a sender while colocated inference resources are released."""

    def send_weights(self, runtime_load_id: str, *, force_full: bool) -> str: ...

    def initialize_version(self, runtime_load_id: str) -> None: ...

    def activate_scenario(self, scenario: str) -> None: ...

    def send_adapter(self, scenario: str, name: str) -> None: ...

    def close(self) -> None: ...


class InferenceOperations(Protocol):
    """Receiver operations with acknowledged completion and no commit policy."""

    def initialize_version(self, runtime_load_id: str) -> None: ...

    def inference_url(self) -> str: ...

    def runtime_load_ids(self) -> Sequence[str]: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def recover(self) -> None: ...

    def abort(self) -> None: ...

    def offload(self, tags: tuple[str, ...] | None) -> None: ...

    def onload_weights(self) -> None: ...

    def onload_kv(self) -> None: ...

    def unload_adapter(self, name: str) -> None: ...
