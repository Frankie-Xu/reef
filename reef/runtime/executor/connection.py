"""Attach to Reef coordinators through executor RPC or named Ray actors.

Only read-only health/version operations retry after actor replacement.
Borrowed handles never shut down the remote coordinator.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from reef.core.batches import TrainingBatch
from reef.runtime.executor import Executor
from reef.runtime.executor.failure import ExecutorFailedError
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.interfaces import PreparedTrainingStep, TrainingJobResult, TrainingRuntimeError

DEFAULT_ACTOR_NAME = "reef-train-bridge"
DEFAULT_NAMESPACE = "reef"


class CoordinatorClient(ABC):
    """Client contract for Reef coordinator RPCs, shared by both runtimes.

    The transport is shared by the separate training and inference runtimes.
    It is not a runtime or a recipe-facing interface. Backends keep their
    actor groups and payload formats private.
    """

    reconnects = False

    def serving_runtime_load_id(self) -> str | None:
        """Return the serving version, or ``None`` when unavailable."""
        return None

    @abstractmethod
    def health(self) -> Mapping[str, Any]:
        """Return backend health and durable training-job state."""
        ...

    @abstractmethod
    def update_serving_weights(self, training_job_id: str) -> TrainingJobResult:
        """Activate one checkpointed candidate in the serving engine."""
        ...

    @abstractmethod
    def reject_training_candidate(self, training_job_id: str) -> None:
        """Finish a rejected candidate without changing serving weights."""
        ...

    @abstractmethod
    def acknowledge_training_commit(self, training_job_id: str) -> None:
        """Acknowledge Reef's durable commit for an activated candidate."""
        ...

    @abstractmethod
    def prepare_training_step(
        self, batch: TrainingBatch, step_preparer: str, algorithm_state: Mapping[str, Any]
    ) -> PreparedTrainingStep: ...

    @abstractmethod
    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult: ...

    def shutdown(self) -> None:
        return


class ExecutorCoordinatorClient(CoordinatorClient):
    """Drive a training coordinator through an executor's control RPC.

    The coordinator owns any backend-specific distributed training group.
    Each operation targets one rank: broadcasting a training job to all
    workers would run the coordinator's side effects more than once.
    """

    def __init__(self, executor: Executor, *, rank: int = 0, timeout_s: float = 300.0) -> None:
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise TrainingRuntimeError("training timeout must be a positive finite number")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise TrainingRuntimeError("training timeout must be a positive finite number")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise TrainingRuntimeError("training coordinator rank must be a non-negative integer")
        self._executor = executor
        self._rank = rank
        self._timeout_s = timeout_s

    @property
    def executor(self) -> Executor:
        return self._executor

    def _rpc(self, method: str, *args: Any) -> Any:
        return self._executor.rpc(self._rank, method, args=args, timeout=self._timeout_s)

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
    ) -> PreparedTrainingStep:
        return self._rpc("prepare_training_step", batch, step_preparer, dict(algorithm_state))

    def serving_runtime_load_id(self) -> str | None:
        version = self._rpc("serving_runtime_load_id")
        return None if version is None else str(version)

    def health(self) -> Mapping[str, Any]:
        try:
            value = self._rpc("health")
        except AttributeError as exc:
            raise TrainingRuntimeError(
                "training backend predates deferred serving-weight updates; "
                "restart the Reef service and training actor together"
            ) from exc
        if not isinstance(value, Mapping):
            raise TrainingRuntimeError("train group returned invalid health")
        if not isinstance(value.get("training_job"), Mapping):
            raise TrainingRuntimeError(
                "training backend predates deferred serving-weight updates; "
                "restart the Reef service and training actor together"
            )
        return dict(value)

    def update_serving_weights(self, training_job_id: str) -> TrainingJobResult:
        return self._rpc("update_serving_weights", training_job_id)

    def reject_training_candidate(self, training_job_id: str) -> None:
        self._rpc("reject_training_candidate", training_job_id)

    def acknowledge_training_commit(self, training_job_id: str) -> None:
        self._rpc("acknowledge_training_commit", training_job_id)

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        return self._rpc("execute_training_job", dict(payload))

    def shutdown(self) -> None:
        """Release the executor; its ownership policy protects attached workers."""
        self._executor.shutdown()


def training_job_status(handle: CoordinatorClient) -> Mapping[str, Any]:
    health = handle.health()
    if not isinstance(health, Mapping):
        raise TrainingRuntimeError(f"train group handle returned invalid health: {type(health).__name__}")
    healthy = health.get("ok")
    if healthy is not None and not isinstance(healthy, bool):
        raise TrainingRuntimeError("train group returned malformed health status")
    # A group that reports its failure as recoverable is retried through
    # the normal reconciliation path instead of being declared dead.
    if healthy is False and health.get("recoverable") is not True:
        phase = health.get("phase")
        detail = f" in phase {phase!r}" if isinstance(phase, str) and phase else ""
        raise TrainingRuntimeError(f"train group is unhealthy{detail}")
    status = health.get("training_job")
    if not isinstance(status, Mapping):
        raise TrainingRuntimeError("train group health is missing training_job status")
    deferred = status.get("deferred_weight_update")
    colocate = health.get("colocate", False)
    lora_adapter = health.get("lora_adapter")
    state = status.get("status", "COMPLETE")
    if deferred is not True:
        raise TrainingRuntimeError("Reef requires deferred serving-weight updates")
    if not isinstance(colocate, bool) or not isinstance(state, str):
        raise TrainingRuntimeError("train group returned malformed training-job status")
    if lora_adapter is not None and (not isinstance(lora_adapter, str) or not lora_adapter):
        raise TrainingRuntimeError("train group returned a malformed serving adapter name")
    lora_mode = health.get("lora_mode")
    if lora_mode is not None and lora_mode not in {"shared", "scenario"}:
        raise TrainingRuntimeError(f"train group returned unknown LoRA mode: {lora_mode!r}")
    lora_adapters = health.get("lora_adapters")
    if lora_adapters is not None and not isinstance(lora_adapters, Mapping):
        raise TrainingRuntimeError("train group returned malformed per-scenario adapters")
    adapter_residency = health.get("adapter_residency")
    if adapter_residency is not None and not isinstance(adapter_residency, Mapping):
        raise TrainingRuntimeError("train group returned malformed adapter residency")
    if "commit_acknowledged" in status and not isinstance(status["commit_acknowledged"], bool):
        raise TrainingRuntimeError("train group returned malformed commit acknowledgement")
    if state not in {
        "IDLE",
        "RUNNING",
        "CHECKPOINT",
        "UPDATING_WEIGHTS",
        "READY_TO_COMMIT",
        "HEAD_COMMITTED",
        "COMPLETE",
        "REJECTED",
        "REJECTING",
    }:
        raise TrainingRuntimeError(f"train group returned unknown training-job status: {state!r}")
    return {
        "inference_url": health.get("inference_url"),
        **dict(status),
        "serving_healthy": healthy is not False and health.get("phase") != "recovering",
        "colocate": colocate,
        "lora_adapter": lora_adapter,
        "lora_mode": lora_mode,
        "lora_adapters": dict(lora_adapters) if lora_adapters else {},
        "adapter_residency": dict(adapter_residency) if adapter_residency is not None else None,
    }


RayRuntimeError = TrainingRuntimeError
RayCoordinatorClient = CoordinatorClient


def _require_ray():
    """Import Ray lazily so other executors need not install it."""
    try:
        import ray
    except ImportError as exc:
        raise RayRuntimeError(
            "remote Ray runtimes require the 'ray' package; install it to connect to a training backend"
        ) from exc
    return ray


class RemoteRayCoordinatorClient(ExecutorCoordinatorClient):
    """Attach a non-owning executor to an existing Ray training coordinator."""

    def __init__(self, train_group_actor: Any, *, timeout_s: float = 300.0) -> None:
        super().__init__(
            RayExecutor.from_workers((train_group_actor,), owned=False),
            timeout_s=timeout_s,
        )


class NamedRayCoordinatorClient(RemoteRayCoordinatorClient):
    """Discover the current coordinator before each RPC, without replaying writes.

    A missing name is retried within the operation's timeout. Once an RPC is
    submitted, a write's error reaches the caller even if a replacement appears.
    Only liveness/version reads retry actor loss. Durable training reconciliation
    decides whether a training operation is replayable.
    """

    reconnects = True

    def __init__(
        self, actor_name: str, namespace: str, *, timeout_s: float = 300, health_timeout_s: float = 300
    ) -> None:
        if (
            isinstance(health_timeout_s, bool)
            or not isinstance(health_timeout_s, (int, float))
            or not math.isfinite(health_timeout_s)
            or health_timeout_s <= 0
        ):
            raise TrainingRuntimeError("health timeout must be a positive finite number")
        super().__init__(_require_ray().get_actor(actor_name, namespace=namespace), timeout_s=timeout_s)
        self._actor_name = actor_name
        self._namespace = namespace
        self._closed = False
        self._health_timeout_s = min(health_timeout_s, timeout_s)

    def _rpc(self, method: str, *args: Any) -> Any:
        ray = _require_ray()
        timeout = self._health_timeout_s if method in {"health", "serving_runtime_load_id"} else self._timeout_s
        deadline = time.monotonic() + timeout
        while not self._closed:
            try:
                actor = ray.get_actor(self._actor_name, namespace=self._namespace)
            except ValueError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TrainingRuntimeError("training coordinator is unavailable during recovery") from exc
                time.sleep(min(0.1, remaining))
                continue
            executor = RayExecutor.from_workers((actor,), owned=False)
            try:
                return executor.rpc(0, method, args=args, timeout=max(0.001, deadline - time.monotonic()))
            except (ray.exceptions.RayActorError, ExecutorFailedError):
                if method not in {"health", "serving_runtime_load_id"} or time.monotonic() >= deadline:
                    raise
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
            finally:
                executor.shutdown()
        raise TrainingRuntimeError("training coordinator connection is closed")

    def shutdown(self) -> None:
        self._closed = True
        super().shutdown()


def connect_ray_coordinator(
    *,
    actor_name: str = DEFAULT_ACTOR_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    ray_address: str | None = None,
    inference_timeout_s: float = 300.0,
    train_timeout_s: float | None = None,
) -> NamedRayCoordinatorClient:
    """Connect to the current named coordinator, borrowing its workers."""
    ray = _require_ray()
    if not ray.is_initialized():
        ray.init(address=ray_address or "auto", namespace=namespace)
    return NamedRayCoordinatorClient(
        actor_name,
        namespace,
        timeout_s=train_timeout_s if train_timeout_s is not None else inference_timeout_s,
        health_timeout_s=inference_timeout_s,
    )
