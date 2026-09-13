"""Backend-neutral component contracts for model deployment ownership.

These contracts describe startup, health observation and cleanup. Training steps, weight
transport and commit-gated activation keep their existing runtime contracts.
Concrete integrations keep framework arguments and allocation handles private.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from reef.runtime.executor import Executor, ExecutorConfig, ExecutorFuture, WorkerSpec
from reef.runtime.executor.failure import ExecutorFailedError
from reef.runtime.training_job.coordinator import InferenceOperations, TrainingCoordinator, TrainingOperations


class DeploymentResources(Protocol):
    """A coordinated allocation and its runtime connection, owned by Reef."""

    def start(self) -> None:
        """Acquire resources; close must also handle a partially failed start."""

    def close(self) -> None:
        """Release owned reservations and connections, idempotently."""


class DeploymentHealth(Protocol):
    """Nonblocking observation of an already started deployment."""

    def poll(self) -> None:
        """Raise on failure; an outstanding probe is not a failed component."""


class ComponentHealth:
    """Observe selected components without importing their implementations."""

    def __init__(self, *components: DeploymentHealth) -> None:
        self.components = components

    def poll(self) -> None:
        for component in self.components:
            component.poll()


class ModelPlanSource(Protocol):
    """Rebuild components and rerun durable recovery preflight for each attempt."""

    def create(self) -> ModelDeploymentPlan:
        """Return a fresh, unallocated plan from the original configuration."""


@dataclass(frozen=True)
class InferenceConnection:
    """Borrowed control transport with an explicitly versioned adapter protocol.

    The protocol identifies the RPC vocabulary, including direct weight-update
    attachment. An HTTP endpoint alone does not satisfy this connection.
    Reusing it for replacement training workers requires the owning deployment
    to retire the prior trainer and perform the protocol's attachment handshake.
    """

    protocol: str
    control: Executor


@dataclass(frozen=True)
class WeightTransferSession:
    """One deployment's borrowed native sender/receiver attachment.

    This carries the existing adapter control protocol, not a universal tensor
    format. Reef creates a new session identity whenever it rebuilds a receiver;
    training workers must not reuse a previous deployment's attachment.
    """

    protocol: str
    receiver: Executor
    session_id: str

    def __post_init__(self) -> None:
        if not self.protocol or not self.session_id:
            raise ValueError("weight transfer sessions require a protocol and identity")


class InferenceService(Protocol):
    """An inference component that owns its engines but borrows reservations."""

    @property
    def connection_protocol(self) -> str:
        """Control protocol provided by the selected engine integration."""

    def start(self, resources: DeploymentResources) -> InferenceConnection:
        """Start engines in supplied resources and return a borrowed connection."""

    def prepare_weight_transfer(self, connection: InferenceConnection) -> None:
        """Fence the receiver and release shared resources before trainer startup."""

    def operations(self, connection: InferenceConnection) -> InferenceOperations:
        """Return this receiver's operations for Reef-owned coordination."""

    def check_health(self) -> None:
        """Raise when the component is not ready."""

    def poll(self) -> None:
        """Observe component failures without blocking behind active work."""

    def close(self) -> None:
        """Release owned engines, including partial starts, idempotently."""


class TrainingService(Protocol):
    """Training workers allocated independently of the inference component."""

    @property
    def weight_transfer_protocol(self) -> str | None:
        """Native weight transport consumed by a separately attached sender."""

    def start(self, resources: DeploymentResources) -> None:
        """Start training workers using only their supplied reservations."""

    def attach_weight_transport(self, session: WeightTransferSession) -> None:
        """Configure a sender after allocation without taking receiver ownership."""

    def operations(self) -> TrainingOperations:
        """Return training-only operations for Reef-owned coordination."""

    def check_health(self) -> None:
        """Raise when the component is not ready."""

    def poll(self) -> None:
        """Observe component failures without blocking behind active work."""

    def close(self) -> None:
        """Release owned training objects, including partial starts, idempotently."""


@dataclass(frozen=True)
class CoordinatorConfig:
    """Executor selection for Reef's coordinator, independent of backend code."""

    backend: str | type[Executor] = "ray"
    options: Mapping[str, Any] = field(default_factory=dict)
    launch_timeout_s: float | None = None


@dataclass(frozen=True)
class ModelDeploymentPlan:
    """Configured components; constructing a plan must not allocate resources.

    A missing inference component explicitly selects a combined compatibility
    lifecycle. It is never a fallback after a separate component fails to start.
    """

    resources: DeploymentResources
    inference: InferenceService | None
    training: TrainingService
    health: DeploymentHealth | None = None
    coordinator: CoordinatorConfig | None = None

    def validate(self) -> None:
        if self.coordinator is not None and self.inference is None:
            raise ValueError("a Reef coordinator requires separate training and inference components")
        required = self.training.weight_transfer_protocol
        provided = self.inference.connection_protocol if self.inference is not None else None
        if required != provided or (self.inference is not None and not provided):
            raise ValueError(
                f"incompatible inference control protocol: training requires {required!r}, got {provided!r}"
            )


_logger = logging.getLogger(__name__)


class ModelDeployment:
    """Own allocation, backend attachment and the generic coordinator process."""

    def __init__(self, plan: ModelDeploymentPlan) -> None:
        self.plan = plan
        self._started = False
        self._closed = False
        self._resources_started = False
        self._inference_started = False
        self._training_started = False
        self.weight_transfer_session: WeightTransferSession | None = None
        self._coordinator: Executor | None = None
        self._coordinator_probe: ExecutorFuture | None = None

    def start(self) -> None:
        if self._started or self._closed:
            raise RuntimeError("model deployment can only be started once")
        self.plan.validate()
        self._started = True
        try:
            self._resources_started = True
            self.plan.resources.start()
            connection = None
            if self.plan.inference is not None:
                self._inference_started = True
                connection = self.plan.inference.start(self.plan.resources)
                if connection.protocol != self.plan.training.weight_transfer_protocol:
                    raise ValueError("inference returned a connection with an incompatible protocol")
                self.plan.inference.check_health()
                # Colocated trainers cannot initialize until inference has
                # acknowledged releasing its initial device allocations.
                self.plan.inference.prepare_weight_transfer(connection)
                self.weight_transfer_session = WeightTransferSession(
                    connection.protocol, connection.control, uuid4().hex
                )
            self._training_started = True
            self.plan.training.start(self.plan.resources)
            if self.weight_transfer_session is not None:
                self.plan.training.attach_weight_transport(self.weight_transfer_session)
            self.plan.training.check_health()
            if connection is not None:
                self._start_coordinator(connection)
            if self.plan.inference is not None:
                self.plan.inference.check_health()
        except BaseException:
            try:
                self.close()
            except Exception:
                _logger.exception("Failed to clean up model deployment after startup failure")
            raise

    def _start_coordinator(self, connection: InferenceConnection) -> None:
        config = self.plan.coordinator
        inference = self.plan.inference
        if config is None or inference is None:
            return
        self._coordinator = Executor.create(
            ExecutorConfig(
                backend=config.backend,
                workers=(
                    WorkerSpec(
                        TrainingCoordinator,
                        args=(self.plan.training.operations(), inference.operations(connection)),
                        kwargs={"owns_training": True},
                    ),
                ),
                options=config.options,
                launch_timeout_s=config.launch_timeout_s,
            )
        )
        # Constructor recovery can take as long as checkpoint loading. The
        # launcher's ready timeout bounds startup; a short RPC timeout must
        # not interrupt a healthy recovery and leave its result ambiguous.
        self._validate_health(self._coordinator.rpc(0, "health"))

    def poll(self) -> None:
        """Observe components without queuing repeated health work behind training."""
        if self.plan.health is not None:
            self.plan.health.poll()
        coordinator = self._coordinator
        if coordinator is None:
            return
        if coordinator.failure is not None:
            raise ExecutorFailedError(coordinator.failure)
        if self._coordinator_probe is None:
            self._coordinator_probe = coordinator.rpc(0, "health", non_block=True)
        try:
            result = self._coordinator_probe.result(timeout=0)
        except TimeoutError:
            return
        self._coordinator_probe = None
        self._validate_health(result, allow_recovery=True)

    @staticmethod
    def _validate_health(result: Any, *, allow_recovery: bool = False) -> None:
        if isinstance(result, Mapping) and (
            result.get("ok") is True or (allow_recovery and result.get("recoverable") is True)
        ):
            return
        raise RuntimeError(f"training coordinator failed its health check: {result!r}")

    def _close_coordinator(self) -> None:
        coordinator = self._coordinator
        if coordinator is not None:
            try:
                coordinator.rpc(0, "shutdown", timeout=90)
            finally:
                coordinator.shutdown()
                self._coordinator = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        try:
            self._close_coordinator()
        except Exception as exc:
            errors.append(exc)
            _logger.exception("Failed to close runtime coordinator")
        for started, component in (
            (self._training_started, self.plan.training),
            (self._inference_started, self.plan.inference),
            (self._resources_started, self.plan.resources),
        ):
            if started and component is not None:
                try:
                    component.close()
                except Exception as exc:
                    errors.append(exc)
                    _logger.exception("Failed to close deployment component %s", type(component).__name__)
        if errors:
            raise errors[0]
