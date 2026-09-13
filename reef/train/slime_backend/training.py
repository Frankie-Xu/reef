"""Slime training workers, initialized independently of the inference service."""

from __future__ import annotations

import logging
from typing import Any

from reef.runtime.deployment import DeploymentResources, WeightTransferSession
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.training_job.operations import TrainingOperations
from reef.train.slime_backend.reef_adapters.bridge import (
    BridgePreparation,
    create_train_groups,
    create_training_operations,
)
from reef.train.slime_backend.resources import SlimeDeploymentResources


class SlimeTrainingService:
    """Own actor/critic workers and a sender attachment, never the coordinator."""

    weight_transfer_protocol = "slime-sglang-control-v2"

    def __init__(
        self,
        args: Any,
        *,
        preparation: BridgePreparation,
        loss_family_config: object | None,
    ) -> None:
        self.args = args
        self.preparation = preparation
        self.loss_family_config = loss_family_config
        self._actor_group: Any = None
        self._critic_group: Any = None
        self._operations: TrainingOperations | None = None
        self._session_id: str | None = None
        self._started = False
        self._closed = False

    def start(self, resources: DeploymentResources) -> None:
        if self._started or self._closed:
            raise RuntimeError("training service can only be started once")
        if not isinstance(resources, SlimeDeploymentResources) or not resources.placement_groups:
            raise ValueError("Slime training requires its supplied model reservations")
        self._started = True
        self._actor_group, self._critic_group = create_train_groups(
            self.args, resources.placement_groups, rollout_manager=None
        )
        # Ongoing observation belongs to operations in the coordinator
        # process. Watching these driver-side handles would report intentional
        # release_train retirement as a deployment failure after serialization.

    def attach_weight_transport(self, session: WeightTransferSession) -> None:
        """Attach the existing native sender protocol after workers are initialized."""
        if self._actor_group is None or self._closed:
            raise RuntimeError("start training workers before attaching their weight transport")
        if session.protocol != self.weight_transfer_protocol:
            raise ValueError("Slime training received an incompatible weight transfer protocol")
        if self._session_id is not None:
            if self._session_id == session.session_id:
                return
            raise RuntimeError("training workers cannot reuse a different deployment's weight transfer session")
        receiver = session.receiver
        if not isinstance(receiver, RayExecutor) or len(receiver.workers) != 1:
            raise ValueError("Slime weight transport requires one native Ray receiver")
        # The pinned native updaters consume one borrowed actor handle. Reef
        # already fenced/offloaded the receiver; attachment only configures
        # sender layout and transport, never its inference lifecycle.
        for group in (self._actor_group, self._critic_group):
            if group is not None:
                group.set_rollout_manager(receiver.workers[0])
        self._session_id = session.session_id

    def operations(self) -> TrainingOperations:
        if self._actor_group is None or self._session_id is None or self._closed:
            raise RuntimeError("training operations require initialized workers and attached weight transport")
        if self._operations is None:
            self._operations = create_training_operations(
                self.args,
                self._actor_group,
                self._critic_group,
                preparation=self.preparation,
                loss_family_config=self.loss_family_config,
            )
        return self._operations

    def check_health(self) -> None:
        self.poll()
        for group in (self._actor_group, self._critic_group):
            if group is not None:
                group.check_health()

    def poll(self) -> None:
        if self._actor_group is None or self._closed:
            raise RuntimeError("training service is not running")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors = []
        for group in (self._critic_group, self._actor_group):
            if group is not None:
                try:
                    group.release()
                except Exception as exc:
                    errors.append(exc)
                    logging.getLogger(__name__).exception("Failed to release Slime training workers")
        if errors:
            raise errors[0]
