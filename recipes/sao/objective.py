"""SAO training objective: one rollout, one DP unit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import StepScheduling, StepSignal
from reef.train.types import TrainingBatch, trajectories


@register_objective
class SaoObjective(TrainingObjective):
    name = "sao"
    loss_family = "sao"

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        samples = trajectories(batch)
        steps = next_steps(state)
        return StepSignal(
            "train",
            {"steps": steps},
            {"steps": steps, "rollouts": len(samples)},
            scheduling=StepScheduling(unit="sample"),
        )
