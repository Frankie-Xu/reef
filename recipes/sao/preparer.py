"""SAO step preparer: one rollout, one DP unit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepScheduling, StepSignal
from reef.train.types import TrainingBatch, trajectories


@register_step_preparer
class SaoPreparer(StepPreparer):
    name = "sao"

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        samples = trajectories(batch)
        steps = next_steps(state)
        return StepSignal(
            "train",
            self.name,
            {"steps": steps},
            {"steps": steps, "rollouts": len(samples)},
            scheduling=StepScheduling(unit="sample"),
        )
