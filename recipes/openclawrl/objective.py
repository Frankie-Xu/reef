"""OpenClaw-RL training objective."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.core.trajectories import trajectory_reward
from reef.train.algos import TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.registry import register_objective
from reef.train.algos.signals import StepScheduling, StepSignal
from reef.train.types import TrainingBatch, trajectories


@register_objective
class OpenClawRLObjective(TrainingObjective):
    name = "openclawrl"
    loss_family = "openclawrl"

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        samples = trajectories(batch)
        # The upstream top-K loss consumes raw rewards without normalization.
        advantages = tuple(trajectory_reward(sample) for sample in samples)
        steps = next_steps(state)
        return StepSignal(
            "train",
            {"steps": steps},
            {"advantages": advantages, "steps": steps},
            advantages,
            StepScheduling(unit="sample"),
        )
