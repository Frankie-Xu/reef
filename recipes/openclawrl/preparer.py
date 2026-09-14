"""OpenClaw-RL step preparer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.core.trajectories import trajectory_reward
from reef.train.algos.base import StepPreparer, register_step_preparer
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepScheduling, StepSignal
from reef.train.types import TrainingBatch, trajectories


@register_step_preparer
class OpenClawRLPreparer(StepPreparer):
    name = "openclawrl"

    def __call__(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        samples = trajectories(batch)
        advantages = tuple(trajectory_reward(sample) for sample in samples)
        steps = next_steps(state)
        return StepSignal(
            "train",
            # The paper objective: the verbatim upstream top-K select loss
            # (recipes/openclawrl/slime); advantages stay the raw
            # per-sample rewards (upstream disables reward normalization).
            self.name,
            {"steps": steps},
            {"advantages": advantages, "steps": steps},
            advantages,
            StepScheduling(unit="sample"),
        )
