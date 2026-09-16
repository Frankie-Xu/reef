"""SDFT payload conversion for the Slime bridge: the policy 5-tuple plus the teacher sequence.

The teacher sequence is the sample's ``teacher_tokens`` (the teacher prompt
ids followed by the student's response ids verbatim). Alignment is exact by
construction and checked here: a sequence whose tail is not the student's
response would make the teacher pass score the wrong positions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

from reef.core.trajectories import source_record_id, trajectory_reward
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.slime_backend.data_builder import build_policy_rollout_data
from reef.train.types import TrajectoryItem

_ROW_SHAPE = "[source_id, tokens, loss_mask, rollout_log_probs, reward, teacher_tokens]"


def sdft_sample_row(sample: TrajectoryItem) -> list[Any]:
    """Shape one Reef sample into SDFT's 6-element wire row."""
    return [
        source_record_id(sample),
        list(sample.training.get("tokens", [])),
        list(sample.training.get("loss_mask", [])),
        list(sample.training.get("rollout_log_probs", [])),
        trajectory_reward(sample),
        list(sample.training.get("teacher_tokens", [])),
    ]


def build_sdft_rollout_data(payload: Mapping[str, Any], samples: Sequence, spec: SlimeAlgorithm) -> dict:
    """Validate and convert Reef SDFT rows into Slime's external rollout payload."""
    base_rows: list[list[Any]] = []
    teacher_rows: list[Any] = []
    for index, row in enumerate(samples):
        if not isinstance(row, Sequence) or isinstance(row, str | bytes) or len(row) != 6:
            raise ValueError(f"SDFT sample {index} must be {_ROW_SHAPE}")
        base_rows.append(list(row[:5]))
        teacher_rows.append(row[5])

    data = build_policy_rollout_data({**dict(payload), "samples": base_rows}, base_rows, spec)
    teacher_tokens: list[list[int]] = []
    for index, (row_teacher, tokens, response_length) in enumerate(
        zip(teacher_rows, data["tokens"], data["response_lengths"], strict=True)
    ):
        if (
            not isinstance(row_teacher, Sequence)
            or isinstance(row_teacher, str | bytes)
            or any(not isinstance(value, Integral) or isinstance(value, bool) for value in row_teacher)
        ):
            raise ValueError(f"SDFT sample {index} teacher_tokens must be a sequence of integers")
        ids = [int(value) for value in row_teacher]
        if len(ids) <= response_length:
            raise ValueError(
                f"SDFT sample {index} teacher sequence must carry a prompt before its {response_length}-token response"
            )
        if ids[-response_length:] != tokens[-response_length:]:
            raise ValueError(f"SDFT sample {index} teacher sequence must end with the student's response ids")
        teacher_tokens.append(ids)
    data["teacher_tokens"] = teacher_tokens
    return data
