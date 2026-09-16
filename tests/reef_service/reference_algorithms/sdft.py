"""Pure-Python reference for the SDFT objective (arXiv:2601.19897).

Transcribes ``_compute_loss`` of the reference implementation
(idanshen/Self-Distillation at ``d77573212fa0``, ``distil_trainer.py``) with
plain floats: at every response position, the KL over the full vocabulary
between the teacher's next-token distribution and the student's; the
per-sample mean over the trained tokens; and the truncated importance-sampling
weight. ``tests/reef_service/test_sdft_parity.py`` pins the tensor code in
``recipes/sdft/slime/objective.py`` to it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def log_softmax(logits: Sequence[float]) -> list[float]:
    largest = max(logits)
    log_sum_exp = largest + math.log(sum(math.exp(value - largest) for value in logits))
    return [value - log_sum_exp for value in logits]


def token_kl(student_logits: Sequence[float], teacher_log_probs: Sequence[float], direction: str) -> float:
    """KL(teacher || student) for ``forward``, KL(student || teacher) for ``reverse``, at one position."""
    student_log_probs = log_softmax(student_logits)
    if direction == "forward":
        return sum(
            math.exp(teacher) * (teacher - student)
            for teacher, student in zip(teacher_log_probs, student_log_probs, strict=True)
        )
    if direction == "reverse":
        return sum(
            math.exp(student) * (student - teacher)
            for teacher, student in zip(teacher_log_probs, student_log_probs, strict=True)
        )
    raise ValueError(f"unknown direction {direction!r}")


def sequence_importance_weight(
    student_log_probs: Sequence[float],
    rollout_log_probs: Sequence[float],
    loss_mask: Sequence[int],
    cap: float,
) -> float:
    """The masked mean of ``min(pi_theta / pi_rollout, cap)`` over the response."""
    ratios = [
        min(math.exp(student - rollout), cap)
        for student, rollout in zip(student_log_probs, rollout_log_probs, strict=True)
    ]
    trained = sum(loss_mask)
    return sum(ratio * mask for ratio, mask in zip(ratios, loss_mask, strict=True)) / max(trained, 1)


def sample_loss(per_token_kl: Sequence[float], loss_mask: Sequence[int], weight: float = 1.0) -> float:
    """The reference's per-sample loss: the masked mean token KL scaled by the sample's weight."""
    trained = sum(loss_mask)
    return weight * sum(kl * mask for kl, mask in zip(per_token_kl, loss_mask, strict=True)) / max(trained, 1)
