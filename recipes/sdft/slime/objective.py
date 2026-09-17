"""Tensor implementation of the SDFT objective (arXiv:2601.19897).

Loaded lazily by the package wrapper in ``recipes.sdft.slime`` so importing
the package never requires torch. Slime's Megatron workers resolve the hooks
by path: ``sdft_actor_pre_train`` scores every sample's teacher sequence with
the current policy before the step, and ``sdft_loss`` computes the per-token
KL between those teacher distributions and the student's.

The loss follows ``distil_trainer.py`` of the reference implementation
(idanshen/Self-Distillation at ``d77573212fa0``): at every response position
of the student's on-policy sample, the KL over the full vocabulary between the
teacher's next-token distribution (prompt with the demonstration) and the
student's (plain prompt); forward KL by default, reverse KL as a switch; the
per-sample mean over the trained response tokens, weighted by the reference's
truncated importance-sampling ratio against the rollout engine's log-probs.

Every kernel here takes the tensor-parallel group explicitly and reduces
across vocab shards itself, so the CPU parity tests run it at world size one
against the pure-Python oracle in ``tests/reef_service/reference_algorithms/sdft.py``.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from reef.train.slime_backend.algorithm import objective

KL_DIRECTIONS = ("forward", "reverse")


class _SumAcrossVocabShards(torch.autograd.Function):
    """Sum per-row partials over the tensor-parallel vocab shards.

    Every rank computes the same total, so the gradient of a replicated loss
    passes through to each rank's partial unchanged.
    """

    @staticmethod
    def forward(ctx: Any, partial: torch.Tensor, tp_group: Any) -> torch.Tensor:
        total = partial.clone()
        dist.all_reduce(total, op=dist.ReduceOp.SUM, group=tp_group)
        return total

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def sum_across_vocab_shards(partial: torch.Tensor, tp_group: Any, tp_world: int) -> torch.Tensor:
    """``partial`` summed over the vocab shards, differentiable; the identity at world size one."""
    if tp_world <= 1:
        return partial
    return _SumAcrossVocabShards.apply(partial, tp_group)


def global_log_sum_exp(logits: torch.Tensor, tp_group: Any, tp_world: int) -> torch.Tensor:
    """log-sum-exp over the full vocabulary of ``[R, V_local]`` logit rows, differentiable."""
    row_max = logits.detach().max(dim=-1).values
    if tp_world > 1:
        dist.all_reduce(row_max, op=dist.ReduceOp.MAX, group=tp_group)
    shard_sum = (logits - row_max[:, None]).exp().sum(dim=-1)
    return row_max + sum_across_vocab_shards(shard_sum, tp_group, tp_world).log()


def token_kl(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    direction: str,
    tp_group: Any,
    tp_world: int,
) -> torch.Tensor:
    """Per-position KL over the full vocabulary between the teacher and the student.

    ``student_logits`` and ``teacher_log_probs`` are ``[R, V_local]`` rows of
    one vocab shard at the same response positions; the teacher rows are
    already normalized over the full vocabulary. ``forward`` is
    KL(teacher || student), the reference's default and what the paper's
    results used (GKD-style); ``reverse`` is KL(student || teacher). Returns
    one value per row.

    The forward KL is assembled from three shard sums, ``sum p_T log p_T``,
    ``sum p_T z_S`` and the teacher's total mass, plus the student's global
    log-sum-exp; the mass is one up to the storage precision of the teacher
    rows, so its gradient is the exact ``p_S - p_T``.
    """
    if direction not in KL_DIRECTIONS:
        raise ValueError(f"sdft kl direction must be one of {', '.join(KL_DIRECTIONS)}, got {direction!r}")
    student_logits = student_logits.to(torch.promote_types(student_logits.dtype, torch.float32))
    teacher_log_probs = teacher_log_probs.to(torch.promote_types(teacher_log_probs.dtype, torch.float32))
    log_sum_exp = global_log_sum_exp(student_logits, tp_group, tp_world)
    if direction == "forward":
        teacher_probs = teacher_log_probs.exp()
        teacher_mass = sum_across_vocab_shards(teacher_probs.sum(dim=-1), tp_group, tp_world)
        cross = sum_across_vocab_shards((teacher_probs * student_logits).sum(dim=-1), tp_group, tp_world)
        negative_entropy = sum_across_vocab_shards((teacher_probs * teacher_log_probs).sum(dim=-1), tp_group, tp_world)
        return negative_entropy - cross + teacher_mass * log_sum_exp
    return _ReverseKl.apply(student_logits, teacher_log_probs, log_sum_exp.detach(), tp_group, tp_world)


class _ReverseKl(torch.autograd.Function):
    """KL(student || teacher) per row over vocab shards, with its gradient written out.

    The divergence is a sum of per-shard terms that all depend on the
    student's global log-sum-exp; letting autograd differentiate a shard's
    term alone drops the other shards' dependence on that log-sum-exp, and
    the missing piece is a push down on every logit in proportion to its
    probability, flattening the student a little more each step. The exact
    gradient of ``sum_v p_v (log p_v - log t_v)`` is ``p_u ((log p_u - log t_u)
    - KL)`` for every logit ``u``, which needs only the local rows and the
    all-reduced divergence.
    """

    @staticmethod
    def forward(
        ctx: Any,
        student_logits: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        log_sum_exp: torch.Tensor,
        tp_group: Any,
        tp_world: int,
    ) -> torch.Tensor:
        student_log_probs = student_logits - log_sum_exp[:, None]
        probs = student_log_probs.exp()
        gap = student_log_probs - teacher_log_probs
        divergence = (probs * gap).sum(dim=-1)
        if tp_world > 1:
            dist.all_reduce(divergence, op=dist.ReduceOp.SUM, group=tp_group)
        ctx.save_for_backward(probs, gap, divergence)
        return divergence

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        probs, gap, divergence = ctx.saved_tensors
        grad = grad_output[:, None] * probs * (gap - divergence[:, None])
        return grad, None, None, None, None


def chunked_token_kl(
    student_logits: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    direction: str,
    chunk_size: int,
    tp_group: Any,
    tp_world: int,
) -> torch.Tensor:
    """:func:`token_kl` over row chunks, each recomputed in backward.

    A chunk's full-vocabulary intermediates (the teacher's probabilities, the
    student's exponentials) are the size of the logits themselves; recomputing
    them in backward keeps only the chunk inputs resident, the way Slime
    chunks its own log-prob and entropy computation. ``chunk_size <= 0``
    computes the rows in one piece.
    """
    rows = student_logits.size(0)
    if chunk_size <= 0 or rows <= chunk_size:
        return token_kl(student_logits, teacher_log_probs, direction=direction, tp_group=tp_group, tp_world=tp_world)

    def _chunk(student_rows: torch.Tensor, teacher_rows: torch.Tensor) -> torch.Tensor:
        return token_kl(student_rows, teacher_rows, direction=direction, tp_group=tp_group, tp_world=tp_world)

    pieces = [
        checkpoint(
            _chunk,
            student_logits[start : start + chunk_size],
            teacher_log_probs[start : start + chunk_size],
            use_reentrant=False,
        )
        for start in range(0, rows, chunk_size)
    ]
    return torch.cat(pieces, dim=0)


def mix_teacher_weights(
    teacher: Mapping[str, torch.Tensor], actor: Mapping[str, torch.Tensor], update_rate: float
) -> None:
    """Move the teacher's floating-point tensors toward the actor's: ``teacher = (1 - rate) * teacher + rate * actor``.

    The reference implementation's ``ref_model_mixup_alpha`` update, applied
    in place. The teacher tensors are the higher-precision accumulator (the
    actor's may be bfloat16); integer buffers are left as they are.
    """
    if not 0 <= update_rate <= 1:
        raise ValueError(f"the teacher update rate must be in [0, 1], got {update_rate}")
    for name, target in teacher.items():
        if not target.is_floating_point():
            continue
        target.mul_(1.0 - update_rate).add_(actor[name].to(dtype=target.dtype), alpha=update_rate)


def sequence_importance_weight(
    student_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    cap: float,
) -> torch.Tensor:
    """The reference's truncated importance-sampling weight of one sample.

    Per token, ``min(pi_theta(y_t) / pi_rollout(y_t), cap)``; the weight is
    its mean over the trained response tokens. It corrects the mismatch
    between the rollout engine and the trainer (and, on Reef, any admitted
    staleness), the way TRL's ``vllm_importance_sampling_correction`` does.
    """
    log_ratio = student_log_probs - rollout_log_probs
    ratio = log_ratio.to(torch.promote_types(log_ratio.dtype, torch.float32)).exp().clamp(max=cap)
    mask = loss_mask.to(ratio.dtype)
    return (ratio * mask).sum() / mask.sum().clamp(min=1.0)


@objective("custom_loss_function_path")
def sdft_loss(
    args: Namespace,
    batch: dict[str, Any],
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``--custom-loss-function-path`` entry point: the per-sample mean token KL to the self-teacher.

    ``batch["sdft_teacher_log_probs"]`` holds, per sample, the teacher's
    ``[R, V_local]`` rows the pre-train hook computed; ``logits`` is the
    training forward over the plain request. Slime's outer ``loss_function``
    divides the returned sum of per-sample means by the step's global batch
    size, which yields the reference's batch mean.
    """
    from megatron.core import mpu
    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_responses

    if mpu.get_context_parallel_world_size() > 1:
        raise NotImplementedError("the sdft loss supports context parallel = 1 only")
    teacher_rows = batch.get("sdft_teacher_log_probs")
    if teacher_rows is None:
        raise RuntimeError("sdft_teacher_log_probs is missing: the sdft pre-train hook did not score this batch")
    direction = str(args.sdft_kl_direction)
    importance_sampling_cap = float(args.sdft_importance_sampling_cap)
    skip_response_tokens = int(args.sdft_skip_response_tokens)
    chunk_size = int(args.log_probs_chunk_size)
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1

    total_lengths = batch["total_lengths"]
    response_lengths = batch["response_lengths"]
    unconcat_tokens = batch["unconcat_tokens"]
    student_rows_per_sample = get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
    )
    per_sample_kl: list[torch.Tensor] = []
    for index, ((student_rows, _), teacher) in enumerate(zip(student_rows_per_sample, teacher_rows, strict=True)):
        if teacher.size(0) != student_rows.size(0):
            raise ValueError(
                f"sdft sample {index} has {teacher.size(0)} teacher rows for a {student_rows.size(0)}-token response"
            )
        per_sample_kl.append(
            chunked_token_kl(
                student_rows,
                teacher.to(device=student_rows.device, non_blocking=True),
                direction=direction,
                chunk_size=chunk_size,
                tp_group=tp_group,
                tp_world=tp_world,
            )
        )

    # The reference leaves the first response tokens out of the loss and of
    # its per-sample denominator.
    loss_masks = batch["loss_masks"]
    if skip_response_tokens > 0:
        loss_masks = [mask.clone() for mask in loss_masks]
        for mask in loss_masks:
            mask[:skip_response_tokens] = 0
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths, response_lengths, loss_masks, None, args.calculate_per_token_loss
        )

    kl = torch.cat(per_sample_kl, dim=0)
    weighted_kl = kl
    metrics: dict[str, torch.Tensor] = {}
    if importance_sampling_cap > 0:
        rollout_log_probs = batch.get("rollout_log_probs")
        if rollout_log_probs is None:
            raise ValueError(
                "the sdft importance-sampling correction needs rollout_log_probs: serve through a backend that "
                "captures them, or pass --sdft-importance-sampling-cap 0"
            )
        with torch.no_grad():
            _, outputs = get_log_probs_and_entropy(
                logits,
                args=args,
                unconcat_tokens=unconcat_tokens,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                with_entropy=False,
            )
            weights = [
                sequence_importance_weight(student, rollout, mask, importance_sampling_cap)
                for student, rollout, mask in zip(outputs["log_probs"], rollout_log_probs, loss_masks, strict=True)
            ]
            student_log_probs = torch.cat(outputs["log_probs"], dim=0).float()
            engine_log_probs = torch.cat(rollout_log_probs, dim=0).float()
        weighted_kl = torch.cat([sample_kl * weight for sample_kl, weight in zip(per_sample_kl, weights, strict=True)])
        # Slime sums a micro-batch's metrics over its samples and divides the
        # step's total by the global batch size, so every value here is a sum
        # of per-sample means, as ``sum_of_sample_mean`` produces.
        metrics["sdft_is_weight"] = torch.stack(weights).sum()
        # How far the trainer's forward sits from the rollout engine on the
        # sampled tokens: a large gap means a mismatch to fix, not to weight.
        metrics["sdft_student_log_prob"] = sum_of_sample_mean(student_log_probs)
        metrics["sdft_rollout_log_prob"] = sum_of_sample_mean(engine_log_probs)
        metrics["sdft_log_prob_abs_diff"] = sum_of_sample_mean((student_log_probs - engine_log_probs).abs())

    loss = sum_of_sample_mean(weighted_kl)
    if weighted_kl.numel() == 0:
        loss = loss + 0 * logits.sum()
    metrics["loss"] = loss.detach().clone()
    metrics["sdft_kl"] = sum_of_sample_mean(kl.detach())
    return loss, metrics


@objective("reef_actor_pre_train_hook_path")
def sdft_actor_pre_train(actor: Any, rollout_data: dict[str, Any]) -> None:
    """Move the teacher toward the policy (seeding it on the first step), then score every teacher sequence."""
    if not rollout_data.get("teacher_tokens"):
        raise ValueError("every sdft sample must carry teacher_tokens")
    from slime.utils.timer import timer

    from recipes.sdft.slime.teacher import compute_sdft_teacher_log_probs

    with timer("sdft_teacher"):
        compute_sdft_teacher_log_probs(actor, rollout_data)
