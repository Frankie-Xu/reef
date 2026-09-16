"""The self-teacher forward pass: the teacher reads the demonstration prompt and scores the student's response.

Modeled on the frozen-base teacher in ``recipes/openclawrl/slime/teacher.py``.
The teacher's weights are the reference implementation's ``ref_model``: a
copy of the initial weights that moves toward the policy by
``--sdft-teacher-update-rate`` after every step. The copy lives in the
actor's weight backups on the host (Slime's ``TensorBackuper``, bfloat16
and pinned) with a float32 accumulator beside it, so the small updates the
reference applies do not vanish in bfloat16 rounding; the pass swaps it in
through ``_switch_model`` and swaps the actor back. At a rate of 1 no copy
exists and the pass runs on the actor's own weights (with LoRA, the same
frozen base plus the same adapter as the student).

The pass is slime's own ``forward_only`` over the teacher sequences the
processor built (teacher prompt ids plus the student's response ids
verbatim). The loss needs the teacher's whole next-token distribution at
every response position rather than a top-K gather, so each sample's rows
``[R, V_local]`` (this rank's vocab shard, normalized over the full
vocabulary) are kept in float16 on the CPU until the loss moves the
micro-batch's rows back to the device.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from slime.backends.megatron_utils.data import DataIterator
from slime.backends.megatron_utils.model import forward_only

from recipes.sdft.slime.objective import global_log_sum_exp, mix_teacher_weights

#: Log-probs are stored in float16; the floor keeps every stored value inside
#: its range while ``exp`` of it is still exactly zero in float32.
TEACHER_LOG_PROB_FLOOR = -1.0e4
#: The actor's backup tags: Slime's own copy of the training weights, and the teacher's.
ACTOR_TAG = "actor"
TEACHER_TAG = "sdft_teacher"
#: The float32 accumulator of the teacher's weights, per rank; the bfloat16
#: backup under ``TEACHER_TAG`` is rewritten from it before every pass.
_teacher_accumulator: dict[str, torch.Tensor] = {}


def initialize_teacher(actor: Any) -> None:
    """Seed the teacher from the actor's weights at init.

    On a fresh start these are the base model's weights, the reference's
    starting teacher. A restart from a Megatron checkpoint seeds the teacher
    from the resumed weights instead: the copy the interrupted run had moved
    is not checkpointed.
    """
    update_rate = float(actor.args.sdft_teacher_update_rate)
    if update_rate >= 1.0:
        return
    actor.weights_backuper.backup(TEACHER_TAG)
    _teacher_accumulator.clear()
    if update_rate > 0.0:
        for name, tensor in actor.weights_backuper.get(TEACHER_TAG).items():
            _teacher_accumulator[name] = tensor.float() if tensor.is_floating_point() else tensor.clone()


def pack_forward_schedule(lengths: list[int], budget: int) -> list[list[int]]:
    """Greedy contiguous packing of sample indices under a token budget.

    Mirrors dynamic batching's invariant (per-microbatch token sum stays
    under ``max_tokens_per_gpu``); order is preserved and ``forward_only``
    unpermutes by these indices afterwards. A single sample over budget gets
    its own microbatch; the caller guards the model's sequence capacity.
    """
    schedule: list[list[int]] = []
    current: list[int] = []
    used = 0
    for index, length in enumerate(lengths):
        if current and used + length > budget:
            schedule.append(current)
            current, used = [], 0
        current.append(index)
        used += length
    if current:
        schedule.append(current)
    return schedule


def gather_teacher_log_probs(
    logits: torch.Tensor,
    *,
    args: Any,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """``forward_only`` callback: per sample, the teacher's log-probs over this rank's vocab shard.

    Indexing mirrors ``get_log_probs_and_entropy``'s cp1 branch: the packed
    stream concatenates samples by ``total_length``, and the logits row
    predicting response token ``j`` sits at
    ``offset + total_length - response_length - 1 + j``. Logits are divided by
    ``rollout_temperature`` as the loss divides the student's, and normalized
    over the full vocabulary in row chunks of ``log_probs_chunk_size``.

    Returns Megatron's legacy 2-tuple ``(loss, reduced)`` the way the
    runtime's own log-prob callback does: an empty loss tensor, and the
    collected rows as the reduced dict.
    """
    if mpu.get_context_parallel_world_size() > 1:
        raise NotImplementedError("the sdft self-teacher supports context parallel = 1 only")
    if logits.size(0) != 1:
        raise ValueError(f"teacher logits must have batch size 1, got {logits.shape}")
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
    chunk_size = int(args.log_probs_chunk_size)

    rows_per_sample: list[torch.Tensor] = []
    with torch.no_grad():
        tempered = logits.squeeze(0).float()
        temperature = float(args.rollout_temperature)
        if temperature != 1.0:
            tempered = tempered / temperature
        offset = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=True):
            rows = tempered[offset + total_length - response_length - 1 : offset + total_length - 1]
            stored: list[torch.Tensor] = []
            step = rows.size(0) if chunk_size <= 0 else chunk_size
            for start in range(0, rows.size(0), step):
                chunk = rows[start : start + step]
                log_probs = chunk - global_log_sum_exp(chunk, tp_group, tp_world)[:, None]
                stored.append(log_probs.clamp_min(TEACHER_LOG_PROB_FLOOR).to(torch.float16).cpu())
            rows_per_sample.append(torch.cat(stored, dim=0))
            offset += total_length
    return torch.empty((0,), device=logits.device), {"teacher_log_probs": rows_per_sample}


def compute_sdft_teacher_log_probs(actor: Any, rollout_data: dict[str, Any]) -> None:
    """Fill ``rollout_data["sdft_teacher_log_probs"]`` from a forward pass of the teacher.

    One forward-only pass over the batch's teacher sequences, packed under
    the token budget. With a teacher copy, the actor's backup is refreshed
    first (so the restore afterwards returns the weights training is about
    to use), the copy moves toward the actor by the update rate, and the
    pass runs on the copy.
    """
    teacher_tokens: list[torch.Tensor] = rollout_data["teacher_tokens"]
    response_lengths = [int(value) for value in rollout_data["response_lengths"]]
    args = actor.args
    capacity = int(args.seq_length)
    lengths = [int(tokens.numel()) for tokens in teacher_tokens]
    for index, (length, response_length) in enumerate(zip(lengths, response_lengths, strict=True)):
        if length <= response_length:
            raise ValueError(f"sdft sample {index} teacher sequence carries no prompt before its response")
        if length > capacity:
            raise ValueError(
                f"sdft sample {index} teacher sequence is {length} tokens, over the trainer's --seq-length "
                f"{capacity}; set the recipe's max_teacher_tokens so such reports are skipped"
            )
    budget = int(args.max_tokens_per_gpu or capacity)
    device = torch.cuda.current_device()
    vpp = mpu.get_virtual_pipeline_model_parallel_world_size() or 1

    pass_tokens = [tokens.to(device=device, dtype=torch.long) for tokens in teacher_tokens]
    schedule = pack_forward_schedule(lengths, budget)
    view = {
        "tokens": pass_tokens,
        "loss_masks": rollout_data["loss_masks"],
        "total_lengths": lengths,
        "response_lengths": response_lengths,
        "micro_batch_indices": schedule,
    }
    update_rate = float(args.sdft_teacher_update_rate)
    teacher_is_a_copy = update_rate < 1.0
    if teacher_is_a_copy:
        backuper = actor.weights_backuper
        if TEACHER_TAG not in backuper.backup_tags:
            raise RuntimeError("the sdft teacher was never initialized; the actor init hook did not run")
        backuper.backup(ACTOR_TAG)
        if update_rate > 0.0:
            mix_teacher_weights(_teacher_accumulator, backuper.get(ACTOR_TAG), update_rate)
            teacher_backup = backuper.get(TEACHER_TAG)
            for name, accumulated in _teacher_accumulator.items():
                teacher_backup[name].copy_(accumulated)
        actor._switch_model(TEACHER_TAG)
    try:
        result = forward_only(
            gather_teacher_log_probs,
            args,
            actor.model,
            [DataIterator(view, schedule) for _ in range(vpp)],
            [len(schedule)],
        )
    finally:
        if teacher_is_a_copy:
            actor._switch_model(ACTOR_TAG)
    if not result:
        return  # not the last pipeline stage; the loss does not run here
    rollout_data["sdft_teacher_log_probs"] = result["teacher_log_probs"]


__all__ = ["compute_sdft_teacher_log_probs", "gather_teacher_log_probs", "initialize_teacher", "pack_forward_schedule"]
