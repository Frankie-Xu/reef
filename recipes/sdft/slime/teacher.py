"""The self-teacher forward pass: the current policy reads the teacher prompt and scores the student's response.

Modeled on the frozen-base teacher in ``recipes/openclawrl/slime/teacher.py``,
with two differences. The teacher IS the current policy (with LoRA, the same
frozen base plus the same adapter), so no weight backup is swapped in: the
pass is slime's own ``forward_only`` over the actor's model on the teacher
sequences the processor built (teacher prompt ids plus the student's response
ids verbatim). And the loss needs the teacher's whole next-token distribution
at every response position rather than a top-K gather, so each sample's rows
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

from recipes.sdft.slime.objective import global_log_sum_exp

#: Log-probs are stored in float16; the floor keeps every stored value inside
#: its range while ``exp`` of it is still exactly zero in float32.
TEACHER_LOG_PROB_FLOOR = -1.0e4


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
    """Fill ``rollout_data["sdft_teacher_log_probs"]`` from a forward pass of the current actor.

    One forward-only pass over the batch's teacher sequences, packed under
    the token budget; the actor's weights are the ones training is about to
    use, so nothing is backed up or restored.
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
    result = forward_only(
        gather_teacher_log_probs,
        args,
        actor.model,
        [DataIterator(view, schedule) for _ in range(vpp)],
        [len(schedule)],
    )
    if not result:
        return  # not the last pipeline stage; the loss does not run here
    rollout_data["sdft_teacher_log_probs"] = result["teacher_log_probs"]


__all__ = ["compute_sdft_teacher_log_probs", "gather_teacher_log_probs", "pack_forward_schedule"]
