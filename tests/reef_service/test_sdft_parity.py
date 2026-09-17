"""Pin the SDFT tensor kernels to the pure-Python reference.

The reference (torch free) is the source of truth for the per-token KL and
the importance-sampling weight of ``recipes/sdft/slime/objective.py``. These
tests run both on the same inputs at tensor-parallel world size one, on CPU
tensors, so they are cheap enough for the minimal CI gate that installs CPU
torch. No Megatron is needed: the kernels take the tensor-parallel group as
an argument and skip the collectives at world size one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from recipes.sdft.slime.objective import (
    chunked_token_kl,
    global_log_sum_exp,
    mix_teacher_weights,
    sequence_importance_weight,
    token_kl,
)

from .reference_algorithms import sdft

_ROWS, _VOCAB = 5, 11


def _rows(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    student = torch.randn(_ROWS, _VOCAB, generator=generator, dtype=torch.float64) * 2
    teacher = torch.log_softmax(torch.randn(_ROWS, _VOCAB, generator=generator, dtype=torch.float64) * 2, dim=-1)
    return student, teacher


@pytest.mark.unit
@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("seed", [0, 1])
def test_token_kl_matches_reference(direction: str, seed: int) -> None:
    student, teacher = _rows(seed)

    kl = token_kl(student, teacher, direction=direction, tp_group=None, tp_world=1)

    expected = [sdft.token_kl(student[row].tolist(), teacher[row].tolist(), direction) for row in range(_ROWS)]
    assert kl.tolist() == pytest.approx(expected, abs=1e-9)
    assert all(value >= 0 for value in kl.tolist())


@pytest.mark.unit
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_token_kl_is_zero_when_student_equals_teacher(direction: str) -> None:
    student, _ = _rows(3)
    teacher = torch.log_softmax(student, dim=-1)

    kl = token_kl(student, teacher, direction=direction, tp_group=None, tp_world=1)

    assert kl.tolist() == pytest.approx([0.0] * _ROWS, abs=1e-12)


@pytest.mark.unit
def test_forward_kl_gradient_is_student_minus_teacher() -> None:
    student, teacher = _rows(4)
    student = student.clone().requires_grad_(True)

    token_kl(student, teacher, direction="forward", tp_group=None, tp_world=1).sum().backward()

    expected = torch.softmax(student.detach(), dim=-1) - teacher.exp()
    assert torch.allclose(student.grad, expected, atol=1e-9)


@pytest.mark.unit
def test_reverse_kl_gradient_matches_autograd_of_reference_form() -> None:
    student, teacher = _rows(5)
    student = student.clone().requires_grad_(True)
    token_kl(student, teacher, direction="reverse", tp_group=None, tp_world=1).sum().backward()

    reference = student.detach().clone().requires_grad_(True)
    log_probs = torch.log_softmax(reference, dim=-1)
    (log_probs.exp() * (log_probs - teacher)).sum().backward()

    assert torch.allclose(student.grad, reference.grad, atol=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_chunked_kl_matches_whole_and_backpropagates_through_checkpoint(direction: str) -> None:
    student, teacher = _rows(6)
    whole = student.clone().requires_grad_(True)
    chunked = student.clone().requires_grad_(True)

    whole_kl = token_kl(whole, teacher, direction=direction, tp_group=None, tp_world=1)
    chunked_kl = chunked_token_kl(chunked, teacher, direction=direction, chunk_size=2, tp_group=None, tp_world=1)
    whole_kl.sum().backward()
    chunked_kl.sum().backward()

    assert torch.allclose(chunked_kl, whole_kl)
    assert torch.allclose(chunked.grad, whole.grad)


@pytest.mark.unit
def test_float16_teacher_rows_with_the_storage_floor_keep_the_kl_finite() -> None:
    # The teacher pass stores float16 rows clamped at -1e4; exp of the floor
    # is exactly zero, so a vocabulary the teacher rules out costs nothing.
    student, teacher = _rows(7)
    teacher[:, 0] = -float("inf")
    stored = teacher.clamp_min(-1.0e4).to(torch.float16)

    kl = token_kl(student, stored, direction="forward", tp_group=None, tp_world=1)

    finite_reference = teacher.clone()
    finite_reference[:, 0] = -1.0e4
    expected = [
        sdft.token_kl(student[row].tolist(), finite_reference[row].tolist(), "forward") for row in range(_ROWS)
    ]
    assert torch.isfinite(kl).all()
    # float16 keeps about three significant digits of each stored log-prob.
    assert kl.tolist() == pytest.approx(expected, rel=2e-2, abs=2e-2)


@pytest.mark.unit
def test_global_log_sum_exp_matches_torch_at_world_size_one() -> None:
    student, _ = _rows(8)
    assert torch.allclose(global_log_sum_exp(student, None, 1), torch.logsumexp(student, dim=-1))


@pytest.mark.unit
@pytest.mark.parametrize("cap", [2.0, 0.5])
def test_sequence_importance_weight_matches_reference(cap: float) -> None:
    student = torch.tensor([-0.5, -1.0, -0.2, -2.0], dtype=torch.float64)
    rollout = torch.tensor([-0.4, -2.0, -0.3, -0.1], dtype=torch.float64)
    mask = torch.tensor([1, 1, 0, 1])

    weight = sequence_importance_weight(student, rollout, mask, cap)

    expected = sdft.sequence_importance_weight(student.tolist(), rollout.tolist(), mask.tolist(), cap)
    assert weight.item() == pytest.approx(expected, abs=1e-12)


@pytest.mark.unit
def test_sequence_importance_weight_of_an_untrained_sample_is_zero() -> None:
    zeros = torch.zeros(3, dtype=torch.float64)
    assert sequence_importance_weight(zeros, zeros, torch.zeros(3, dtype=torch.int64), 2.0).item() == 0.0


@pytest.mark.unit
def test_teacher_weights_move_toward_the_actor_in_float32() -> None:
    # The reference's ref = 0.99 * ref + 0.01 * policy, accumulated where a
    # 1% step of a small change survives (bfloat16 would round it away).
    teacher = {"w": torch.full((4,), 1.0, dtype=torch.float32), "steps": torch.tensor([3])}
    actor = {"w": torch.full((4,), 1.0 + 1e-3, dtype=torch.bfloat16), "steps": torch.tensor([9])}

    mix_teacher_weights(teacher, actor, 0.01)

    expected = 0.99 * 1.0 + 0.01 * torch.full((4,), 1.0 + 1e-3, dtype=torch.bfloat16).float()
    assert torch.allclose(teacher["w"], expected)
    assert teacher["w"].dtype == torch.float32
    assert teacher["steps"].tolist() == [3]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        mix_teacher_weights(teacher, actor, 1.5)


# --- the vocab shards -----------------------------------------------------------


def _sharded_kl_worker(rank: int, world: int, port: int, direction: str, seed: int) -> None:
    """One tensor-parallel rank: its vocab shard's KL and gradient must match the dense computation."""
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        generator = torch.Generator().manual_seed(seed)
        rows, vocab = 5, 24
        student = torch.randn(rows, vocab, generator=generator, dtype=torch.float64) * 3
        teacher = torch.log_softmax(torch.randn(rows, vocab, generator=generator, dtype=torch.float64) * 3, dim=-1)
        dense = student.clone().requires_grad_(True)
        dense_kl = token_kl(dense, teacher, direction=direction, tp_group=None, tp_world=1)
        dense_kl.sum().backward()

        shard = slice(rank * vocab // world, (rank + 1) * vocab // world)
        local = student[:, shard].clone().requires_grad_(True)
        kl = token_kl(local, teacher[:, shard], direction=direction, tp_group=dist.group.WORLD, tp_world=world)
        kl.sum().backward()
        assert torch.allclose(kl, dense_kl.detach(), atol=1e-9), (rank, kl, dense_kl)
        assert torch.allclose(local.grad, dense.grad[:, shard], atol=1e-9), (rank, local.grad, dense.grad[:, shard])
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_token_kl_gradient_matches_the_dense_computation_across_vocab_shards(direction: str) -> None:
    # Every rank sees one vocab shard and the all-reduced totals, the way
    # tensor-parallel Megatron hands the loss its logits; the gradient on a
    # shard must be the dense gradient's slice, coupling through the global
    # log-sum-exp included.
    import socket

    import torch.multiprocessing as multiprocessing

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    multiprocessing.spawn(_sharded_kl_worker, args=(4, port, direction, 3), nprocs=4, join=True)
