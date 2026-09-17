"""Slime implementation of SDFT (Self-Distillation Fine-Tuning)."""

from __future__ import annotations

import argparse
import math
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from recipes.sdft.slime.utils.data_builder import build_sdft_rollout_data, sdft_sample_row
from reef.train.slime_backend.algorithm import SlimeAlgorithm, register_loss_family

KL_DIRECTIONS = ("forward", "reverse")
# The reference trainer's defaults (idanshen/Self-Distillation, ``distil_config.py``):
# forward KL, truncated importance sampling capped at 2, every response token trained.
DEFAULT_KL_DIRECTION = "forward"
DEFAULT_IMPORTANCE_SAMPLING_CAP = 2.0
DEFAULT_SKIP_RESPONSE_TOKENS = 0
# The reference's ``ref_model_mixup_alpha`` as ``main.py`` runs it: the teacher
# is a copy of the weights that moves toward the policy by this fraction after
# every step (``sync_ref_model=True, ref_model_sync_steps=1``).
DEFAULT_TEACHER_UPDATE_RATE = 0.01


@dataclass(frozen=True)
class SdftSettings:
    """SDFT driver options parsed from the ``--sdft-*`` flag family."""

    kl_direction: str = DEFAULT_KL_DIRECTION
    importance_sampling_cap: float = DEFAULT_IMPORTANCE_SAMPLING_CAP
    skip_response_tokens: int = DEFAULT_SKIP_RESPONSE_TOKENS
    teacher_update_rate: float = DEFAULT_TEACHER_UPDATE_RATE

    def __post_init__(self) -> None:
        if self.kl_direction not in KL_DIRECTIONS:
            raise ValueError(f"sdft kl_direction must be one of: {', '.join(KL_DIRECTIONS)}")
        cap = self.importance_sampling_cap
        if not isinstance(cap, Real) or isinstance(cap, bool) or not math.isfinite(cap) or cap < 0:
            raise ValueError("sdft importance_sampling_cap must be a finite number >= 0 (0 disables the correction)")
        skip = self.skip_response_tokens
        if not isinstance(skip, Integral) or isinstance(skip, bool) or skip < 0:
            raise ValueError("sdft skip_response_tokens must be a non-negative integer")
        rate = self.teacher_update_rate
        if not isinstance(rate, Real) or isinstance(rate, bool) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("sdft teacher_update_rate must be a number in [0, 1]")


@register_loss_family
class SdftAlgorithm(SlimeAlgorithm):
    """The self-teacher loss family: a per-token KL to the demonstration-conditioned model.

    Before each step the pre-train hook runs a forward pass over every
    sample's teacher sequence and keeps the teacher's next-token distribution
    at each response position. The loss then puts the student's distribution
    at the same positions, computed from the training forward over the plain
    request, against it.

    The teacher's weights follow the reference implementation: a copy that
    moves toward the policy by ``--sdft-teacher-update-rate`` after every
    step (``ref_model_mixup_alpha``), kept on the host beside the actor's
    own backup and swapped in for the pass. A rate of 1 makes the current
    policy the teacher with no copy at all; 0 freezes the initial weights.
    """

    loss_family = "sdft"
    loss_type = "custom_loss"
    # The reference's truncated importance-sampling weight compares the
    # policy against the rollout engine's log-probs; Reef ships them per row.
    requires_rollout_logprobs = True
    advantages = "forbidden"
    forbidden_advantages_message = (
        "sdft distils the demonstration-conditioned teacher; the Reef payload must omit advantages"
    )
    # ``teacher_tokens`` is one variable-length id sequence per sample,
    # partitioned with the sample. ``sdft_teacher_log_probs`` is filled by the
    # pre-train hook and forwarded into every micro-batch; both are hidden
    # from Slime's numeric rollout logger.
    rollout_data_keys = ("teacher_tokens",)
    rollout_tensor_dtypes: Mapping[str, str] = {"teacher_tokens": "long"}
    external_batch_keys = ("rollout_log_probs", "sdft_teacher_log_probs")
    rollout_log_skip_keys = ("teacher_tokens", "sdft_teacher_log_probs")
    required_objective_hooks = ("custom_loss_function_path", "reef_actor_pre_train_hook_path")

    # --- stage 1: configure ---

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        # The teacher is the policy the step starts from. A second optimizer
        # step per rollout would distil distributions the first step already
        # moved away from.
        if int(getattr(args, "num_steps_per_rollout", 1) or 1) != 1:
            raise RuntimeError(
                f"{source} requires --num-steps-per-rollout=1: the self-teacher's distributions are computed once "
                "before the step from the current policy"
            )

    def parse_specific_options(self, arguments: Sequence[str]) -> tuple[SdftSettings, list[str]]:
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, argument_default=argparse.SUPPRESS)
        parser.add_argument(
            "--sdft-kl-direction",
            dest="kl_direction",
            choices=list(KL_DIRECTIONS),
            help=(
                "Which per-token KL to minimize: 'forward' is KL(teacher || student), the reference's default "
                "(GKD-style, what the paper's results used); 'reverse' is KL(student || teacher)."
            ),
        )
        parser.add_argument(
            "--sdft-importance-sampling-cap",
            dest="importance_sampling_cap",
            type=float,
            help=(
                "Cap of the truncated importance-sampling weight between the policy and the rollout engine's "
                f"log-probs, averaged over the response. 0 disables it. Default {DEFAULT_IMPORTANCE_SAMPLING_CAP}."
            ),
        )
        parser.add_argument(
            "--sdft-skip-response-tokens",
            dest="skip_response_tokens",
            type=int,
            help=(
                "Response tokens at the start of every sample left out of the loss and its denominator. "
                f"The paper's runs used 3. Default {DEFAULT_SKIP_RESPONSE_TOKENS}."
            ),
        )
        parser.add_argument(
            "--sdft-teacher-update-rate",
            dest="teacher_update_rate",
            type=float,
            help=(
                "Fraction of the current policy mixed into the teacher's weights after every step, the reference's "
                "ref_model_mixup_alpha. 1 makes the current policy the teacher, 0 freezes the initial weights. "
                f"Default {DEFAULT_TEACHER_UPDATE_RATE}."
            ),
        )
        options, remaining = parser.parse_known_args(list(arguments))
        return SdftSettings(**vars(options)), remaining

    def apply_driver_options(self, args: Namespace, options: object | None) -> None:
        super().apply_driver_options(args, options)
        settings = options if isinstance(options, SdftSettings) else SdftSettings()
        args.sdft_kl_direction = settings.kl_direction
        args.sdft_importance_sampling_cap = settings.importance_sampling_cap
        args.sdft_skip_response_tokens = settings.skip_response_tokens
        args.sdft_teacher_update_rate = settings.teacher_update_rate

    def bind(self, config=None, *, critic_steps_per_actor=None, critic_only_steps=0):
        # The settings travel on args; the bound instance stays stateless.
        if config is not None and not isinstance(config, SdftSettings):
            raise TypeError("sdft bridge algorithm config must be SdftSettings")
        return self

    # --- stage 2: shape row ---

    def shape_sample_row(self, sample):
        return sdft_sample_row(sample)

    # --- stage 3: build batch ---

    def build_rollout_data(self, payload, samples):
        return build_sdft_rollout_data(payload, samples, self)


__all__ = ["SdftAlgorithm", "SdftSettings"]
