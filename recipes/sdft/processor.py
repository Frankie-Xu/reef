"""SDFT reported-feedback processor: one rollout and its demonstration, one training unit."""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping

from recipes.sdft.teacher_prompt import ChatTemplateTokenizer, TeacherPromptTokenizer, resolve_teacher_prompt_builder
from reef.core.chat_request import recorded_request
from reef.core.reports import TeacherContextReport
from reef.train.processors.reported import GroupDecision, ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem

logger = logging.getLogger(__name__)

OVERFLOW_GROUP = "sdft-teacher-overflow"


class SDFTProcessor(ReportedFeedbackProcessor):
    """Turn a rollout and its demonstration into one self-distillation sample.

    A report references one recorded request and carries the demonstration
    as ``context``. The sample keeps the student's policy tensors as every
    weight recipe does and adds ``teacher_tokens``: the teacher prompt (the
    request and the demonstration composed by the configured
    ``TeacherPromptBuilder``) followed by the student's response ids
    verbatim, so the trainer's teacher pass scores the student's own tokens.
    With the recipe default ``batch_size=1`` a report trains as soon as it
    arrives.

    A teacher sequence longer than ``max_teacher_tokens`` cannot be scored by
    the trainer's window. Such a report is not training data: it is released
    with its inference record and counted in ``teacher_overflow_reports``.
    """

    output_schema = TrainingBatch
    exclusive_sources = True

    def __init__(self, context: ProcessorContext, tokenizer: TeacherPromptTokenizer | None = None) -> None:
        config = context.config
        self._assembly = SampleAssembly.from_config(context)
        self._prompt_builder = resolve_teacher_prompt_builder(config)
        self._max_teacher_tokens = int(config.get("max_teacher_tokens", 0))
        if self._max_teacher_tokens < 0:
            raise ValueError("max_teacher_tokens must be non-negative (0 disables the limit)")
        if tokenizer is None:
            tokenizer_path = str(config.get("tokenizer_path", "")).strip()
            if not tokenizer_path:
                raise ValueError(
                    "SDFT requires tokenizer_path: the served model's tokenizer renders the teacher prompt"
                )
            tokenizer = ChatTemplateTokenizer(tokenizer_path)
        self._tokenizer = tokenizer
        self._overflow_reports: set[str] = set()
        self._overflow_count = 0
        super().__init__(context)

    def operational_metrics(self) -> Mapping[str, float | int]:
        return {**super().operational_metrics(), "teacher_overflow_reports": self._overflow_count}

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        parsed = context.parsed_report
        if not isinstance(parsed, TeacherContextReport):
            raise ValueError("SDFTProcessor requires the recipe's report schema")
        if len(context.inferences) != 1:
            raise ValueError(
                f"SDFT trains one recorded request per report; report {context.report.agent_record_id} "
                f"references {len(context.inferences)}"
            )
        # The demonstration is the signal; a reported score is metadata only.
        sample = self._assembly.build(context, 0.0 if context.score is None else context.score)
        tokens = [int(token) for token in sample.training.get("tokens", [])]
        response_length = len(sample.training.get("loss_mask", []))
        if not 0 < response_length < len(tokens):
            raise ValueError("SDFT requires the recorded prompt and response tokens of the inference")
        messages, tools = recorded_request(context.inferences[0].payload)
        teacher = self._prompt_builder.build(messages, tools, parsed.context)
        prompt_ids = self._tokenizer.prompt_token_ids(teacher.messages, teacher.tools)
        teacher_tokens = [*prompt_ids, *tokens[-response_length:]]
        if self._max_teacher_tokens and len(teacher_tokens) > self._max_teacher_tokens:
            self._overflow_reports.add(context.report.agent_record_id)
            logger.warning(
                "sdft report %s skipped: its teacher sequence is %d tokens, over max_teacher_tokens %d",
                context.report.agent_record_id,
                len(teacher_tokens),
                self._max_teacher_tokens,
            )
        return sample.with_training(teacher_tokens=teacher_tokens)

    def grouping(self, context: ReportContext) -> tuple[Hashable | None, Hashable | None]:
        # Only an overflowing report forms a group, so that the group decision
        # can release it; every other report is an independent unit.
        report_id = context.report.agent_record_id
        if report_id in self._overflow_reports:
            return (OVERFLOW_GROUP, report_id), None
        return None, None

    def decide_group(self, key: Hashable, items: tuple[TrainDataItem, ...]) -> GroupDecision:
        if not isinstance(key, tuple) or key[0] != OVERFLOW_GROUP:
            raise ValueError(f"SDFTProcessor groups only overflowing reports, got group key {key!r}")
        self._overflow_reports.discard(key[1])
        self._overflow_count += 1
        return GroupDecision.DISCARD

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:sdft:{batch_number}", items)
