"""Shared policy processor test double: assemble every valid scored report."""

from __future__ import annotations

from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor, ReportSample, SampleAssembly
from reef.train.types import PolicyBatch, ProcessorContext


class ThresholdProcessor(ReportedFeedbackProcessor):
    """One report becomes one sample, with no score filtering."""

    output_schema = PolicyBatch

    def __init__(self, context: ProcessorContext) -> None:
        self._assembly = SampleAssembly.from_config(context)
        super().__init__(context)

    def make_sample(self, context: ReportContext) -> ReportSample:
        return ReportSample(self._assembly.build(context, context.require_score()))

    def make_batch(self, units, batch_number: int) -> PolicyBatch:
        return PolicyBatch(
            f"{self.scenario}:threshold:{batch_number}",
            tuple(unit.candidates[0].value for unit in units),
        )
