"""SAO reported-feedback processor: one completed rollout, one training unit."""

from __future__ import annotations

from dataclasses import replace

from reef.core.records_types import AgentRecord
from reef.train.processors.common import make_policy_sample
from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor, ReportSample, SampleAssembly
from reef.train.types import PolicyBatch, PolicySample, ProcessorContext


def make_sao_sample(item: AgentRecord, reward: float) -> PolicySample:
    """Convert inference data and its evaluated reward into an SAO sample.

    ``make_policy_sample`` builds the policy 5-tuple, so SAO and the
    group-relative processors resolve every shared field identically —
    including the ``runtime_load_id`` fallback chain the durable runtime needs
    to identify a training job's producing version. SAO then fills the two
    fields that path leaves at their defaults: ``action_mask`` (read from
    ``response.training`` first, the top-level payload second) and
    ``rollout_created_at``, for the backend's queue-age metric.
    """
    base = make_policy_sample(item, reward)
    payload = item.payload
    response = payload.get("response", {})
    training = response.get("training", {}) if isinstance(response, dict) else {}
    action_mask = training.get("action_mask", payload.get("action_mask", ())) if isinstance(training, dict) else ()
    return replace(base, action_mask=tuple(int(value) for value in action_mask), rollout_created_at=item.created_at)


class SAOProcessor(ReportedFeedbackProcessor):
    """Turn scored rollouts into independently-scheduled SAO samples.

    Single-Rollout Asynchronous Optimization ships each completed rollout on
    its own — no comparison group, no slowest-sample barrier. With the recipe
    default ``batch_size=1`` the dispatcher trains once per accepted rollout, so
    a rollout enters training the moment its score arrives.

    SAO reuses ``PolicySample`` / ``PolicyBatch`` and fills ``action_mask``
    and ``rollout_created_at``. The training backend validates required
    tensors; malformed training input fails explicitly.
    """

    output_schema = PolicyBatch
    exclusive_sources = True

    def __init__(self, context: ProcessorContext) -> None:
        self._assembly = SampleAssembly.from_config(context, make_sample=make_sao_sample)
        super().__init__(context)

    def make_sample(self, context: ReportContext) -> ReportSample:
        sample = self._assembly.build(context, context.require_score())
        if not sample.action_mask:
            sample = replace(sample, action_mask=sample.loss_mask)
        return ReportSample(sample)

    def make_batch(self, units, batch_number: int) -> PolicyBatch:
        return PolicyBatch(
            f"{self.scenario}:sao:{batch_number}",
            tuple(unit.candidates[0].value for unit in units),
        )
