"""Reported feedback contracts: existing references, sample assembly, and consumption."""

from __future__ import annotations

from dataclasses import replace

import pytest

from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportValidationError
from reef.train.processors.reported import (
    BatchUnit,
    Candidate,
    GroupDecision,
    ReportContext,
    ReportedFeedbackProcessor,
    ReportSample,
)
from reef.train.types import PolicyBatch, ProcessorContext


def inference(record_id: str) -> AgentRecord:
    return AgentRecord.create(
        scenario="math", request_type=RequestType.INFERENCE, payload={}, agent_record_id=record_id
    )


def report(record_id: str, *references: str, score: float = 1.0) -> AgentRecord:
    return AgentRecord.create(
        scenario="math",
        request_type=RequestType.REPORT,
        payload={"score": score, "references": list(references)},
        agent_record_id=record_id,
    )


class SampleProcessor(ReportedFeedbackProcessor):
    output_schema = PolicyBatch
    exclusive_sources = True

    def __init__(self, batch_size: int = 1) -> None:
        super().__init__(ProcessorContext("math", {"batch_size": batch_size}))
        self.assembled: list[str] = []
        self.fail_assembly = False

    def make_sample(self, context: ReportContext) -> ReportSample:
        if self.fail_assembly:
            raise ValueError("broken training data")
        self.assembled.append(context.report.agent_record_id)
        return ReportSample(context.report.agent_record_id)

    def make_batch(self, units: tuple[BatchUnit, ...], batch_number: int) -> PolicyBatch:
        return PolicyBatch(f"batch:{batch_number}", tuple(c.value for unit in units for c in unit.candidates))


class GroupProcessor(SampleProcessor):
    def __init__(self, batch_size: int = 1) -> None:
        super().__init__(batch_size)
        self.discard = False

    def make_sample(self, context: ReportContext) -> ReportSample:
        metadata = context.report.payload.get("metadata", {})
        return ReportSample(
            context.report.agent_record_id, group_key=metadata.get("group", "g"), slot=metadata.get("slot")
        )

    def decide_group(self, key: object, candidates: tuple[Candidate, ...]) -> GroupDecision:
        if len(candidates) < 2:
            return GroupDecision.INCOMPLETE
        return GroupDecision.DISCARD if self.discard else GroupDecision.READY


def test_processor_requires_sample_assembly_instead_of_judge() -> None:
    with pytest.raises(TypeError, match="abstract"):
        ReportedFeedbackProcessor(ProcessorContext("math"))
    assert not hasattr(ReportedFeedbackProcessor, "judge")


def test_inferences_wait_for_reports_and_batch_counts_completed_samples() -> None:
    processor = SampleProcessor(batch_size=2)
    for record_id in ("i1", "i2", "i3"):
        processor.ingest(inference(record_id))
    assert not processor.ready()
    processor.ingest(report("r1", "i1"))
    assert not processor.ready()
    processor.ingest(report("r2", "i2"))
    assert processor.build_batch().samples == ("r1", "r2")
    assert "i3" in processor.retention_decision().protected_agent_record_ids


def test_report_before_inference_fails_without_queuing_it() -> None:
    processor = SampleProcessor()
    with pytest.raises(ReportValidationError, match="unavailable"):
        processor.ingest(report("r1", "i1"))
    processor.ingest(inference("i1"))
    assert not processor.ready()
    assert processor.assembled == []
    processor.ingest(report("r1", "i1"))
    assert processor.build_batch().samples == ("r1",)


@pytest.mark.parametrize("references", [(), ("i1", "i1"), ("i1", "missing")])
def test_invalid_references_do_not_create_a_sample(references: tuple[str, ...]) -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    with pytest.raises(ReportValidationError):
        processor.ingest(report("r1", *references))
    assert not processor.ready()
    assert processor.retention_decision().protected_agent_record_ids == {"i1"}


@pytest.mark.parametrize("request_type", [RequestType.INFERENCE, RequestType.REPORT])
def test_processor_rejects_cross_scenario_records(request_type: RequestType) -> None:
    processor = SampleProcessor()
    record = inference("i1") if request_type is RequestType.INFERENCE else report("r1", "i1")
    with pytest.raises(ReportValidationError, match="scenario"):
        processor.ingest(replace(record, scenario="other"))


@pytest.mark.parametrize("eligible", [True, False, None])
def test_reports_cannot_set_training_eligibility(eligible: object) -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    record = report("r1", "i1")
    record = replace(record, payload={**record.payload, "metadata": {"training": {"eligible": eligible}}})
    with pytest.raises(ReportValidationError, match="eligible"):
        processor.ingest(record)
    assert not processor.ready()


@pytest.mark.parametrize("score", [0.0, -1.0, 1.0])
def test_valid_feedback_is_assembled_without_score_filtering(score: float) -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1", score=score))
    assert processor.build_batch().samples == ("r1",)


def test_assembly_failure_protects_inputs_and_does_not_acknowledge_a_retry() -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    processor.fail_assembly = True
    for _ in range(2):
        with pytest.raises(ValueError, match="broken training data"):
            processor.ingest(report("r1", "i1"))
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "r1"}
    assert not processor.ready()
    processor.fail_assembly = False
    processor.ingest(report("r1", "i1"))
    assert processor.build_batch().samples == ("r1",)


def test_duplicate_reports_and_batch_polls_do_not_assemble_twice() -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1"))
    processor.ingest(report("r1", "i1"))
    batch = processor.build_batch()
    assert processor.build_batch() is batch
    assert processor.assembled == ["r1"]
    processor.release_batch(batch.batch_id)
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "r1"}
    batch = processor.build_batch()
    assert processor.acknowledge(batch.batch_id) == {"i1", "r1"}
    processor.ingest(report("late", "i1"))
    assert not processor.ready()
    assert processor.assembled == ["r1"]


def test_live_report_keeps_a_shared_source_protected_until_its_consumption() -> None:
    processor = SampleProcessor()
    processor.ingest(inference("i1"))
    processor.ingest(report("r1", "i1"))
    processor.ingest(report("r2", "i1"))
    processor.acknowledge(processor.build_batch().batch_id)
    assert processor.retention_decision().protected_agent_record_ids == {"i1", "r2"}
    processor.acknowledge(processor.build_batch().batch_id)
    assert processor.retention_decision().releasable_agent_record_ids == {"i1", "r1", "r2"}
    processor.compaction_applied(frozenset({"i1", "r1", "r2"}))
    assert processor.retention_decision().protected_agent_record_ids == set()
    assert processor.retention_decision().releasable_agent_record_ids == set()


def test_group_waits_for_complete_samples() -> None:
    processor = GroupProcessor()
    for index in (1, 2):
        processor.ingest(inference(f"i{index}"))
        processor.ingest(report(f"r{index}", f"i{index}"))
        assert processor.ready() == (index == 2)
    assert processor.build_batch().samples == ("r1", "r2")


def test_group_discard_releases_all_members_and_refuses_later_members() -> None:
    processor = GroupProcessor()
    processor.discard = True
    for index in (1, 2, 3):
        processor.ingest(inference(f"i{index}"))
        processor.ingest(report(f"r{index}", f"i{index}"))
    assert not processor.ready()
    assert processor.retention_decision().releasable_agent_record_ids == {"i1", "i2", "i3", "r1", "r2", "r3"}


def test_slot_retry_preserves_first_report() -> None:
    processor = GroupProcessor()
    for record_id, slot in (("first", "a"), ("retry", "a"), ("second", "b")):
        processor.ingest(inference(record_id))
        record = report(f"r-{record_id}", record_id)
        processor.ingest(replace(record, payload={**record.payload, "metadata": {"slot": slot}}))
    assert processor.build_batch().samples == ("r-first", "r-second")
    assert {"r-retry", "retry"} <= processor.retention_decision().releasable_agent_record_ids


def test_ordered_groups_can_coexist_with_singleton_samples() -> None:
    class MixedProcessor(SampleProcessor):
        ordered_groups = True

        def make_sample(self, context: ReportContext) -> ReportSample:
            record_id = context.report.agent_record_id
            return ReportSample(record_id, group_key=record_id if record_id.startswith("g-") else None)

        def decide_group(self, key: object, candidates: tuple[Candidate, ...]) -> GroupDecision:
            return GroupDecision.READY

    processor = MixedProcessor(batch_size=3)
    for record_id in ("g-z", "g-a", "single"):
        processor.ingest(inference(f"source-{record_id}"))
        processor.ingest(report(record_id, f"source-{record_id}"))
    assert processor.build_batch().samples == ("single", "g-a", "g-z")
