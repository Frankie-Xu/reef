"""Synthetic CaseEvents encoded with Reef's report contract and record type."""

from dataclasses import dataclass
from datetime import UTC, datetime

from reef_casegraph_adapter import CaseEvent, digest

from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports.base import ReportBase


@dataclass(frozen=True)
class CaseEventReport(ReportBase):
    """Recipe-local report schema; preserves the full event including provenance."""

    event_json: str

    def validate(self) -> None:
        CaseEvent.deserialize(self.event_json)


def event_from_record(record: AgentRecord) -> CaseEvent:
    if record.request_type != RequestType.REPORT:
        raise ValueError("expected a report record")
    report = CaseEventReport.from_dict(record.payload)
    if not isinstance(report, CaseEventReport):
        raise TypeError("expected CaseEventReport")
    return CaseEvent.deserialize(report.event_json)


def event_to_record(event: CaseEvent, *, scenario: str, observation: AgentRecord | None = None) -> AgentRecord:
    """Build a record; feedback references its observation receipt in the same scenario.

    This function does not append. Callers must keep retained/drift arms out of
    training storage; the integration test uses a disposable local SQLite store.
    """
    if not scenario.startswith("synthetic-"):
        raise ValueError("expected a synthetic scenario")
    references: tuple[str, ...] = ()
    if event.event_type == "feedback":
        if observation is None or observation.scenario != scenario:
            raise ValueError("feedback needs an observation receipt in the same scenario")
        source = event_from_record(observation)
        if (
            source.event_type != "observation"
            or source.case_id != event.case_id
            or source.artifact_version != event.artifact_version
            or source.observed_at > event.observed_at
        ):
            raise ValueError("feedback observation does not match case, version, or time")
        references = (observation.agent_record_id,)
    elif event.event_type != "observation" or observation is not None:
        raise ValueError("expected an observation without references or feedback with a receipt")
    report = CaseEventReport(event_json=event.serialize())
    record_id = "caseevent:" + digest([scenario, event.event_id])
    payload = report.to_dict(references=references, agent_record_id=record_id)
    if event.feedback_type is not None:
        payload["feedback"] = event.feedback_type
    return AgentRecord.create(
        scenario=scenario,
        request_type=RequestType.REPORT,
        payload=payload,
        agent_record_id=record_id,
        created_at=datetime.fromisoformat(event.observed_at).replace(tzinfo=UTC).timestamp(),
        references=references,
    )
