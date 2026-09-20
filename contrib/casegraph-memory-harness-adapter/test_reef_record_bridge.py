from dataclasses import replace
from pathlib import Path

import pytest
from reef_casegraph_adapter import CaseGraphAdapter, SyntheticCaseEventGenerator
from reef_record_bridge import event_from_record, event_to_record

from reef.core.records_types import RequestType
from reef.storage.records import RecordConflict
from reef.storage.sqlite import SQLiteRecordStore


@pytest.mark.integration
def test_real_sqlite_report_round_trip_and_receipt_linkage(tmp_path: Path) -> None:
    generator = SyntheticCaseEventGenerator()
    splits = generator.split(generator.generate())
    events = splits["replay"] + splits["adapt"]
    database = tmp_path / "records.sqlite"
    store = SQLiteRecordStore(database)
    try:
        observations = {}
        for event in events:
            record = event_to_record(
                event, scenario="synthetic-casegraph", observation=observations.get(event.case_id)
            )
            receipt = store.append(record)
            if event.event_type == "observation":
                observations[event.case_id] = receipt
            else:
                assert receipt.references == ()
                assert receipt.payload["metadata"]["source_report_id"] == observations[event.case_id].agent_record_id
                assert receipt.payload["feedback"] == event.feedback_type
            assert receipt.request_type == RequestType.REPORT
            assert not store.append_result(record).inserted
            assert event_from_record(receipt) == event
        with pytest.raises(RecordConflict):
            store.append(replace(receipt, payload={**receipt.payload, "feedback": "changed"}))
    finally:
        store.close()
    reopened = SQLiteRecordStore(database)
    try:
        restored = tuple(event_from_record(record) for record in reopened.replay("synthetic-casegraph"))
        assert restored == events
        assert reopened.replay("synthetic-other") == ()
        assert not ({e.case_id for e in restored} & {e.case_id for e in splits["retained"] + splits["drift"]})
        assert CaseGraphAdapter().replay(restored) == CaseGraphAdapter().replay(events)
    finally:
        reopened.close()


@pytest.mark.unit
def test_bridge_rejects_wrong_receipt() -> None:
    events = SyntheticCaseEventGenerator().generate()
    observation = event_to_record(events[0], scenario="synthetic-one")
    for feedback, scenario, source in [
        (events[1], "synthetic-two", observation),
        (events[3], "synthetic-one", observation),
        (events[1], "synthetic-one", None),
    ]:
        with pytest.raises(ValueError):
            event_to_record(feedback, scenario=scenario, observation=source)


@pytest.mark.integration
def test_report_lineage_passes_real_dispatcher_ingress(tmp_path: Path) -> None:
    from reef.artifact import InMemoryRepositoryBackend
    from reef.core.reports import ReportValidationError
    from reef.dispatcher import Dispatcher
    from reef.recipe import Recipe
    from reef.storage.sqlite import SQLiteScenarioStorage

    initial = tmp_path / "initial"
    initial.mkdir()
    dispatcher = Dispatcher(
        Recipe(),
        InMemoryRepositoryBackend.factory(initial, root=tmp_path / "artifacts"),
        scenario_storage=SQLiteScenarioStorage(tmp_path / "records"),
    )
    try:
        events = SyntheticCaseEventGenerator().generate()
        observation = dispatcher.accept_record(event_to_record(events[0], scenario="synthetic-ingress"))
        feedback = event_to_record(events[1], scenario="synthetic-ingress", observation=observation)
        # Phase 3 used report-to-report native references: storage accepted them,
        # but actual ingress correctly refuses to treat reports as inference.
        legacy = replace(feedback, references=(observation.agent_record_id,))
        with pytest.raises(ReportValidationError, match="existing inference"):
            dispatcher.accept_record(legacy)
        accepted = dispatcher.accept_record(feedback)
        assert accepted.references == ()
        assert accepted.payload["metadata"]["source_report_id"] == observation.agent_record_id
        assert event_from_record(accepted) == events[1]
    finally:
        dispatcher.close()
