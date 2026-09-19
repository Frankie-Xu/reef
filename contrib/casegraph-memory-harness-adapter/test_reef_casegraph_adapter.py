from __future__ import annotations

import pytest

from reef_casegraph_adapter import (
    FEEDBACK_TYPES,
    CaseGraphAdapter,
    SyntheticCaseEventGenerator,
    VersionedArtifact,
    fixture_digest,
    select_or_rollback,
    validate_temporal_splits,
)


@pytest.mark.unit
def test_generator_is_deterministic_with_fixed_seed() -> None:
    first = SyntheticCaseEventGenerator(seed=7).generate()
    second = SyntheticCaseEventGenerator(seed=7).generate()
    assert fixture_digest(first) == fixture_digest(second)


@pytest.mark.unit
def test_case_event_round_trip_preserves_stable_event_id() -> None:
    event = SyntheticCaseEventGenerator().generate()[0]
    restored = type(event).deserialize(event.serialize())
    assert restored == event
    assert restored.event_id == event.event_id


@pytest.mark.regression
def test_invalid_provenance_is_rejected_on_deserialization() -> None:
    event = SyntheticCaseEventGenerator().generate()[0]
    payload = event.to_dict()
    payload["provenance"] = {**payload["provenance"], "kind": "clinical"}
    with pytest.raises(ValueError, match="provenance.kind"):
        type(event).from_dict(payload)


@pytest.mark.regression
def test_tampered_event_id_is_rejected() -> None:
    event = SyntheticCaseEventGenerator().generate()[0]
    payload = event.to_dict()
    payload["event_id"] = "0" * 64
    with pytest.raises(ValueError, match="event_id"):
        type(event).from_dict(payload)


@pytest.mark.integration
def test_splits_are_temporal_and_case_disjoint() -> None:
    generator = SyntheticCaseEventGenerator()
    splits = generator.split(generator.generate())
    assert set(splits) == {"replay", "adapt", "retained", "drift"}
    case_sets = {name: {event.case_id for event in items} for name, items in splits.items()}
    assert sum(len(cases) for cases in case_sets.values()) == 8
    assert not (case_sets["replay"] & case_sets["adapt"])
    assert not (case_sets["adapt"] & case_sets["retained"])
    assert max(event.valid_from for event in splits["replay"]) < min(event.valid_from for event in splits["adapt"])


@pytest.mark.regression
def test_temporal_leakage_is_rejected() -> None:
    generator = SyntheticCaseEventGenerator()
    splits = generator.split(generator.generate())
    leaked = dict(splits)
    leaked["adapt"] = (leaked["adapt"][0].__class__(**{**leaked["adapt"][0].canonical(), "valid_from": "2026-01-01"}),) + leaked["adapt"][1:]
    with pytest.raises(ValueError, match="temporal leakage"):
        validate_temporal_splits(leaked)


@pytest.mark.integration
def test_feedback_credit_assignment_covers_all_four_failure_modes() -> None:
    events = SyntheticCaseEventGenerator().generate()
    assert {event.feedback_type for event in events if event.feedback_type} >= set(FEEDBACK_TYPES)
    adapter = CaseGraphAdapter()
    adapter.replay(events)
    assert adapter.state.harness_failures == {
        "execution_strategy_failure": 2,
        "verifier_error": 2,
    }
    memory_kinds = {entry["kind"] for entries in adapter.state.memory_nodes.values() for entry in entries}
    assert {"fact_error", "retrieval_omission"} <= memory_kinds


@pytest.mark.integration
def test_replay_and_candidate_identity_are_stable() -> None:
    events = SyntheticCaseEventGenerator().generate()
    first = CaseGraphAdapter()
    second = CaseGraphAdapter()
    assert first.replay(events[:8]) == second.replay(events[:8])
    assert first.candidate("v1") == second.candidate("v1")


@pytest.mark.regression
def test_retained_regression_rolls_back_to_current_artifact() -> None:
    current = VersionedArtifact("artifact:v0:base", "v0", None, "base", (), {}, "selected")
    candidate = VersionedArtifact("artifact:v1:candidate", "v1", current.artifact_id, "candidate", (), {}, "candidate")
    assert select_or_rollback(current, candidate, retained_score=0.74, baseline_score=0.80) == current
    assert select_or_rollback(current, candidate, retained_score=0.80, baseline_score=0.80).status == "selected"


@pytest.mark.regression
def test_duplicate_event_is_idempotent() -> None:
    event = SyntheticCaseEventGenerator().generate(8)[0]
    adapter = CaseGraphAdapter()
    adapter.apply(event)
    digest = adapter.state_digest()
    adapter.apply(event)
    assert adapter.state_digest() == digest


@pytest.mark.regression
def test_artifact_version_mismatch_is_rejected() -> None:
    event = SyntheticCaseEventGenerator().generate()[0]
    with pytest.raises(ValueError, match="artifact version"):
        CaseGraphAdapter("casegraph-synthetic-v2").apply(event)
