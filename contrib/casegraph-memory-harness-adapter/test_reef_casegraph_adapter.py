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


@pytest.mark.unit
def test_invalid_provenance_is_rejected_on_deserialization() -> None:
    event = SyntheticCaseEventGenerator().generate()[0]
    payload = event.to_dict()
    payload["provenance"] = {**payload["provenance"], "kind": "clinical"}
    with pytest.raises(ValueError, match=r"provenance\.kind"):
        type(event).from_dict(payload)


@pytest.mark.unit
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


@pytest.mark.unit
def test_temporal_leakage_is_rejected() -> None:
    generator = SyntheticCaseEventGenerator()
    splits = generator.split(generator.generate())
    leaked = dict(splits)
    leaked["adapt"] = (
        leaked["adapt"][0].__class__(**{**leaked["adapt"][0].canonical(), "valid_from": "2026-01-01"}),
        *leaked["adapt"][1:],
    )
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


@pytest.mark.unit
def test_retained_regression_rolls_back_to_current_artifact() -> None:
    current = VersionedArtifact("artifact:v0:base", "v0", None, "base", (), {}, "selected")
    candidate = VersionedArtifact("artifact:v1:candidate", "v1", current.artifact_id, "candidate", (), {}, "candidate")
    assert select_or_rollback(current, candidate, retained_score=0.74, baseline_score=0.80) == current
    assert select_or_rollback(current, candidate, retained_score=0.80, baseline_score=0.80).status == "selected"


@pytest.mark.unit
def test_duplicate_event_is_idempotent() -> None:
    event = SyntheticCaseEventGenerator().generate(8)[0]
    adapter = CaseGraphAdapter()
    adapter.apply(event)
    digest = adapter.state_digest()
    adapter.apply(event)
    assert adapter.state_digest() == digest


@pytest.mark.unit
def test_artifact_version_mismatch_is_rejected() -> None:
    event = SyntheticCaseEventGenerator().generate()[0]
    with pytest.raises(ValueError, match="artifact version"):
        CaseGraphAdapter("casegraph-synthetic-v2").apply(event)


@pytest.mark.unit
@pytest.mark.parametrize("case_count", [8, 9, 16, 101])
def test_split_generator_is_complete_and_disjoint(case_count: int) -> None:
    from collections import Counter

    generator = SyntheticCaseEventGenerator()
    events = generator.generate(case_count)
    splits = generator.split(iter(events))
    assert Counter(e.event_id for arm in splits.values() for e in arm) == Counter(e.event_id for e in events)
    assert splits == generator.split(events)
    assert sum(len({e.case_id for e in arm}) for arm in splits.values()) == case_count


@pytest.mark.unit
@pytest.mark.parametrize("arm, observed_at", [("replay", "2026-01-10"), ("adapt", "2026-01-01")])
def test_observation_time_leakage_is_rejected(arm: str, observed_at: str) -> None:
    from dataclasses import replace

    generator = SyntheticCaseEventGenerator()
    splits = generator.split(generator.generate())
    splits[arm] = (replace(splits[arm][0], observed_at=observed_at), *splits[arm][1:])
    with pytest.raises(ValueError, match="temporal leakage"):
        validate_temporal_splits(splits)


@pytest.mark.unit
def test_invalid_observed_date_is_rejected() -> None:
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(SyntheticCaseEventGenerator().generate()[0], observed_at="yesterday")


@pytest.mark.unit
def test_explicit_case_overlap_is_rejected() -> None:
    generator = SyntheticCaseEventGenerator()
    splits = generator.split(generator.generate())
    splits["adapt"] = (*splits["adapt"], splits["replay"][0])
    with pytest.raises(ValueError, match="case overlap"):
        validate_temporal_splits(splits)
