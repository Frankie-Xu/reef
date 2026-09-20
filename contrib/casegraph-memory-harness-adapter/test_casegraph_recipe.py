from pathlib import Path

import pytest

from casegraph_recipe import CaseGraphRecipe, FrozenFixture, ingest_training, open_scenario

FIXTURE = Path(__file__).with_name("reef-caseevent-fixture.json")


@pytest.mark.integration
def test_real_recipe_publishes_only_after_commit_and_recovers(tmp_path: Path) -> None:
    recipe = CaseGraphRecipe(fixture=FrozenFixture.load(FIXTURE), work_dir=tmp_path / "candidates")
    scenario = open_scenario(tmp_path, recipe)
    try:
        old = scenario.current_artifact_ref()
        ingest_training(scenario, recipe.fixture)
        result = scenario.prepare_training_step()
        assert result is not None and result.artifact is not None
        assert scenario.current_artifact_ref() == scenario.repository.backend.current() == old
        scenario.commit(result)
        published = scenario.current_artifact_ref()
        assert published != old
        assert scenario.repository.backend.current() == published
        assert scenario.records.count(scenario.name) == 0
        assert len(scenario.store.history()) == 1
    finally:
        scenario.close()
    restored = open_scenario(tmp_path, recipe)
    try:
        assert restored.current_artifact_ref() == published
        assert restored.trainer.state == result.state
        assert restored.prepare_training_step() is None
    finally:
        restored.close()


@pytest.mark.integration
def test_rejected_candidate_keeps_published_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from casegraph_recipe import CaseGraphBackend, initial_policy

    recipe = CaseGraphRecipe(fixture=FrozenFixture.load(FIXTURE), work_dir=tmp_path / "candidates")
    scenario = open_scenario(tmp_path, recipe)
    try:
        old = scenario.current_artifact_ref()
        previous = dict(scenario.trainer.state)
        backend = scenario.trainer.candidate_backend
        assert isinstance(backend, CaseGraphBackend)

        def regressing_policy(events: tuple[str, ...], incumbent: dict[str, str]) -> dict[str, str]:
            return dict.fromkeys(initial_policy(), "episodic_memory")

        monkeypatch.setattr(backend, "learn_policy", regressing_policy)
        ingest_training(scenario, recipe.fixture)
        result = scenario.prepare_training_step()
        assert result is not None and result.artifact is None
        selection = result.metrics["selection"]
        assert selection["outcome"] == "reject"
        assert selection["evaluation"]["metrics"] == {"baseline_accuracy": 1.0, "candidate_accuracy": 0.5}
        assert not any(recipe.work_dir.iterdir())
        scenario.commit(result)
        assert scenario.current_artifact_ref() == old
        assert scenario.trainer.state == previous
        assert scenario.store.history()[0].metrics["selection"]["outcome"] == "reject"
    finally:
        scenario.close()
    restored = open_scenario(tmp_path, recipe)
    try:
        assert restored.current_artifact_ref() == old
        assert restored.trainer.state == previous
        assert restored.prepare_training_step() is None
    finally:
        restored.close()


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["evaluation", "journal"])
def test_failed_attempt_can_restart_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from casegraph_recipe import RetainedRoutingEvaluator
    from reef.storage.commit_log import CommitLogScenarioStore

    recipe = CaseGraphRecipe(fixture=FrozenFixture.load(FIXTURE), work_dir=tmp_path / "candidates")
    scenario = open_scenario(tmp_path, recipe)
    try:
        old = scenario.current_artifact_ref()
        previous = dict(scenario.trainer.state)
        ingest_training(scenario, recipe.fixture)
        with monkeypatch.context() as patch:
            if failure == "evaluation":

                def fail_evaluation(self, candidate):
                    raise RuntimeError("evaluation unavailable")

                patch.setattr(RetainedRoutingEvaluator, "evaluate", fail_evaluation)
                with pytest.raises(RuntimeError, match="evaluation unavailable"):
                    scenario.prepare_training_step()
                assert not any(recipe.work_dir.iterdir())
            else:
                result = scenario.prepare_training_step()
                assert result is not None
                assert isinstance(scenario.store, CommitLogScenarioStore)
                assert scenario.store.commit_log is not None

                def fail_append(record):
                    raise RuntimeError("journal unavailable")

                patch.setattr(scenario.store.commit_log, "append", fail_append)
                with pytest.raises(RuntimeError, match="journal unavailable"):
                    scenario.commit(result)
            assert scenario.current_artifact_ref() == scenario.repository.backend.current() == old
            assert scenario.trainer.state == previous
            assert scenario.store.history() == ()
            assert scenario.records.count(scenario.name) == len(recipe.fixture.training)
    finally:
        scenario.close()
    restored = open_scenario(tmp_path, recipe)
    try:
        assert restored.current_artifact_ref() == old
        assert restored.trainer.state == previous
        recovered_result = restored.prepare_training_step()
        assert recovered_result is not None
        restored.commit(recovered_result)
        assert restored.current_artifact_ref() != old
        assert restored.scenario_step == 1
    finally:
        restored.close()


@pytest.mark.integration
def test_restart_repairs_pointer_after_durable_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from reef.artifact import ArtifactPublicationError

    recipe = CaseGraphRecipe(fixture=FrozenFixture.load(FIXTURE), work_dir=tmp_path / "candidates")
    scenario = open_scenario(tmp_path, recipe)
    try:
        old = scenario.current_artifact_ref()
        ingest_training(scenario, recipe.fixture)
        result = scenario.prepare_training_step()
        assert result is not None

        def unavailable(*args, **kwargs):
            raise ArtifactPublicationError("pointer unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(scenario.repository.backend, "commit_release", unavailable)
            scenario.commit(result)
        published = scenario.current_artifact_ref()
        assert published != old
        assert scenario.repository.backend.current() == old
        assert scenario.commit_status["artifact_head_sync"]["state"] == "pending"
    finally:
        scenario.close()
    restored = open_scenario(tmp_path, recipe)
    try:
        assert restored.current_artifact_ref() == restored.repository.backend.current() == published
        assert len(restored.store.history()) == 1
        assert restored.prepare_training_step() is None
    finally:
        restored.close()


@pytest.mark.integration
def test_held_out_and_altered_identity_never_enter_training_storage(tmp_path: Path) -> None:
    from dataclasses import replace

    from reef_casegraph_adapter import CaseEvent
    from reef_record_bridge import event_to_record

    fixture = FrozenFixture.load(FIXTURE)
    recipe = CaseGraphRecipe(fixture=fixture, work_dir=tmp_path / "candidates")
    scenario = open_scenario(tmp_path, recipe)
    try:
        retained = CaseEvent.deserialize(fixture.retained[0])
        drift = CaseEvent.deserialize(fixture.drift[0])
        training = CaseEvent.deserialize(fixture.training[0])
        for event in (retained, drift, replace(training, confidence=0.5)):
            with pytest.raises(ValueError, match="held-out or altered"):
                ingest_training(scenario, fixture, (training, event))
            assert scenario.records.count(scenario.name) == 0
            with pytest.raises(ValueError, match="frozen training"):
                scenario.trainer.processor.ingest(event_to_record(event, scenario=scenario.name))
        forged = replace(fixture, training=fixture.retained)
        with pytest.raises(ValueError, match="does not match"):
            ingest_training(scenario, forged)
        receipts = ingest_training(scenario, fixture)
        assert len(receipts) == len(fixture.training)
        assert all(not receipt.references for receipt in receipts)
        stored = scenario.records.replay(scenario.name)
        assert {record.agent_record_id for record in stored} == {record.agent_record_id for record in receipts}
        assert {record.payload["metadata"]["event_json"] for record in stored} == set(fixture.training)
        assert set(fixture.training).isdisjoint(fixture.retained + fixture.drift)
    finally:
        scenario.close()


@pytest.mark.unit
def test_fixture_is_frozen_after_loading(tmp_path: Path) -> None:
    import json

    from reef_casegraph_adapter import CaseEvent

    path = tmp_path / "fixture.json"
    path.write_bytes(FIXTURE.read_bytes())
    fixture = FrozenFixture.load(path)
    before = fixture.retained
    event = CaseEvent.deserialize(before[0])
    event.cost["compute_ms"] = 999
    path.write_text(json.dumps({"events": []}))
    assert fixture.retained == before
    assert CaseEvent.deserialize(fixture.retained[0]).cost["compute_ms"] == 1


@pytest.mark.integration
def test_registered_fixture_cannot_change_even_before_first_commit(tmp_path: Path) -> None:
    from dataclasses import replace

    from reef_casegraph_adapter import CaseEvent

    fixture = FrozenFixture.load(FIXTURE)
    recipe = CaseGraphRecipe(fixture=fixture, work_dir=tmp_path / "candidates")
    scenario = open_scenario(tmp_path, recipe)
    original = scenario.current_artifact_ref()
    scenario.close()
    changed = replace(CaseEvent.deserialize(fixture.retained[0]), confidence=0.5)
    other_fixture = replace(fixture, retained=(changed.serialize(), *fixture.retained[1:]))
    with pytest.raises(ValueError, match="registered scenario"):
        open_scenario(tmp_path, replace(recipe, fixture=other_fixture))
    restored = open_scenario(tmp_path, recipe)
    try:
        assert restored.current_artifact_ref() == original
        assert restored.store.history() == ()
    finally:
        restored.close()
