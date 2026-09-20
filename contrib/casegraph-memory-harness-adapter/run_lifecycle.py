"""Run the real local Reef lifecycle and emit deterministic synthetic task results."""

import argparse
import json
from pathlib import Path

from casegraph_recipe import (
    CaseGraphRecipe,
    FrozenFixture,
    ingest_training,
    initial_policy,
    open_scenario,
    read_policy,
    task_results,
)
from reef_casegraph_adapter import digest


def run_lifecycle(work_dir: Path) -> dict[str, object]:
    if work_dir.exists() and any(work_dir.iterdir()):
        raise ValueError("use a fresh work directory for a reproducible run")
    fixture = FrozenFixture.load(Path(__file__).with_name("reef-caseevent-fixture.json"))
    recipe = CaseGraphRecipe(fixture=fixture, work_dir=work_dir / "candidates")
    scenario = open_scenario(work_dir, recipe)
    try:
        old = scenario.current_artifact_ref()
        receipts = ingest_training(scenario, fixture)
        prepared = scenario.prepare_training_step()
        if prepared is None or prepared.artifact is None:
            raise RuntimeError("expected a selected candidate artifact")
        if scenario.current_artifact_ref() != old or scenario.repository.backend.current() != old:
            raise RuntimeError("preparation changed the published head")
        scenario.commit(prepared)
        published = scenario.current_artifact_ref()
        if published == old or published != scenario.repository.backend.current():
            raise RuntimeError("commit did not publish the selected artifact")
        artifact = scenario.artifact_for_version(published.release_id).materialize()
        if artifact.local_path is None:
            raise RuntimeError("published artifact has no materialized files")
        policy = read_policy(json.loads((artifact.local_path / "policy.json").read_text()))
        if policy != scenario.trainer.state["policy"]:
            raise RuntimeError("published policy differs from durable state")
        training_ids = sorted(record.agent_record_id for record in receipts)
        history = scenario.store.history()
        if len(history) != 1 or set(history[0].consumed_ids) != set(training_ids):
            raise RuntimeError("commit did not consume exactly the frozen training records")
        audit = scenario.records.audit_page(scenario.name)
        if {row.item.agent_record_id for row in audit} != set(training_ids):
            raise RuntimeError("training audit contains unexpected records")
        report: dict[str, object] = {
            "schema": "casegraph-lifecycle-report/1",
            "synthetic": True,
            "scope": "Deterministic corrective-surface routing; no clinical/model quality claim.",
            "sample_warning": "Retained has one case and two correlated event tasks; diagnostic only.",
            "fixture_digest": fixture.checksum,
            "retained_digest": digest(fixture.retained),
            "training_record_ids": training_ids,
            "selection": prepared.metrics["selection"],
            "policy": policy,
            "model_calls": 0,
            "measured_runtime_ms": None,
            "actual_interfaces": [
                "ScenarioFactory.load_or_create",
                "RecordStore.append",
                "Scenario.prepare_training_step",
                "CandidateBackend.prepare_step",
                "CandidateEvaluationPlugin.evaluate/decide",
                "CandidateBackend.settle_step",
                "Scenario.commit",
            ],
            "arms": {
                name: {
                    "baseline_tasks": task_results(initial_policy(), events),
                    "candidate_tasks": task_results(policy, events),
                }
                for name, events in (
                    ("training", fixture.training),
                    ("retained", fixture.retained),
                    ("drift", fixture.drift),
                )
            },
            "verified": {
                "head_unchanged_before_commit": True,
                "published_after_commit": True,
                "held_out_absent_from_training_audit": True,
            },
        }
    finally:
        scenario.close()
    restored = open_scenario(work_dir, recipe)
    try:
        if restored.current_artifact_ref() != published or restored.repository.backend.current() != published:
            raise RuntimeError("restart did not recover the committed head")
        if read_policy(restored.trainer.state["policy"]) != policy or restored.prepare_training_step() is not None:
            raise RuntimeError("restart changed state or retrained consumed records")
    finally:
        restored.close()
    report["restart_recovered_without_retraining"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_lifecycle(args.work_dir)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
