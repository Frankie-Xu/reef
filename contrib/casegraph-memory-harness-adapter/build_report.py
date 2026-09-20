"""Rebuild a fixture-derived protocol report without model calls or fabricated scores."""

import argparse
import json
from pathlib import Path

from reef_casegraph_adapter import CaseEvent, CaseGraphAdapter, SyntheticCaseEventGenerator, fixture_digest


def build_report(fixture_path: Path) -> dict[str, object]:
    fixture = json.loads(fixture_path.read_text())
    events = tuple(CaseEvent.from_dict(item) for item in fixture["events"])
    if fixture_digest(events) != fixture["event_digest"]:
        raise ValueError("fixture digest mismatch")
    splits = SyntheticCaseEventGenerator().split(events)
    manifest = {name: sorted({event.case_id for event in items}) for name, items in splits.items()}
    if manifest != fixture["splits"]:
        raise ValueError("fixture split manifest mismatch")
    adapter = CaseGraphAdapter(fixture["artifact_version"])
    adapter.replay(splits["replay"])
    adapter.replay(splits["adapt"])
    candidate = adapter.candidate("protocol-v1")
    arms = {}
    for name, items in splits.items():
        arms[name] = {
            "event_count": len(items),
            "case_ids": manifest[name],
            "serialization_round_trip_rate": sum(CaseEvent.deserialize(e.serialize()) == e for e in items)
            / len(items),
            "synthetic_reference_coverage": sum(e.provenance.get("kind") == "synthetic" for e in items) / len(items),
            "fixture_cost_estimates": {
                key: sum(e.cost[key] for e in items)
                for key in ("compute_ms", "reviewer_seconds", "human_intervention_seconds")
            },
            "quality": None,
            "negative_transfer": None,
        }
    return {
        "schema": "casegraph-protocol-report/1",
        "synthetic": True,
        "fixture_digest": fixture_digest(events),
        "measurement_scope": "Serialization and synthetic provenance proxies; costs are fixture estimates, not timings.",
        "unmeasured": "Model/clinical quality, evidence correctness, review speed and negative transfer require evaluation.",
        "arms": arms,
        "candidate_state_digest": candidate.state_digest,
        "candidate_event_ids": list(candidate.event_ids),
        "retained_event_ids": [e.event_id for e in splits["retained"]],
        "selection_decision": "not_evaluated",
        "model_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("reef-caseevent-fixture.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("retained_evaluation_report.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_report(args.fixture), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
