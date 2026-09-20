import json
from pathlib import Path

import pytest

from build_report import build_report

HERE = Path(__file__).parent


@pytest.mark.unit
def test_committed_report_is_reproducible_and_retained_is_excluded() -> None:
    fixture = HERE / "reef-caseevent-fixture.json"
    before = fixture.read_bytes()
    result = build_report(fixture)
    assert result == build_report(fixture)
    assert result == json.loads((HERE / "retained_evaluation_report.json").read_text())
    assert fixture.read_bytes() == before
    assert set(result["candidate_event_ids"]).isdisjoint(result["retained_event_ids"])
    assert result["arms"]["replay"]["fixture_cost_estimates"]["reviewer_seconds"] == 12
    assert all(arm["quality"] is None and arm["negative_transfer"] is None for arm in result["arms"].values())
    assert result["selection_decision"] == "not_evaluated"


@pytest.mark.unit
@pytest.mark.parametrize("field", ["event_digest", "splits"])
def test_report_rejects_stale_fixture_metadata(tmp_path: Path, field: str) -> None:
    data = json.loads((HERE / "reef-caseevent-fixture.json").read_text())
    data[field] = "invalid"
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mismatch"):
        build_report(fixture)
