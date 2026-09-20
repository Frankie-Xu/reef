import json
from pathlib import Path

import pytest

from run_lifecycle import run_lifecycle


@pytest.mark.integration
def test_lifecycle_report_is_reproducible_from_real_runs(tmp_path: Path) -> None:
    first = run_lifecycle(tmp_path / "first")
    second = run_lifecycle(tmp_path / "second")
    expected = json.loads(Path(__file__).with_name("lifecycle_report.json").read_text())
    assert first == second == expected
    assert first["selection"]["outcome"] == "select"
    assert first["selection"]["evaluation"]["metrics"]["candidate_accuracy"] == 1.0
    assert first["restart_recovered_without_retraining"]
    with pytest.raises(ValueError, match="fresh work directory"):
        run_lifecycle(tmp_path / "first")
