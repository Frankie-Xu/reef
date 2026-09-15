"""The shipped gate scorer for task directory episodes: the Harbor verifier's reward from the terminus row."""

from __future__ import annotations

import pytest

from reef.harness.episodes.run import EpisodeResult
from reef.train.cordis_backend.strategies import resolve_episode_scorer, verifier_reward

TASK = "/tasks/openenv-00012-003-deduction"


def episode(*events: dict[str, object], exit_code: int = 0) -> EpisodeResult:
    return EpisodeResult(exit_code=exit_code, stdout="", stderr="", trajectory=tuple(events), residue=())


def verifier(reward: object, *, task: str = TASK, failed: bool = False) -> dict[str, object]:
    return {"type": "verifier", "task": task, "rewards": {"reward": reward}, "reward": reward, "failed": failed}


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (episode(verifier(1.0), {"type": "step"}), 1.0),
        (episode(verifier(0.25)), 0.25),
        (episode(verifier(0)), 0.0),
        (episode(verifier(None, failed=True)), 0.0),
        (episode(verifier(1.0), exit_code=1), 0.0),
    ],
)
def test_the_verifier_reward_is_the_score(result: EpisodeResult, expected: float) -> None:
    assert verifier_reward(TASK, result) == expected


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (episode({"type": "step"}), "expected one verifier record"),
        (episode(verifier(1.0), verifier(0.0)), "expected one verifier record"),
        (episode(verifier(1.0, task="/tasks/other")), "names '/tasks/other'"),
        (episode(verifier(float("nan"))), "must be a finite number"),
        (episode(verifier(True)), "must be a finite number"),
        (episode(verifier("1")), "must be a finite number"),
    ],
)
def test_a_record_that_cannot_be_scored_is_an_error(result: EpisodeResult, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        verifier_reward(TASK, result)


def test_the_scorer_resolves_from_its_dotted_reference() -> None:
    scorer = resolve_episode_scorer("reef.train.cordis_backend.strategies:verifier_reward")
    assert scorer(TASK, episode(verifier(0.5))) == 0.5
