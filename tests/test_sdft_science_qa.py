"""The Science Q&A example's host-side pieces: the training order and the scorer's answer rule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "recipes" / "sdft" / "examples" / "science_qa"


@pytest.fixture(scope="module")
def science():
    """``science.py`` loaded from the example directory, the way run.py imports it."""
    spec = importlib.util.spec_from_file_location("science", EXAMPLE / "science.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["science"] = module
    spec.loader.exec_module(module)
    return module


def test_wave_schedule_covers_each_epoch_once_and_drops_only_the_tail(science) -> None:
    schedule = science.wave_schedule(2674, 2, 32, 42)

    assert len(schedule) == 167
    assert all(len(step) == 32 for step in schedule)
    flat = [index for step in schedule for index in step]
    # Steps run across the epoch boundary; the first 2674 prompts are one shuffle of the epoch.
    assert sorted(flat[:2674]) == list(range(2674))
    assert sorted(flat[2674:] + [index for index in range(2674) if index not in set(flat[2674:])]) == list(range(2674))
    assert len(set(flat[2674:])) == 167 * 32 - 2674
    # The order is fixed by the seed and differs between epochs.
    assert schedule == science.wave_schedule(2674, 2, 32, 42)
    assert flat[:2674] != flat[2674:] + flat[2674 : 2674 + 4]
    assert science.wave_schedule(2674, 2, 32, 7) != schedule


def test_extract_answer_follows_the_reference_scorer(science) -> None:
    text = "<reasoning>\nsome steps\n</reasoning>\n<answer>\nB\n</answer>"
    assert science.extract_answer(text) == "B"
    # The last tag wins, and text without a tag is scored as is.
    assert science.extract_answer("<answer>A</answer> ... <answer> C </answer>") == "C"
    assert science.extract_answer("D") == "D"
