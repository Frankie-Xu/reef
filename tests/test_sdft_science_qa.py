"""The Science Q&A example: the training order, the scorer's answer rule, and the demonstration-SFT control."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


# --- the SFT control ----------------------------------------------------------


@pytest.fixture
def baseline_sft():
    """The control's module, with its dotted-reference registrations undone afterwards."""
    import importlib

    from reef.train.algos.registry import OBJECTIVES, unregister_objective
    from reef.train.slime_backend.loss_families import LOSS_FAMILIES, unregister_loss_family

    module = importlib.import_module("recipes.sdft.examples.science_qa.baseline_sft")
    yield module
    if module.LOSS_FAMILY in LOSS_FAMILIES.names:
        unregister_loss_family(module.LOSS_FAMILY)
    if module.LOSS_FAMILY in OBJECTIVES.names:
        unregister_objective(module.LOSS_FAMILY)


class CountingDemonstrationTokenizer:
    """Prompt ids one per message; response ids one per ten characters of the demonstration, then an end token."""

    def sequence_ids(self, messages, tools, demonstration):
        return [100 + index for index in range(len(messages))], [*range(200, 200 + len(demonstration) // 10), 999]


def _science_inference(agent_record_id: str):
    from reef.core import AgentRecord, RequestType

    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.INFERENCE,
        payload={
            "messages": [{"role": "system", "content": "Answer."}, {"role": "user", "content": "Which acid?"}],
            "tokens": [5, 6, 7, 1, 2, 3],
            "loss_mask": [1, 1, 1],
            "rollout_log_probs": [-0.1, -0.2, -0.3],
        },
        agent_record_id=agent_record_id,
    )


def test_sft_control_trains_the_demonstration_as_the_assistant_turn(baseline_sft) -> None:
    from reef.core import AgentRecord, RequestType
    from reef.core.reports import TeacherContextReport
    from reef.train import ProcessorContext
    from reef.train.algos import StepScheduling
    from reef.train.slime_backend.data_builder import to_slime_rollout_data
    from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step

    tokenizer = CountingDemonstrationTokenizer()
    processor = baseline_sft.DemonstrationSFTProcessor(
        ProcessorContext("science", {"batch_size": 1}, TeacherContextReport), tokenizer
    )
    demonstration = "<reasoning>\nthirty characters of reasoning\n</reasoning>\n<answer>\nB\n</answer>"
    processor.ingest(_science_inference("i1"))
    processor.ingest(
        AgentRecord.create(
            scenario="science",
            request_type=RequestType.REPORT,
            payload=TeacherContextReport(context=demonstration).to_dict(references=("i1",)),
            agent_record_id="r1",
            references=("i1",),
        )
    )
    batch = processor.build_batch()

    (sample,) = batch.items
    prompt_ids, response_ids = tokenizer.sequence_ids(
        _science_inference("i1").payload["messages"], None, demonstration
    )
    assert prompt_ids == [100, 101]
    assert list(sample.training["tokens"]) == [*prompt_ids, *response_ids]
    assert list(sample.training["loss_mask"]) == [1] * len(response_ids)
    assert list(sample.training["rollout_log_probs"]) == []
    assert list(sample.training["runtime_load_spans"]) == []

    spec = baseline_sft.DemonstrationSFTRecipe.training_spec()
    prepared = prepare_slime_step(batch, spec.objective, {}, StepScheduling(unit="sample"))
    payload = prepared.payload
    assert payload is not None
    assert payload["loss"] == spec.loss_family == f"{baseline_sft.MODULE}:DEMONSTRATION_SFT"
    data = to_slime_rollout_data({key: value for key, value in payload.items() if key != "source_rows"})
    assert data["loss"] == baseline_sft.LOSS_FAMILY
    assert data["tokens"] == [[*prompt_ids, *response_ids]]
    assert data["loss_masks"] == [[1] * len(response_ids)]
    assert "rollout_log_probs" not in data
    with pytest.raises(ValueError, match="must omit advantages"):
        to_slime_rollout_data({**payload, "advantages": [1.0]})


def test_sft_control_recipe_binds_the_shared_report_and_its_own_family(baseline_sft) -> None:
    from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

    from reef.core.reports import TeacherContextReport
    from reef.recipe.registry import build_recipe
    from reef.train.slime_backend.loss_families import resolve_loss_family

    recipe = build_recipe(
        f"{baseline_sft.MODULE}:DemonstrationSFTRecipe",
        {},
        {"data": {"tokenizer_path": "/models/served", "batch_size": 32}},
        **runtime_bindings(StubTrainingRuntime()),
    )
    assert recipe.report_type is TeacherContextReport
    assert recipe.batch_size == 32
    family = resolve_loss_family(baseline_sft.DemonstrationSFTRecipe.training_spec().loss_family)
    assert (family.loss_family, family.loss_type, family.advantages) == (
        baseline_sft.LOSS_FAMILY,
        "sft_loss",
        "forbidden",
    )
    family.validate_backend_args(SimpleNamespace(loss_type="sft_loss", use_rollout_logprobs=False))
