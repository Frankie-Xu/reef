"""Reef-side SDFT pipeline: report contract, teacher prompt, processor, recipe and Slime wire payload.

Everything here is torch/ray free so it runs in the minimal CI gate; the
tensor kernels are pinned to the pure-Python reference in
``test_sdft_parity.py``. The teacher prompt tokenizer is a fake that counts
tokens deterministically, so no model files are needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

from recipes.sdft import SdftObjective, SDFTProcessor, SDFTRecipe, TeacherContextReport
from recipes.sdft.slime import SdftSettings
from recipes.sdft.teacher_prompt import (
    DEFAULT_CONTEXT_TEMPLATE,
    AppendContextBuilder,
    TeacherPromptBuilder,
    TeacherPromptTokenizer,
    TeacherRequest,
)
from reef.artifact.artifact import LiveWeightArtifactRef
from reef.core import AgentRecord, RequestType
from reef.core.reports import ReportValidationError
from reef.core.trajectories import source_record_id
from reef.recipe.checkpoint_strategy import EveryNVersions
from reef.recipe.errors import RecipeConfigError
from reef.recipe.registry import build_recipe, recipe_class_for
from reef.train import ProcessorContext
from reef.train.algos import StepScheduling
from reef.train.slime_backend.data_builder import to_slime_rollout_data
from reef.train.slime_backend.loss_families import resolve_loss_family
from reef.train.slime_backend.reef_adapters.preparation import prepare_slime_step
from reef.train.types import TrainingBatch

TOKENIZER_PATH = "/models/served"
STUDENT_TOKENS = (5, 6, 7, 1, 2, 3)  # three prompt ids, three response ids
STUDENT_LOSS_MASK = (1, 1, 1)
STUDENT_LOG_PROBS = (-0.1, -0.2, -0.3)


class CountingTokenizer(TeacherPromptTokenizer):
    """One token per message plus one per ten characters of text; records what it rendered."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[Mapping[str, Any]], Sequence[Any] | None]] = []

    def prompt_token_ids(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None) -> list[int]:
        self.calls.append((list(messages), tools))
        text = "".join(str(message.get("content") or "") for message in messages)
        return [100 + index for index in range(len(messages) + len(text) // 10)]


class PrefixBuilder(TeacherPromptBuilder):
    """A deployment's own composition: the context as a system message, and no tools for the teacher."""

    def __init__(self, label: str) -> None:
        self.label = label

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PrefixBuilder:
        return cls(str(config.get("context_template", "demo")))

    def build(
        self, request_messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None, context: str
    ) -> TeacherRequest:
        return TeacherRequest([{"role": "system", "content": f"{self.label}: {context}"}, *request_messages], None)


def _inference(agent_record_id: str, *, messages: list[dict[str, Any]] | None = None) -> AgentRecord:
    payload: dict[str, Any] = {
        "messages": messages or [{"role": "user", "content": "What is the boiling point of water?"}],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tokens": list(STUDENT_TOKENS),
        "loss_mask": list(STUDENT_LOSS_MASK),
        "rollout_log_probs": list(STUDENT_LOG_PROBS),
    }
    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.INFERENCE,
        payload=payload,
        agent_record_id=agent_record_id,
        artifact_ref=LiveWeightArtifactRef(
            content_id="science", release_id="slime-v3", parent_release_id=None, runtime_load_id="slime-v3"
        ),
    )


def _report(agent_record_id: str, references: tuple[str, ...], context: str = "100 degrees Celsius.") -> AgentRecord:
    body = TeacherContextReport(context=context).to_dict(references=references)
    return AgentRecord.create(
        scenario="science",
        request_type=RequestType.REPORT,
        payload=body,
        agent_record_id=agent_record_id,
        references=references,
    )


def _processor(tokenizer: CountingTokenizer | None = None, **config: Any) -> SDFTProcessor:
    return SDFTProcessor(ProcessorContext("science", {"batch_size": 1, **config}, TeacherContextReport), tokenizer)


# --- report contract --------------------------------------------------------


@pytest.mark.unit
def test_report_carries_the_context_and_an_optional_score() -> None:
    report = TeacherContextReport.from_dict({"metadata": {"context": "demo"}})
    assert report == TeacherContextReport(context="demo")
    assert report.to_dict(references=("i1",)) == {"metadata": {"context": "demo"}, "references": ["i1"]}

    scored = TeacherContextReport.from_dict({"score": 0.5, "metadata": {"context": "demo"}})
    assert scored.score == 0.5


@pytest.mark.unit
@pytest.mark.parametrize("payload", [{"metadata": {}}, {"metadata": {"context": "   "}}, {"metadata": {"context": 3}}])
def test_report_rejects_a_missing_or_empty_context(payload: dict[str, Any]) -> None:
    with pytest.raises(ReportValidationError, match="context"):
        TeacherContextReport.from_dict(payload)


# --- teacher prompt ---------------------------------------------------------


@pytest.mark.unit
def test_default_builder_adds_the_demonstration_to_the_final_user_message() -> None:
    request = [
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Question?"}]},
    ]
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    teacher = AppendContextBuilder().build(request, tools, "The answer.")

    assert teacher.messages == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Question?\n\n" + DEFAULT_CONTEXT_TEMPLATE.replace("{context}", "The answer.")},
    ]
    assert teacher.tools == tools
    # The request is left as recorded.
    assert request[1]["content"] == [{"type": "text", "text": "Question?"}]


@pytest.mark.unit
def test_default_builder_appends_a_user_message_after_a_tool_result() -> None:
    request = [
        {"role": "user", "content": "Run it."},
        {"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]},
        {"role": "tool", "content": "a.txt"},
    ]

    teacher = AppendContextBuilder("Demonstration: {context}").build(request, None, "call bash with cat a.txt")

    assert teacher.messages[-1] == {"role": "user", "content": "Demonstration: call bash with cat a.txt"}
    assert teacher.messages[:3] == [
        {"role": "user", "content": "Run it."},
        {"role": "assistant", "tool_calls": [{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}]},
        {"role": "tool", "content": "a.txt"},
    ]
    assert teacher.tools is None


@pytest.mark.unit
def test_default_builder_requires_the_context_placeholder() -> None:
    with pytest.raises(ValueError, match=r"\{context\}"):
        AppendContextBuilder("no placeholder")


# --- processor --------------------------------------------------------------


@pytest.mark.unit
def test_processor_emits_one_sample_with_the_teacher_sequence() -> None:
    tokenizer = CountingTokenizer()
    processor = _processor(tokenizer)
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", ("i1",)))

    batch = processor.build_batch()

    assert isinstance(batch, TrainingBatch)
    assert len(batch.items) == 1
    sample = batch.items[0]
    assert source_record_id(sample) == "i1"
    prompt_ids = tokenizer.prompt_token_ids(*tokenizer.calls[0])
    # The teacher sequence is the rendered teacher prompt plus the response ids verbatim.
    assert list(sample.training["teacher_tokens"]) == [*prompt_ids, 1, 2, 3]
    assert list(sample.training["tokens"]) == list(STUDENT_TOKENS)
    rendered, tools = tokenizer.calls[0]
    assert tools == [{"type": "function", "function": {"name": "lookup"}}]
    assert rendered[-1]["role"] == "user"
    assert rendered[-1]["content"].startswith("What is the boiling point of water?\n\n")
    assert "100 degrees Celsius." in rendered[-1]["content"]
    assert processor.operational_metrics()["teacher_overflow_reports"] == 0


@pytest.mark.unit
def test_processor_skips_and_counts_a_teacher_sequence_over_the_window() -> None:
    tokenizer = CountingTokenizer()
    long_request = _inference("i1")
    short_request = _inference("i2", messages=[{"role": "user", "content": "q"}])
    # The window admits the short request's teacher sequence and not the long one's.
    short_teacher = AppendContextBuilder().build(short_request.payload["messages"], None, "ok")
    window = len(tokenizer.prompt_token_ids(short_teacher.messages, None)) + 3
    processor = _processor(tokenizer, max_teacher_tokens=window)
    processor.ingest(long_request)
    processor.ingest(_report("r1", ("i1",)))

    assert not processor.ready()
    assert processor.operational_metrics()["teacher_overflow_reports"] == 1
    # The report and its inference are released for compaction.
    assert {"r1", "i1"} <= processor.retention_decision().releasable_agent_record_ids

    # A later report that fits still trains.
    processor.ingest(short_request)
    processor.ingest(_report("r2", ("i2",), context="ok"))
    assert len(processor.build_batch().items) == 1
    assert processor.operational_metrics()["teacher_overflow_reports"] == 1


@pytest.mark.unit
def test_processor_uses_the_configured_teacher_prompt_builder() -> None:
    tokenizer = CountingTokenizer()
    processor = _processor(
        tokenizer,
        teacher_prompt_builder=f"{__name__}:PrefixBuilder",
        context_template="Expert move",
    )
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", ("i1",)))

    batch = processor.build_batch()

    rendered, tools = tokenizer.calls[0]
    assert rendered[0] == {"role": "system", "content": "Expert move: 100 degrees Celsius."}
    assert rendered[1:] == _inference("i1").payload["messages"]
    assert tools is None
    prompt_ids = tokenizer.prompt_token_ids(rendered, tools)
    assert list(batch.items[0].training["teacher_tokens"]) == [*prompt_ids, 1, 2, 3]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("reference", "error", "message"),
    [
        ("no-colon", ValueError, "package.module:Builder"),
        ("no.such.module:Builder", ValueError, "cannot import"),
        (f"{__name__}:CountingTokenizer", TypeError, "TeacherPromptBuilder"),
    ],
)
def test_processor_rejects_a_bad_teacher_prompt_builder_reference(
    reference: str, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _processor(CountingTokenizer(), teacher_prompt_builder=reference)


@pytest.mark.unit
def test_processor_requires_one_recorded_request_per_report() -> None:
    processor = _processor(CountingTokenizer(), accept_multi_turn_policy_samples=True)
    processor.ingest(_inference("i1"))
    processor.ingest(_inference("i2"))

    with pytest.raises(ValueError, match="one recorded request per report"):
        processor.ingest(_report("r1", ("i1", "i2")))


@pytest.mark.unit
def test_processor_requires_the_tokenizer_path_and_a_valid_template() -> None:
    with pytest.raises(ValueError, match="tokenizer_path"):
        _processor()
    with pytest.raises(ValueError, match=r"\{context\}"):
        _processor(CountingTokenizer(), context_template="no placeholder")
    with pytest.raises(ValueError, match="max_teacher_tokens"):
        _processor(CountingTokenizer(), max_teacher_tokens=-1)


# --- recipe -----------------------------------------------------------------


@pytest.mark.unit
def test_sdft_recipe_resolves_by_dotted_reference() -> None:
    reference = "recipes.sdft.recipe:SDFTRecipe"
    assert recipe_class_for(reference) is SDFTRecipe

    recipe = build_recipe(
        reference, {}, {"data": {"tokenizer_path": TOKENIZER_PATH}}, **runtime_bindings(StubTrainingRuntime())
    )

    assert isinstance(recipe, SDFTRecipe)
    assert recipe.name == "sdft"
    assert recipe.report_type is TeacherContextReport
    assert recipe.batch_size == 1
    assert recipe.max_teacher_tokens == 0
    assert recipe.context_template == DEFAULT_CONTEXT_TEMPLATE
    assert recipe.checkpoint_strategy == EveryNVersions(1)
    spec = SDFTRecipe.training_spec()
    assert (spec.objective, spec.processor, spec.loss_family) == ("sdft", SDFTProcessor, "sdft")
    assert spec.scheduling == StepScheduling(unit="sample")


@pytest.mark.unit
def test_sdft_recipe_reads_reef_side_config_and_hands_it_to_the_processor() -> None:
    recipe = SDFTRecipe.from_environment(
        {},
        config={
            "data": {
                "batch_size": 4,
                "tokenizer_path": TOKENIZER_PATH,
                "max_teacher_tokens": 24000,
                "context_template": "Example: {context}",
            },
            "artifact": {"checkpoint_every_n_versions": 4},
        },
        **runtime_bindings(StubTrainingRuntime()),
    )

    assert recipe.batch_size == 4
    assert recipe.checkpoint_strategy == EveryNVersions(4)
    assert recipe.processor_config() == {
        "batch_size": 4,
        "tokenizer_path": TOKENIZER_PATH,
        "max_teacher_tokens": 24000,
        "context_template": "Example: {context}",
        "teacher_prompt_builder": "",
    }


@pytest.mark.unit
def test_sdft_recipe_validates_the_teacher_prompt_composition_at_construction() -> None:
    bindings = runtime_bindings(StubTrainingRuntime())
    recipe = SDFTRecipe.from_environment(
        {},
        config={"data": {"tokenizer_path": TOKENIZER_PATH, "teacher_prompt_builder": f"{__name__}:PrefixBuilder"}},
        **bindings,
    )
    assert recipe.teacher_prompt_builder == f"{__name__}:PrefixBuilder"

    with pytest.raises(RecipeConfigError, match="cannot import teacher_prompt_builder"):
        SDFTRecipe.from_environment(
            {}, config={"data": {"tokenizer_path": TOKENIZER_PATH, "teacher_prompt_builder": "no.such:B"}}, **bindings
        )
    with pytest.raises(RecipeConfigError, match=r"context_template must contain \{context\}"):
        SDFTRecipe.from_environment(
            {}, config={"data": {"tokenizer_path": TOKENIZER_PATH, "context_template": "bare"}}, **bindings
        )


@pytest.mark.unit
def test_sdft_recipe_rejects_a_missing_tokenizer_and_backend_objective_config() -> None:
    with pytest.raises(RecipeConfigError, match="tokenizer_path"):
        SDFTRecipe.from_environment({}, config={}, **runtime_bindings(StubTrainingRuntime()))
    with pytest.raises(RecipeConfigError, match=r"training\.options"):
        SDFTRecipe.from_environment(
            {},
            config={"data": {"tokenizer_path": TOKENIZER_PATH}, "optimization": {"kl_direction": "reverse"}},
            **runtime_bindings(StubTrainingRuntime()),
        )


# --- objective and Slime wire payload ----------------------------------------


@pytest.mark.unit
def test_objective_trains_every_batch_and_refuses_multiple_epochs() -> None:
    objective = SdftObjective()
    tokenizer = CountingTokenizer()
    processor = _processor(tokenizer)
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", ("i1",)))
    batch = processor.build_batch()

    signal = objective.prepare(batch, {"steps": 2})

    assert (signal.action, signal.next_algorithm_state, signal.advantages) == ("train", {"steps": 3}, None)
    assert signal.metrics == {"steps": 3, "samples": 1}
    with pytest.raises(ValueError, match="epochs"):
        objective.validate_scheduling(StepScheduling(unit="sample", epochs=2))


@pytest.mark.unit
def test_slime_payload_carries_the_teacher_sequence_beside_the_policy_row() -> None:
    tokenizer = CountingTokenizer()
    processor = _processor(tokenizer)
    processor.ingest(_inference("i1"))
    processor.ingest(_report("r1", ("i1",)))
    batch = processor.build_batch()

    prepared = prepare_slime_step(batch, "sdft", {}, StepScheduling(unit="sample"))
    payload = prepared.payload
    assert payload is not None
    assert payload["loss"] == "sdft"
    assert "advantages" not in payload
    (row,) = payload["samples"]
    prompt_ids = tokenizer.prompt_token_ids(*tokenizer.calls[0])
    assert row == [
        "i1",
        list(STUDENT_TOKENS),
        list(STUDENT_LOSS_MASK),
        list(STUDENT_LOG_PROBS),
        0.0,
        [*prompt_ids, 1, 2, 3],
    ]

    data = to_slime_rollout_data({key: value for key, value in payload.items() if key != "source_rows"})
    assert data["loss"] == "sdft"
    assert data["tokens"] == [list(STUDENT_TOKENS)]
    assert data["response_lengths"] == [3]
    assert data["rollout_log_probs"] == [list(STUDENT_LOG_PROBS)]
    assert data["teacher_tokens"] == [[*prompt_ids, 1, 2, 3]]


def _payload(teacher_tokens: list[Any], **overrides: Any) -> dict[str, Any]:
    row = ["i1", list(STUDENT_TOKENS), list(STUDENT_LOSS_MASK), list(STUDENT_LOG_PROBS), 0.0, teacher_tokens]
    return {"samples": [row], "rollout_ids": [0], "loss": "sdft", **overrides}


@pytest.mark.unit
def test_slime_payload_rejects_a_teacher_sequence_that_does_not_end_with_the_response() -> None:
    with pytest.raises(ValueError, match="end with the student's response ids"):
        to_slime_rollout_data(_payload([9, 9, 1, 2, 4]))
    with pytest.raises(ValueError, match="carry a prompt"):
        to_slime_rollout_data(_payload([1, 2, 3]))
    with pytest.raises(ValueError, match="sequence of integers"):
        to_slime_rollout_data(_payload([9, "1", 2, 3]))
    with pytest.raises(ValueError, match="must be \\[source_id"):
        to_slime_rollout_data({"samples": [["i1", [5, 1], [1], [-0.1], 0.0]], "rollout_ids": [0], "loss": "sdft"})
    with pytest.raises(ValueError, match="must omit advantages"):
        to_slime_rollout_data(_payload([9, 1, 2, 3], advantages=[1.0]))


@pytest.mark.unit
def test_sdft_driver_options_travel_from_argv_onto_args() -> None:
    family = resolve_loss_family("sdft")
    settings, remaining = family.parse_driver_options(
        ["--sdft-kl-direction=reverse", "--sdft-importance-sampling-cap=0", "--sdft-skip-response-tokens=3", "--lr=1"]
    )

    assert settings == SdftSettings(kl_direction="reverse", importance_sampling_cap=0.0, skip_response_tokens=3)
    assert remaining == ["--lr=1"]
    args = SimpleNamespace()
    family.apply_driver_options(args, settings)
    assert args.loss_family == "sdft"
    assert (args.sdft_kl_direction, args.sdft_importance_sampling_cap, args.sdft_skip_response_tokens) == (
        "reverse",
        0.0,
        3,
    )

    defaults = SimpleNamespace()
    family.apply_driver_options(defaults, None)
    assert (defaults.sdft_kl_direction, defaults.sdft_importance_sampling_cap, defaults.sdft_skip_response_tokens) == (
        "forward",
        2.0,
        0,
    )
    assert family.bind(settings) is family
    with pytest.raises(TypeError, match="SdftSettings"):
        family.bind(SimpleNamespace())


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("kl_direction", "jsd"),
        ("importance_sampling_cap", -1.0),
        ("importance_sampling_cap", float("nan")),
        ("skip_response_tokens", -1),
    ],
)
def test_sdft_settings_reject_invalid_values(name: str, value: Any) -> None:
    with pytest.raises(ValueError, match=name):
        SdftSettings(**{name: value})


@pytest.mark.unit
def test_sdft_backend_validation_pins_the_loss_type_and_one_step_per_rollout() -> None:
    family = resolve_loss_family("sdft")
    accepted = {"loss_type": "custom_loss", "use_rollout_logprobs": True, "num_steps_per_rollout": 1}
    family.validate_backend_args(SimpleNamespace(**accepted))

    with pytest.raises(RuntimeError, match="loss-type custom_loss"):
        family.validate_backend_args(SimpleNamespace(**{**accepted, "loss_type": "policy_loss"}))
    with pytest.raises(RuntimeError, match="use-rollout-logprobs"):
        family.validate_backend_args(SimpleNamespace(**{**accepted, "use_rollout_logprobs": False}))
    with pytest.raises(RuntimeError, match="num-steps-per-rollout"):
        family.validate_backend_args(SimpleNamespace(**{**accepted, "num_steps_per_rollout": 2}))
