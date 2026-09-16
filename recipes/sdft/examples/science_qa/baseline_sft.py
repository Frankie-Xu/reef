"""The SFT control of this example: supervised fine-tuning on the same demonstrations, through the same stack.

Table 5 of the paper compares SDFT with SFT on the same demonstrations. This
module is that baseline for Reef: the same reports (a rollout's receipt and
the demonstration as ``context``), the same optimizer and step cadence, but
the sample the trainer sees is the recorded request followed by the
demonstration rendered as the assistant turn, trained with Slime's stock
``sft_loss`` over the demonstration's tokens. The student's own sample is
recorded and ignored, so ``run.py`` drives both arms unchanged.

The recipe, processor, objective and loss family live here rather than in
the recipe package because they are this experiment's control, not a method:
``serve-sft.yaml`` names them by dotted reference, which is also how the
Slime driver and its workers import the loss family.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from recipes.sdft.teacher_prompt import normalize_messages_for_template, recorded_request
from reef.core.reports import ReportBase, TeacherContextReport
from reef.recipe.base import WeightTrainingRecipe, WeightTrainingSpec
from reef.recipe.config_fields import config_field
from reef.recipe.errors import RecipeConfigError
from reef.train.algos import StepScheduling, TrainingObjective
from reef.train.algos.helpers import next_steps
from reef.train.algos.signals import StepSignal
from reef.train.processors.reported import ReportContext, ReportedFeedbackProcessor, SampleAssembly
from reef.train.slime_backend.algorithm import SlimeAlgorithm
from reef.train.types import ProcessorContext, TrainDataItem, TrainingBatch, TrajectoryItem, trajectories

LOSS_FAMILY = "demonstration-sft"
MODULE = "recipes.sdft.examples.science_qa.baseline_sft"


class DemonstrationTokenizer(ABC):
    """Render a request and its demonstration into prompt ids and the demonstration's response ids."""

    @abstractmethod
    def sequence_ids(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None, demonstration: str
    ) -> tuple[list[int], list[int]]:
        """``(prompt_ids, response_ids)``: the request with the generation prompt, then the assistant turn."""


class ChatTemplateDemonstrationTokenizer(DemonstrationTokenizer):
    """The served model's Hugging Face tokenizer applying its own chat template.

    The demonstration becomes the assistant message of the conversation; its
    response ids are what the template adds after the generation prompt (the
    content, the end-of-turn token and the template's turn separator), so
    the trained sequence is exactly the model's own rendering of that answer.
    """

    def __init__(self, tokenizer_path: str) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def sequence_ids(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None, demonstration: str
    ) -> tuple[list[int], list[int]]:
        chat = normalize_messages_for_template(messages)
        template_tools = list(tools) if tools else None
        prompt = self._tokenizer.apply_chat_template(
            chat, tools=template_tools, tokenize=False, add_generation_prompt=True
        )
        conversation = self._tokenizer.apply_chat_template(
            [*chat, {"role": "assistant", "content": demonstration}],
            tools=template_tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not conversation.startswith(prompt):
            raise ValueError(
                "the chat template does not render the generation prompt as a prefix of the assistant turn"
            )
        prompt_ids = [int(token) for token in self._tokenizer(prompt, add_special_tokens=False)["input_ids"]]
        response_ids = [
            int(token) for token in self._tokenizer(conversation[len(prompt) :], add_special_tokens=False)["input_ids"]
        ]
        return prompt_ids, response_ids


class DemonstrationSFTProcessor(ReportedFeedbackProcessor):
    """One report, one supervised sample: the recorded request followed by its demonstration."""

    output_schema = TrainingBatch
    exclusive_sources = True

    def __init__(self, context: ProcessorContext, tokenizer: DemonstrationTokenizer | None = None) -> None:
        self._assembly = SampleAssembly.from_config(context)
        if tokenizer is None:
            tokenizer_path = str(context.config.get("tokenizer_path", "")).strip()
            if not tokenizer_path:
                raise ValueError(
                    "demonstration SFT requires tokenizer_path: the served model's tokenizer renders the sample"
                )
            tokenizer = ChatTemplateDemonstrationTokenizer(tokenizer_path)
        self._tokenizer = tokenizer
        super().__init__(context)

    def make_sample(self, context: ReportContext) -> TrajectoryItem:
        parsed = context.parsed_report
        if not isinstance(parsed, TeacherContextReport):
            raise ValueError("DemonstrationSFTProcessor requires the TeacherContextReport schema")
        if len(context.inferences) != 1:
            raise ValueError("demonstration SFT trains one recorded request per report")
        sample = self._assembly.build(context, 0.0 if context.score is None else context.score)
        messages, tools = recorded_request(context.inferences[0].payload)
        prompt_ids, response_ids = self._tokenizer.sequence_ids(messages, tools, parsed.context)
        if not response_ids:
            raise ValueError("the demonstration rendered to no tokens")
        # The student's sample gives way to the demonstration; the token
        # spans that described the student's response no longer apply.
        return sample.with_training(
            tokens=[*prompt_ids, *response_ids],
            loss_mask=[1] * len(response_ids),
            rollout_log_probs=[],
            runtime_load_spans=[],
        )

    def make_batch(self, items: tuple[TrainDataItem, ...], batch_number: int) -> TrainingBatch:
        return TrainingBatch(f"{self.scenario}:demonstration-sft:{batch_number}", items)


class DemonstrationSftAlgorithm(SlimeAlgorithm):
    """Slime's stock ``sft_loss`` over the default policy row: every masked token, unweighted."""

    loss_family = LOSS_FAMILY
    loss_type = "sft_loss"
    advantages = "forbidden"
    forbidden_advantages_message = (
        "demonstration SFT trains every demonstration token unweighted; the payload must omit advantages"
    )

    def validate_specific_args(self, args: Namespace, source: str) -> None:
        return None


#: The family instance the dotted references below name. Every process that
#: names the family resolves this one object, so nothing registers at import.
DEMONSTRATION_SFT = DemonstrationSftAlgorithm()


class DemonstrationSftObjective(TrainingObjective):
    name = LOSS_FAMILY
    # The dotted reference: the driver imports the family from this module and
    # stamps the reference for its workers.
    loss_family = f"{MODULE}:DEMONSTRATION_SFT"

    def prepare(self, batch: TrainingBatch, state: Mapping[str, Any]) -> StepSignal:
        samples = trajectories(batch)
        steps = next_steps(state)
        return StepSignal("train", {"steps": steps}, {"steps": steps, "samples": len(samples)})


@dataclass(frozen=True, kw_only=True)
class DemonstrationSFTRecipe(WeightTrainingRecipe):
    """SFT on demonstrations, the control arm of the Science Q&A comparison.

    Accepts the same ``TeacherContextReport`` the sdft recipe does and trains
    the demonstration as the assistant turn of the recorded request.
    ``batch_size`` must equal the Slime driver's ``--global-batch-size``.
    """

    name: str = LOSS_FAMILY
    batch_size: int = config_field(1)
    tokenizer_path: str = config_field("")

    @property
    def report_type(self) -> type[ReportBase]:
        return TeacherContextReport

    @classmethod
    def training_spec(cls) -> WeightTrainingSpec:
        return WeightTrainingSpec(
            objective=f"{MODULE}:DemonstrationSftObjective",
            processor=DemonstrationSFTProcessor,
            scheduling=StepScheduling(unit="sample"),
        )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self.tokenizer_path.strip():
            raise ValueError("tokenizer_path is required: the served model's tokenizer renders the sample")

    @classmethod
    def _validate_config(cls, settings: Mapping[str, Any]) -> None:
        if settings.get("optimization"):
            raise RecipeConfigError("demonstration SFT has no objective options; the optimizer is training.options")
