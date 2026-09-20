"""CPU-only synthetic routing policy using Reef's real recipe and commit lifecycle."""

import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from reef_casegraph_adapter import (
    FEEDBACK_TYPES,
    UPDATE_SURFACES,
    CaseEvent,
    SyntheticCaseEventGenerator,
    digest,
    fixture_digest,
)
from reef_record_bridge import CaseEventReport, event_from_record, event_to_record

from reef.artifact import Artifact, GitLFSRepositoryBackend
from reef.core.evaluation import CandidateEvaluationPlugin, EvaluationResult, SelectionDecision, UpdateCandidate
from reef.core.records_types import AgentRecord, RequestType
from reef.inference.model_config import ModelConfig
from reef.observability import ExperimentLogger, NullExperimentTracker
from reef.recipe import Recipe
from reef.scenario import Scenario
from reef.scenario.factory import ScenarioFactory
from reef.storage.records import RecordStore
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.backend import CandidateBackend, PreparedStep
from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.trainer import Trainer
from reef.train.types import ProcessorContext, TrainingBatch, TrainStepResult


@dataclass(frozen=True)
class FrozenFixture:
    """JSON strings freeze nested event values, not just their outer dataclass."""

    training: tuple[str, ...]
    retained: tuple[str, ...]
    drift: tuple[str, ...]
    checksum: str

    @property
    def identity(self) -> str:
        return digest([self.training, self.retained, self.drift, self.checksum])

    @classmethod
    def load(cls, path: Path) -> "FrozenFixture":
        raw = json.loads(path.read_text())
        events = tuple(CaseEvent.from_dict(item) for item in raw["events"])
        if fixture_digest(events) != raw["event_digest"]:
            raise ValueError("fixture digest mismatch")
        splits = SyntheticCaseEventGenerator().split(events)
        manifest = {name: sorted({e.case_id for e in arm}) for name, arm in splits.items()}
        if manifest != raw["splits"]:
            raise ValueError("fixture split manifest mismatch")
        return cls(
            tuple(e.serialize() for e in splits["replay"] + splits["adapt"]),
            tuple(e.serialize() for e in splits["retained"]),
            tuple(e.serialize() for e in splits["drift"]),
            fixture_digest(events),
        )


def initial_policy() -> dict[str, str]:
    return {"observation": "episodic_memory", **dict.fromkeys(FEEDBACK_TYPES, "procedural_harness")}


def read_policy(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"observation", *FEEDBACK_TYPES}:
        raise ValueError("policy must define observation and every feedback category")
    if not all(
        isinstance(key, str) and isinstance(surface, str) and surface in UPDATE_SURFACES
        for key, surface in value.items()
    ):
        raise ValueError("policy has an invalid corrective surface")
    return dict(value)


def task_results(policy: Mapping[str, str], events: tuple[str, ...]) -> list[dict[str, object]]:
    """Execute routing decisions; exact expected actions come from frozen labels."""
    results = []
    for serialized in events:
        event = CaseEvent.deserialize(serialized)
        predicted = policy[event.feedback_type or "observation"]
        results.append(
            {
                "event_id": event.event_id,
                "case_id": event.case_id,
                "predicted": predicted,
                "expected": event.update_surface,
                "correct": predicted == event.update_surface,
            }
        )
    return results


@dataclass(frozen=True, kw_only=True)
class RoutingCandidate(UpdateCandidate):
    artifact: Artifact
    incumbent_json: str
    policy_json: str


class RetainedRoutingEvaluator(CandidateEvaluationPlugin):
    def __init__(self, retained: tuple[str, ...]) -> None:
        if not retained:
            raise ValueError("retained tasks must not be empty")
        self.retained = retained

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        if not isinstance(candidate, RoutingCandidate) or candidate.artifact.local_path is None:
            raise TypeError("expected a materialized routing candidate")
        actual = (candidate.artifact.local_path / "policy.json").read_text()
        if actual != candidate.policy_json:
            raise ValueError("candidate artifact changed after preparation")
        incumbent = task_results(read_policy(json.loads(candidate.incumbent_json)), self.retained)
        proposed = task_results(read_policy(json.loads(actual)), self.retained)
        return EvaluationResult(
            "synthetic-feedback-routing",
            "1",
            {
                "baseline_accuracy": sum(row["correct"] is True for row in incumbent) / len(incumbent),
                "candidate_accuracy": sum(row["correct"] is True for row in proposed) / len(proposed),
            },
            {"baseline_tasks": incumbent, "candidate_tasks": proposed, "retained_digest": digest(self.retained)},
        )

    def decide(self, candidate: UpdateCandidate, evaluation: EvaluationResult) -> SelectionDecision:
        selected = evaluation.metrics["candidate_accuracy"] >= evaluation.metrics["baseline_accuracy"]
        return SelectionDecision(
            "select" if selected else "reject",
            "retained-routing-no-regression",
            "1",
            "candidate matches or improves incumbent routing" if selected else "candidate routing regressed",
            evaluation,
        )


@dataclass(frozen=True)
class RoutingBatch(TrainingBatch):
    events: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()


class CaseGraphProcessor(DataProcessor):
    required_request_types = frozenset({RequestType.REPORT})
    output_schema = RoutingBatch

    def __init__(self, context: ProcessorContext, training: tuple[str, ...]) -> None:
        super().__init__(context)
        self.training = frozenset(training)
        self.buffered: dict[str, str] = {}
        self.batch: RoutingBatch | None = None
        self.released: set[str] = set()

    def ingest(self, item: AgentRecord) -> None:
        event = event_from_record(item)
        if item.scenario != self.scenario or event.serialize() not in self.training:
            raise ValueError("record is not an exact frozen training event in this scenario")
        self.buffered[item.agent_record_id] = event.serialize()

    def ready(self) -> bool:
        return self.batch is not None or frozenset(self.buffered.values()) == self.training

    def build_batch(self) -> RoutingBatch:
        if not self.ready():
            raise RuntimeError("all frozen training records are required")
        if self.batch is None:
            records = sorted(self.buffered.items())
            self.batch = RoutingBatch(
                batch_id=digest(records),
                events=tuple(event for _, event in records),
                record_ids=tuple(record_id for record_id, _ in records),
            )
        return self.batch

    def acknowledge(self, batch_id: str) -> frozenset[str]:
        if self.batch is None or self.batch.batch_id != batch_id:
            raise ValueError("unknown routing batch")
        consumed = frozenset(self.batch.record_ids)
        self.released.update(consumed)
        for record_id in consumed:
            self.buffered.pop(record_id)
        self.batch = None
        return consumed

    def release_batch(self, batch_id: str) -> None:
        if self.batch is None or self.batch.batch_id != batch_id:
            raise ValueError("unknown routing batch")
        self.batch = None

    def retention_decision(self) -> RetentionDecision:
        return RetentionDecision(frozenset(self.buffered), frozenset(self.released))

    def compaction_applied(self, agent_record_ids: frozenset[str]) -> None:
        self.released.difference_update(agent_record_ids)


class CaseGraphBackend(CandidateBackend):
    """Learn a finite routing table; Reef alone publishes and commits it."""

    def __init__(self, work_dir: Path, checksum: str, fixture_identity: str) -> None:
        self.work_dir = work_dir
        self.checksum = checksum
        self.fixture_identity = fixture_identity
        self.artifacts: list[Artifact] = []

    def initial_state(self) -> Mapping[str, object]:
        return {"policy": initial_policy(), "fixture_digest": self.checksum}

    def learn_policy(self, events: tuple[str, ...], incumbent: dict[str, str]) -> dict[str, str]:
        policy = dict(incumbent)
        for serialized in events:
            event = CaseEvent.deserialize(serialized)
            policy[event.feedback_type or "observation"] = event.update_surface
        return policy

    def prepare_step(self, batch: TrainingBatch, state: Mapping[str, object], scenario_step: int) -> PreparedStep:
        if not isinstance(batch, RoutingBatch):
            raise TypeError("expected RoutingBatch")
        incumbent = read_policy(state["policy"])
        policy = read_policy(self.learn_policy(batch.events, incumbent))
        encoded = json.dumps(policy, sort_keys=True, indent=2) + "\n"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix="candidate-", dir=self.work_dir))
        (path / "policy.json").write_text(encoded)
        (path / "fixture-identity.txt").write_text(self.fixture_identity + "\n")
        artifact = Artifact.local(path, metadata={"synthetic": True, "fixture_digest": self.checksum})
        self.artifacts.append(artifact)
        candidate = RoutingCandidate(
            candidate_id=digest([batch.batch_id, scenario_step, policy]),
            artifact=artifact,
            incumbent_json=json.dumps(incumbent, sort_keys=True),
            policy_json=encoded,
        )
        return PreparedStep.with_candidate(candidate, state=state)

    def evaluate(self, candidate: UpdateCandidate) -> EvaluationResult:
        raise RuntimeError("CaseGraphRecipe requires a separately bound retained evaluator")

    def settle_step(self, prepared: PreparedStep, decision: SelectionDecision) -> TrainStepResult:
        candidate = prepared.candidate
        if not isinstance(candidate, RoutingCandidate):
            raise TypeError("expected RoutingCandidate")
        metrics = {"selection": decision.to_dict()}
        if not decision.selected:
            candidate.artifact.discard()
            return TrainStepResult(prepared.state, metrics)
        return TrainStepResult(
            {"policy": read_policy(json.loads(candidate.policy_json)), "fixture_digest": self.checksum},
            metrics,
            artifact=candidate.artifact,
        )

    def abort_step(self, prepared: PreparedStep) -> None:
        if isinstance(prepared.candidate, RoutingCandidate):
            prepared.candidate.artifact.discard()

    def close(self) -> None:
        for artifact in self.artifacts:
            artifact.discard()
        self.artifacts.clear()


@dataclass(frozen=True, kw_only=True)
class CaseGraphRecipe(Recipe):
    fixture: FrozenFixture
    work_dir: Path
    name: str = "synthetic-casegraph-routing"

    @property
    def report_type(self) -> type[CaseEventReport]:
        return CaseEventReport

    def build(
        self,
        scenario: str,
        records: RecordStore,
        *,
        algorithm_state: Mapping[str, object] | None = None,
        experiment_logger: ExperimentLogger | None = None,
    ) -> Trainer:
        evaluator = RetainedRoutingEvaluator(self.fixture.retained)
        backend = CaseGraphBackend(self.work_dir, self.fixture.checksum, self.fixture.identity)
        state = backend.initial_state() if algorithm_state is None else algorithm_state
        if state.get("fixture_digest") != self.fixture.checksum:
            raise ValueError("cannot resume with a different frozen fixture")
        read_policy(state["policy"])
        context = ProcessorContext(scenario, report_type=self.report_type, training_mode=self.training_mode)
        processor = CaseGraphProcessor(context, self.fixture.training)
        return Trainer(
            scenario=scenario,
            records=records,
            processor=processor,
            candidate_backend=backend,
            candidate_evaluator=evaluator,
            state=state,
        )

    def base_artifact_files(self) -> Mapping[str, str]:
        return {
            "policy.json": json.dumps(initial_policy(), sort_keys=True, indent=2) + "\n",
            "fixture-identity.txt": self.fixture.identity + "\n",
        }


def open_scenario(work_dir: Path, recipe: CaseGraphRecipe) -> Scenario:
    """Open real durable stores without starting HTTP or background training workers."""
    factory = GitLFSRepositoryBackend.factory(
        work_dir / "artifacts.git",
        work_dir=work_dir / "git-work",
        cache_dir=work_dir / "git-cache",
        bootstrap_files=recipe.base_artifact_files(),
    )
    storage = SQLiteScenarioStorage(work_dir / "records")
    try:
        scenario = ScenarioFactory(
            recipe, factory, experiment_tracker=NullExperimentTracker(), scenario_storage=storage
        ).load_or_create("synthetic-casegraph", model_config=ModelConfig())
        try:
            artifact = scenario.artifact_for_version(scenario.current_artifact_ref().release_id).materialize()
            if artifact.local_path is None:
                raise RuntimeError("published artifact is not materialized")
            if (artifact.local_path / "fixture-identity.txt").read_text().strip() != recipe.fixture.identity:
                raise ValueError("frozen fixture differs from the registered scenario")
            return scenario
        except (OSError, ValueError, RuntimeError):
            scenario.close()
            raise
    finally:
        # Storage is a session factory; the returned scenario owns its open session.
        storage.close()


def ingest_training(
    scenario: Scenario, fixture: FrozenFixture, events: tuple[CaseEvent, ...] | None = None
) -> tuple[AgentRecord, ...]:
    """Validate the entire batch before appending; only exact frozen identities pass."""
    selected = tuple(CaseEvent.deserialize(value) for value in fixture.training) if events is None else events
    allowed = frozenset(fixture.training)
    processor = scenario.trainer.processor
    if not isinstance(processor, CaseGraphProcessor) or processor.training != allowed:
        raise ValueError("admission fixture does not match the scenario training manifest")
    if any(event.serialize() not in allowed for event in selected):
        raise ValueError("held-out or altered event cannot enter training storage")
    observations: dict[str, AgentRecord] = {}
    records = []
    for event in selected:
        source = observations.get(event.case_id)
        record = event_to_record(event, scenario=scenario.name, observation=source)
        CaseEventReport.from_dict(record.payload)
        receipt = scenario.records.append(record)
        records.append(receipt)
        if event.event_type == "observation":
            observations[event.case_id] = receipt
    return tuple(records)
