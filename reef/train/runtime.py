"""Generic training runtime over Reef's backend-neutral coordinator client.

Native integrations choose their connection and retain their own model code.
This runtime prepares scenario jobs and returns exported checkpoint candidates.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.core.batches import TrainingBatch, policy_samples
from reef.core.evaluation import SelectionDecision
from reef.runtime.executor.connection import CoordinatorClient, training_job_status
from reef.runtime.interfaces import (
    CandidateTrainingDeferred,
    ModelCandidate,
    PreparedTrainingStep,
    StaleCandidate,
    TrainingJobResult,
    TrainingRuntime,
    TrainingRuntimeError,
)


class ExecutorTrainingRuntime(TrainingRuntime):
    """Train and export checkpoints through a supplied worker control connection."""

    def __init__(self, train_group_handle: CoordinatorClient, *, max_staleness: int = 0) -> None:
        if not isinstance(max_staleness, int) or isinstance(max_staleness, bool) or max_staleness < 0:
            raise ValueError("max_staleness must be a non-negative integer")
        self._train_group_handle = train_group_handle
        self._max_staleness = max_staleness

    @property
    def train_group_handle(self) -> CoordinatorClient:
        return self._train_group_handle

    @property
    def max_staleness(self) -> int:
        return self._max_staleness

    @property
    def concurrent_training_scenarios(self) -> bool:
        return self.training_job_status().get("lora_mode") == "scenario"

    def training_job_status(self) -> Mapping[str, Any]:
        return training_job_status(self._train_group_handle)

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
        scenario_step: int,
        *,
        serving_runtime_load_id: str | None = None,
    ) -> PreparedTrainingStep:
        prepared = self._train_group_handle.prepare_training_step(batch, step_preparer, algorithm_state)
        if not isinstance(prepared, PreparedTrainingStep):
            raise TrainingRuntimeError(
                f"train group handle returned invalid prepared training step: {type(prepared).__name__}"
            )
        if prepared.action == "skip":
            return prepared
        if prepared.payload is None:
            raise TrainingRuntimeError("non-skip training preparation must carry a payload")
        payload = dict(prepared.payload)
        samples = policy_samples(batch)
        source_rows = payload.pop("source_rows", None)
        if source_rows is not None:
            # Wire rows follow the step schedule (epochs repeat rows, shuffle
            # reorders rollouts); producing versions and timestamps must follow
            # that same order.
            try:
                samples = tuple(samples[row] for row in source_rows)
            except (IndexError, TypeError) as exc:
                raise TrainingRuntimeError(f"prepared payload names invalid source rows: {exc}") from exc
        versions = tuple(sample.runtime_load_id for sample in samples)
        if not samples:
            raise TrainingRuntimeError("a training job requires at least one policy sample")
        version_spans = [
            [
                {
                    "start": span.start,
                    "end": span.end,
                    "runtime_load_id": span.runtime_load_id,
                }
                for span in sample.runtime_load_spans
            ]
            for sample in samples
        ]
        if any(version_spans):
            payload["producing_runtime_load_spans"] = version_spans
        if self._max_staleness == 0 and any(
            version is None and not spans for version, spans in zip(versions, version_spans, strict=True)
        ):
            raise TrainingRuntimeError("a training job requires a recorded producing runtime load ID for every sample")
        span_versions = {span["runtime_load_id"] for spans in version_spans for span in spans}
        recorded_versions = span_versions | {version for version in versions if version is not None}
        requires_staleness_admission = (
            self._max_staleness > 0 or any(version is None for version in versions) or len(recorded_versions) != 1
        )
        if not requires_staleness_admission:
            expected_runtime_load_id = recorded_versions.pop()
            if expected_runtime_load_id is None:
                raise TrainingRuntimeError("recorded producing runtime load ID cannot be null")
        else:
            expected_runtime_load_id = serving_runtime_load_id
            if expected_runtime_load_id is None:
                raise TrainingRuntimeError("token staleness admission requires a verified serving runtime load ID")
            payload["max_staleness"] = self._max_staleness
            payload["producing_runtime_load_ids"] = list(versions)
        # The scenario step crosses into the backend job as ``rollout_id`` —
        # the training backend's own (wire) name for the same integer.
        payload.update(rollout_id=scenario_step, expected_runtime_load_id=expected_runtime_load_id)
        return PreparedTrainingStep(
            action="train",
            payload=payload,
            next_algorithm_state=prepared.next_algorithm_state,
            metrics=prepared.metrics,
        )

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        return self._validated_result(self._train_group_handle.execute_training_job(payload))

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        checkpoint = self.execute_training_job(payload)
        if checkpoint.outcome == "storage_blocked":
            if not isinstance(checkpoint.storage, Mapping):
                raise TrainingRuntimeError("training runtime returned invalid checkpoint storage status")
            raise CandidateTrainingDeferred(checkpoint.storage)
        if checkpoint.outcome == "stale":
            raise StaleCandidate(checkpoint.metrics)
        if checkpoint.outcome != "checkpoint" or checkpoint.training_job_id is None:
            raise TrainingRuntimeError("candidate training must stop after exporting a checkpoint")
        if checkpoint.checkpoint_path is None:
            raise TrainingRuntimeError("exported checkpoint must carry a checkpoint path")
        return ModelCandidate(
            candidate_id=checkpoint.training_job_id,
            training_job_id=checkpoint.training_job_id,
            checkpoint_path=checkpoint.checkpoint_path,
            current_runtime_load_id=None,
            training_metrics=dict(checkpoint.metrics or {}),
        )

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        self.reject_training_job(candidate.training_job_id)

    def reject_training_job(self, training_job_id: str) -> None:
        self._train_group_handle.reject_training_candidate(training_job_id)

    def shutdown(self) -> None:
        self._train_group_handle.shutdown()

    @staticmethod
    def _validated_result(result: Any) -> TrainingJobResult:
        if not isinstance(result, TrainingJobResult):
            raise TrainingRuntimeError(f"train group handle returned invalid training result: {type(result).__name__}")
        return result
