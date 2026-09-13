"""Reef scheduling and publication ordering across independent backend contracts.

The scheduler coordinates training execution, inference admission and recovery
against the durable scenario head. It never executes inference requests or
implements a backend: each operation crosses one of the two runtime contracts.
The remote training-job coordinator owns worker operations and their persisted
journal; this scheduler completes that journal's handshake with Reef commits.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import replace
from typing import Any

from reef.core.batches import TrainingBatch
from reef.core.evaluation import SelectionDecision
from reef.runtime.base import (
    InferenceRuntime,
    PreparedTrainingStep,
    RuntimeContractError,
    TrainingJobResult,
    TrainingRuntime,
    TrainingRuntimeError,
)
from reef.runtime.weights.candidates import ActivatedModel, CandidateTrainingDeferred, ModelCandidate, StaleCandidate


class RuntimeScheduler:
    """Schedule backend work while keeping unpublished weights unavailable.

    Each scenario invokes recovery with its own durable commit identity. When
    backends share one inference service, admission remains engine-wide while
    acknowledgement remains scoped to the scenario that owns the pending job.
    """

    def __init__(self, training_runtime: TrainingRuntime, inference_runtime: InferenceRuntime) -> None:
        self.training_runtime = training_runtime
        self.inference_runtime = inference_runtime
        status = training_runtime.training_job_status()
        self._colocated = bool(status and status.get("colocate"))
        if status is None:
            inference_runtime.mark_published()
        else:
            self._sync_inference_admission(status)

    def recover_pending_step(
        self,
        scenario_step: int,
        *,
        scenario: str | None = None,
        committed_training_job_id: str | None = None,
        committed_training_without_job_id: bool = False,
    ) -> None:
        """Reconcile a backend journal only against the owning scenario commit."""
        if committed_training_job_id is not None and (
            not isinstance(committed_training_job_id, str) or not committed_training_job_id
        ):
            raise TrainingRuntimeError("committed_training_job_id must be a non-empty string or None")
        if not isinstance(committed_training_without_job_id, bool):
            raise TrainingRuntimeError("committed_training_without_job_id must be a boolean")
        scenario = scenario if self.training_runtime.concurrent_training_scenarios else None
        training_job = self.training_runtime.training_job_status()
        if training_job is None:
            if committed_training_job_id is not None:
                self._finish_committed_training_job(committed_training_job_id)
            return
        status = training_job["status"]
        if self._sync_inference_admission(training_job):
            return
        job_scenario = training_job.get("scenario")
        if scenario is not None and isinstance(job_scenario, str) and job_scenario != scenario:
            # The pending job belongs to another scenario sharing this
            # runtime; admission is engine-global and already synced above,
            # but its commit handshake is that scenario's to finish.
            return
        if status == "REJECTING":
            training_job_id = training_job.get("training_job_id")
            if not isinstance(training_job_id, str) or not training_job_id:
                raise TrainingRuntimeError("rejecting training job is missing its durable identity")
            self.training_runtime.reject_training_job(training_job_id)
            self.inference_runtime.resume_admission()
            return
        if status not in {"UPDATING_WEIGHTS", "READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"}:
            return
        rollout_id = training_job.get("rollout_id")
        training_job_id = training_job.get("training_job_id")
        if (
            not isinstance(rollout_id, int)
            or isinstance(rollout_id, bool)
            or not isinstance(training_job_id, str)
            or not training_job_id
        ):
            raise TrainingRuntimeError("training-job status is missing its durable identity")
        if status == "UPDATING_WEIGHTS":
            self.inference_runtime.resume_weight_update(training_job_id)
        if (
            status == "COMPLETE"
            and training_job.get("commit_acknowledged") is not True
            and scenario_step == rollout_id + 1
            and committed_training_job_id is None
            and committed_training_without_job_id
        ):
            # Older bridges resumed before Reef committed and could not write
            # their job identity into the old commit schema. The exact
            # next-step training record is the strongest durable migration
            # proof available; rollback/non-training commits are excluded.
            self._finish_committed_training_job(training_job_id)
            return
        if scenario_step > rollout_id and committed_training_job_id == training_job_id:
            self._finish_committed_training_job(training_job_id)

    def acknowledge_commit(self, scenario_step: int, training_job_id: str, *, scenario: str | None = None) -> None:
        """Release a selected update only after its matching durable commit."""
        self.recover_pending_step(scenario_step, scenario=scenario, committed_training_job_id=training_job_id)

    def _finish_committed_training_job(self, training_job_id: str) -> None:
        self.inference_runtime.acknowledge_publication(training_job_id)
        self.inference_runtime.mark_published()
        self.inference_runtime.resume_admission()

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
        scenario_step: int,
    ) -> PreparedTrainingStep:
        return self.training_runtime.prepare_training_step(
            batch,
            step_preparer,
            algorithm_state,
            scenario_step,
            serving_runtime_load_id=(
                self.inference_runtime.serving_runtime_load_id()
                if self.training_runtime.max_staleness > 0
                else self.inference_runtime.current_runtime_load_id()
            ),
        )

    def execute_training_job(
        self,
        payload: Mapping[str, Any],
    ) -> TrainingJobResult:
        """Execute a native checkpoint job and stage its uncommitted weights."""
        if self._colocated:
            # New requests wait while colocated workers hand shared devices
            # from inference to training. Backend operations preserve already
            # admitted requests across their own memory release and restore.
            self.inference_runtime.pause_admission()

        try:
            checkpoint = self._validated_result(self.training_runtime.execute_training_job(payload))
        except BaseException:
            # A colocated pause may have succeeded before the backend rejected
            # the job. Reopen only when the durable status proves that no
            # training or checkpoint work started.
            job_state = None
            if self._colocated:
                with suppress(Exception):
                    job_state = (self.training_runtime.training_job_status() or {}).get("status")
            if job_state == "IDLE":
                self.inference_runtime.resume_admission()
            raise
        if checkpoint.outcome in {"stale", "storage_blocked"}:
            if self._colocated:
                self.inference_runtime.resume_admission()
            return checkpoint
        if checkpoint.outcome == "complete":
            if (self.training_runtime.training_job_status() or {}).get("commit_acknowledged") is True:
                self.inference_runtime.resume_admission()
            else:
                self.inference_runtime.pause_admission()
            return checkpoint
        if checkpoint.outcome != "checkpoint" or checkpoint.training_job_id is None:
            raise TrainingRuntimeError("deferred weight updates require a checkpoint with a training_job_id")

        if not self._colocated:
            # Training and checkpointing may overlap inference on disjoint
            # GPUs. Close admission only for the short serving-weight update.
            self.inference_runtime.pause_admission()
        updated = self.inference_runtime.resume_weight_update(checkpoint.training_job_id)
        return TrainingJobResult(
            "complete",
            updated.runtime_load_id,
            checkpoint.checkpoint_path,
            metrics=checkpoint.metrics,
            training_job_id=checkpoint.training_job_id,
        )

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        """Train a checkpoint, preserving the currently published source version."""
        current = self.inference_runtime.current_runtime_load_id()
        if self._colocated:
            self.inference_runtime.pause_admission()
        try:
            candidate = self.training_runtime.train_candidate(payload)
        except (CandidateTrainingDeferred, StaleCandidate):
            if self._colocated:
                self.inference_runtime.resume_admission()
            raise
        except BaseException:
            status = None
            if self._colocated:
                with suppress(Exception):
                    status = self.training_runtime.training_job_status()
            if status is not None and status.get("status") == "IDLE":
                self.inference_runtime.resume_admission()
            raise
        if not isinstance(candidate, ModelCandidate):
            raise RuntimeContractError("training runtime must return ModelCandidate")
        return replace(candidate, current_runtime_load_id=current)

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        """Stage selected weights behind closed inference admission."""
        self.inference_runtime.pause_admission()
        return self.inference_runtime.activate_candidate(candidate)

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        """Discard a candidate before reopening the unchanged serving version."""
        self.training_runtime.reject_candidate(candidate, decision)
        self.inference_runtime.resume_admission()

    def _sync_inference_admission(self, training_job: Mapping[str, Any]) -> bool:
        """Apply states that need no weight-update recovery; return if settled."""
        status = training_job["status"]
        if status in {"IDLE", "REJECTED"} or (
            status == "COMPLETE" and training_job.get("commit_acknowledged") is True
        ):
            self.inference_runtime.mark_published()
            self.inference_runtime.resume_admission()
            return True
        if status in {"RUNNING", "CHECKPOINT"}:
            if self._colocated:
                self.inference_runtime.pause_admission()
            else:
                if self.inference_runtime.current_runtime_load_id() is None:
                    self.inference_runtime.mark_published()
                self.inference_runtime.resume_admission()
            return True
        self.inference_runtime.pause_admission()
        return False

    @staticmethod
    def _validated_result(result: Any) -> TrainingJobResult:
        if not isinstance(result, TrainingJobResult):
            raise TrainingRuntimeError(f"train group handle returned invalid training result: {type(result).__name__}")
        return result


__all__ = ["RuntimeScheduler"]
