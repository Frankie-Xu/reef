"""Reef-owned scheduling of independent training and inference operations.

Backend adapters expose individual worker operations. This coordinator owns
resource handoff, durable training and publication, scenario adapter residency,
startup recovery and the commit barrier. It imports no backend implementation.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from reef.runtime.adapter_residency import AdapterCapacityExhausted, AdapterEvictionFailed, AdapterResidencyManager
from reef.runtime.base import PreparedTrainingStep, TrainingJobResult
from reef.runtime.runtime_load_id import RuntimeLoadId, new_runtime_load_id_incarnation
from reef.runtime.training_job.admission import (
    _admission_runtime_load_id_groups,
    _scenario_staleness_admission,
    _stale_drop_decision,
    _staleness_admission,
)
from reef.runtime.training_job.execution import (
    PreparedTrainingJob,
    TrainingCheckpoint,
    TrainingExecution,
    TrainingMetrics,
    max_staleness,
    uses_staleness_admission,
)
from reef.runtime.training_job.marker import marker_path, marker_result, read_marker, write_marker
from reef.runtime.training_job.publication import TrainingPublication
from reef.runtime.training_job.scenarios import ScenarioHistory
from reef.surface.adapter import adapter_name, parse_adapter_name


@dataclass(frozen=True)
class TrainingCoordinationConfig:
    """Deployment policy interpreted only by Reef's coordinator."""

    save_hf_template: str | None
    colocate: bool = False
    lora: bool = False
    adapter_capacity: int | None = None
    keep_lora_base_resident: bool = False


def _initial_runtime_load_id() -> str:
    return str(RuntimeLoadId(new_runtime_load_id_incarnation(), 0))


@dataclass
class TrainingContext:
    """Scheduling state available to backend preparation without control handles."""

    next_rollout_id: int = 0
    runtime_load_id: str = field(default_factory=_initial_runtime_load_id)
    history: ScenarioHistory | None = None


class TrainingOperations(Protocol):
    """Training-only preparation, checkpoint I/O and native weight sending.

    Sender methods must never pause, resume, offload or restart inference.
    Reef supplies the exact identity for each transfer. The sender must echo
    that identity; Reef independently verifies every receiver before commit.
    """

    config: TrainingCoordinationConfig
    context: TrainingContext

    def start(self) -> None: ...

    def check_health(self) -> None: ...

    def prepare_training_step(
        self, batch: Any, step_preparer: str, algorithm_state: Mapping[str, Any]
    ) -> PreparedTrainingStep: ...

    def prepare(
        self, payload: Mapping[str, Any], *, job_id: str, rollout_id: int, prior_marker: Mapping[str, Any] | None
    ) -> AbstractContextManager[PreparedTrainingJob | TrainingJobResult]: ...

    def prepare_weights(self, runtime_load_id: str, *, force_full: bool) -> None:
        """Prepare a sender while colocated inference resources are released."""

    def send_weights(self, runtime_load_id: str, *, force_full: bool) -> str: ...

    def initialize_version(self, runtime_load_id: str) -> None: ...

    def activate_scenario(self, scenario: str) -> None: ...

    def send_adapter(self, scenario: str, name: str) -> None: ...

    def close(self) -> None: ...


class InferenceOperations(Protocol):
    """Receiver operations with acknowledged completion and no commit policy."""

    def initialize_version(self, runtime_load_id: str) -> None: ...

    def inference_url(self) -> str: ...

    def runtime_load_ids(self) -> Sequence[str]: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def recover(self) -> None: ...

    def abort(self) -> None: ...

    def offload(self, tags: tuple[str, ...] | None) -> None: ...

    def onload_weights(self) -> None: ...

    def onload_kv(self) -> None: ...

    def unload_adapter(self, name: str) -> None: ...


@dataclass(frozen=True)
class _AdapterTransfer:
    scenario: str
    name: str


class _CoordinatedAdapterEngine:
    """Reef residency operations across an explicit sender and receiver."""

    def __init__(self, training: TrainingOperations, inference: InferenceOperations) -> None:
        self._training = training
        self._inference = inference

    def load_adapter(self, name: str, payload: Any) -> None:
        if not isinstance(payload, _AdapterTransfer) or payload.name != name:
            raise TypeError(f"adapter {name!r} requires a matching adapter transfer")
        self._training.send_adapter(payload.scenario, payload.name)

    def unload_adapter(self, name: str) -> None:
        self._inference.unload_adapter(name)


class TrainingCoordinator:
    """Serialize backend work and preserve Reef's serving commit barrier."""

    def __init__(
        self, training: TrainingOperations, inference: InferenceOperations, *, owns_training: bool = True
    ) -> None:
        self._training = training
        self._inference = inference
        self._owns_training = owns_training
        self._context = training.context
        config = training.config
        self._save_hf_template = config.save_hf_template
        self._colocate = config.colocate
        self._lora = config.lora
        self._history = self._context.history
        self._residency = AdapterResidencyManager(config.adapter_capacity) if config.lora else None
        self._adapter_engine = _CoordinatedAdapterEngine(training, inference)
        self._release_tags = (
            ("kv_cache", "cuda_graph")
            if (config.lora and config.colocate and config.keep_lora_base_resident)
            else None
        )
        self._generation_paused = False
        path = marker_path(config.save_hf_template) if config.save_hf_template is not None else None
        self._publication = TrainingPublication(path, _CoordinatedWeightPublisher(self))
        self._execution = TrainingExecution(path, _ScheduledTrainingBackend(self), self._publication.state)
        self._closed = False
        self._completed_train_steps = 0
        self._last_train_rollout_id: int | None = None
        self._last_train_metrics: dict[str, Any] = {}
        self._operation_lock = Lock()
        self._training.start()
        marker = self._recover_marker()
        with self._publication.recovery(marker):
            self._restore_serving(marker)

    @property
    def _runtime_load_id(self) -> str:
        return self._context.runtime_load_id

    @_runtime_load_id.setter
    def _runtime_load_id(self, value: str) -> None:
        self._context.runtime_load_id = value

    @property
    def _next_rollout_id(self) -> int:
        return self._context.next_rollout_id

    @_next_rollout_id.setter
    def _next_rollout_id(self, value: int) -> None:
        self._context.next_rollout_id = value

    @property
    def _phase(self) -> str:
        return self._publication.phase

    @_phase.setter
    def _phase(self, phase: str) -> None:
        self._publication.phase = phase

    def prepare_training_step(
        self, batch: Any, step_preparer: str, algorithm_state: Mapping[str, Any]
    ) -> PreparedTrainingStep:
        return self._training.prepare_training_step(batch, step_preparer, algorithm_state)

    def shutdown(self) -> None:
        """Close training workers; deployment ownership closes inference separately."""
        with self._operation_lock:
            if self._closed:
                return
            self._closed = True
            self._phase = "stopped"
            if self._owns_training:
                self._training.close()

    def _prepare_training(self) -> None:
        if self._colocate:
            self._pause_generation()
            self._inference.offload(self._release_tags)

    def _restore_serving(self, marker: dict[str, Any] | None) -> None:
        """Reconstruct backend state inside Reef's startup recovery barrier."""
        marker_status = None if marker is None else str(marker["status"])
        if marker_status == "UPDATING_WEIGHTS":
            # The previous fan-out may have updated only some engines. Recover
            # dead actors first, keep every engine paused, and force a complete
            # tensor transfer from the durable checkpoint-backed actor state.
            self._inference.recover()
        if self._colocate and self._lora:
            # Cold startup releases everything before training initializes.
            # Restore the frozen base before registering scenario adapters,
            # including runs that only release KV/graphs on later steps.
            self._inference.onload_weights()
        if marker_status == "REJECTING":
            if marker is None:
                raise RuntimeError("REJECTING marker status has no marker payload")
            self._publication.reject(str(marker["job_id"]))
            marker["status"] = marker_status = "REJECTED"
            self._pause_generation(reconcile=True)
        self._inference_url = self._inference.inference_url()
        versions = self._inference.runtime_load_ids()
        if not versions or (marker_status != "UPDATING_WEIGHTS" and len({str(version) for version in versions}) != 1):
            raise RuntimeError(f"serving engines disagree at coordinator startup: {versions!r}")
        # A receiver observation never selects a new Reef identity. Pending
        # recovery reuses its persisted target; a fresh deployment keeps the
        # incarnation Reef assigned before either backend was started.
        recovered_runtime_load_id = None
        if marker_status in {"READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"}:
            if marker is None:
                raise RuntimeError(f"{marker_status} marker status has no marker payload")
            recovered_runtime_load_id = str(marker["runtime_load_id"])
        if self._history is not None:
            self._recover_scenario_adapters(marker)
        if self._save_hf_template is not None and marker_status != "REJECTED" and not (self._lora and marker is None):
            # The Megatron checkpoint can be newer than the HF checkpoint used
            # to boot inference. Publish actor weights before construction returns
            # so the first Reef inference uses the actual training version. A
            # LoRA bridge that never trained has nothing to publish: the frozen
            # base inference booted from is exactly what every fresh adapter
            # computes, and the history replay above restored trained ones.
            self._runtime_load_id = self._update_serving(
                # A fresh sender may only capture a delta baseline unless a
                # real full transfer is requested. Startup must load bytes
                # and acknowledge Reef's identity on every receiver.
                force_full=True,
                scenario=self._marker_scenario(marker),
                runtime_load_id=recovered_runtime_load_id or self._publication_target(marker),
            )
        elif self._lora and marker is None:
            # Nothing to publish, but the engines still need Reef's canonical
            # version token (they boot with a backend default), and colocated
            # engines boot released: give them their weights and KV back
            # before the first request.
            if self._colocate:
                self._inference.onload_weights()
                self._inference.onload_kv()
            self._initialize_version()
        elif marker_status == "REJECTED":
            if marker is None:
                raise RuntimeError("REJECTED status requires a durable job marker")
            self._runtime_load_id = str(marker.get("parent_runtime_load_id") or versions[0])
            self._initialize_version()
        elif self._save_hf_template is None:
            self._initialize_version()
        self._publication.finish_recovery(marker, self._runtime_load_id)

    def _initialize_version(self) -> None:
        self._training.initialize_version(self._runtime_load_id)
        self._inference.initialize_version(self._runtime_load_id)
        observed = [str(value) for value in self._inference.runtime_load_ids()]
        if not observed or set(observed) != {self._runtime_load_id}:
            raise RuntimeError(f"serving engines disagree after version sync: {observed!r}")

    def _recover_scenario_adapters(self, marker: Mapping[str, Any] | None) -> None:
        """Re-register every scenario's committed adapter after a restart.

        The Megatron checkpoint restores only the slot's last occupant; the
        other scenarios come back from their persisted slot snapshots. Each
        is loaded under the name its last publication recorded, so Reef's
        routing for that scenario keeps resolving. The marker's scenario is
        activated last: the regular startup republication then publishes it
        under the recovered runtime load ID.
        """
        history = self._require_history()
        residency = self._require_residency()
        active = marker.get("scenario") if marker is not None else None
        pending = [
            (scenario, adapter)
            for scenario in history.scenarios
            if (adapter := history.adapter(scenario)) is not None and scenario != active
        ]
        if not pending and active is None:
            return
        self._pause_generation()
        if self._colocate:
            self._inference.onload_weights()
        for scenario, adapter in pending:
            _, version = parse_adapter_name(adapter)
            residency.activate(
                scenario,
                version,
                self._adapter_engine,
                payload=_AdapterTransfer(scenario, adapter),
            )
        if active is not None:
            self._training.activate_scenario(active)

    def _require_history(self) -> ScenarioHistory:
        """The per-scenario history; only LoRA runs with a checkpoint save path keep one."""
        if self._history is None:
            raise RuntimeError("scenario bookkeeping requires LoRA training with a checkpoint save path")
        return self._history

    def _require_residency(self) -> AdapterResidencyManager:
        if self._residency is None:
            raise RuntimeError("adapter residency requires a LoRA bridge")
        return self._residency

    def _marker_scenario(self, marker: Mapping[str, Any] | None) -> str | None:
        """The scenario a marker's publication belongs to, when the bridge trains per scenario."""
        if self._history is None or marker is None:
            return None
        scenario = marker.get("scenario")
        return str(scenario) if isinstance(scenario, str) and scenario else None

    def health(self) -> dict[str, Any]:
        """Return a lightweight liveness marker for container health checks."""
        self._training.check_health()
        training_job: dict[str, Any] = {
            "deferred_weight_update": self._save_hf_template is not None,
            "status": "COMPLETE" if self._save_hf_template is None else "IDLE",
        }
        if self._save_hf_template is not None and (marker := read_marker(self._marker_path())) is not None:
            training_job.update(
                status=marker["status"],
                training_job_id=marker["job_id"],
                # Reef reasons in scenario steps; in per-scenario mode the
                # marker's rollout id is the bridge-global checkpoint index.
                rollout_id=marker.get("scenario_step", marker["rollout_id"]),
                runtime_load_id=marker.get("runtime_load_id"),
                commit_acknowledged=marker.get("commit_acknowledged", False),
            )
            if "scenario" in marker:
                training_job["scenario"] = marker["scenario"]
        ok = self._phase not in {"training_failed", "checkpoint_failed", "weight_sync_failed", "stopped"}
        return {
            "ok": ok,
            # A publication failure with a durable UPDATING_WEIGHTS marker is
            # replayable in place: ``update_serving_weights`` recovers the
            # engines and republishes from the checkpoint. The coordinator decides
            # which failures are retryable so callers never re-derive it from
            # phase and marker.
            "recoverable": not ok
            and self._phase == "weight_sync_failed"
            and training_job.get("status") == "UPDATING_WEIGHTS",
            "start_rollout_id": self._next_rollout_id,
            "phase": self._phase,
            "colocate": self._colocate,
            # Where the serving engines answer; Reef dials this when the
            # deployment leaves ``reef.inference_url`` unset.
            "inference_url": self._inference_url,
            "lora_adapter": None,
            "lora_mode": "scenario" if self._lora else None,
            "lora_adapters": {} if self._history is None else self._history.status(),
            "adapter_residency": None if self._residency is None else self._residency.status(),
            "completed_train_steps": self._completed_train_steps,
            "last_train_rollout_id": self._last_train_rollout_id,
            "last_train_metrics": dict(self._last_train_metrics),
            "training_job": training_job,
        }

    def start_rollout_id(self) -> int:
        return self._next_rollout_id

    def republish_serving(self) -> str:
        """Recover serving actors and republish unchanged weights in place.

        This path is for an inference-engine replacement, not a training step.
        Keep the current token because the checkpoint/model tensors have not
        changed; the next optimizer-backed publication advances it normally.
        """
        with self._operation_lock:
            return self._publication.republish(self._runtime_load_id)

    def execute_training_job(self, payload: Mapping[str, Any]) -> TrainingJobResult:
        """Delegate job replay, train ordering and checkpoint recording to Reef."""
        if self._save_hf_template is None:
            raise RuntimeError("training checkpoint path is not configured")
        with self._operation_lock:
            return self._execution.execute(payload)

    def update_serving_weights(self, training_job_id: str) -> TrainingJobResult:
        """Delegate durable publication ordering to Reef's shared coordinator."""
        with self._operation_lock:
            publication = self._publication.publish(training_job_id)
            marker = publication.marker
            if publication.published:
                rollout_id = int(marker["rollout_id"])
                self._next_rollout_id = max(self._next_rollout_id, rollout_id + 1)
                self._completed_train_steps += 1
                self._last_train_rollout_id = rollout_id
                recorded_train_metrics = marker.get("train_metrics")
                self._last_train_metrics = (
                    dict(recorded_train_metrics) if isinstance(recorded_train_metrics, Mapping) else {}
                )
            return marker_result(marker)

    def reject_training_candidate(self, training_job_id: str) -> None:
        """Finish a checkpointed job without changing the serving weights."""
        with self._operation_lock:
            marker = self._publication.reject(training_job_id)
            self._next_rollout_id = max(self._next_rollout_id, int(marker["rollout_id"]) + 1)

    def _restore_incumbent_serving(self) -> None:
        if not self._colocate:
            return
        # Pairs with the training step's offload: resuming a region that was
        # never released fails, because the receiver resumes by removing the tag
        # from the set release added it to.
        if self._release_tags is None:
            self._inference.onload_weights()
        self._inference.onload_kv()
        self._continue_generation()

    def acknowledge_training_commit(self, training_job_id: str) -> None:
        """Resume requests only through Reef's durable commit gate."""
        with self._operation_lock:
            self._publication.acknowledge(training_job_id)

    def serving_runtime_load_id(self) -> str:
        """Return the last successfully published serving-runtime load ID.

        Failed swaps can consume a backend counter before raising, so this
        caches only completed publications. Reef recovery uses the value to
        reconcile the serving engine with its recovered head.
        """
        return self._runtime_load_id

    def _next_runtime_load_id(self) -> str:
        """Allocate the next serving identity independently of a sender attempt."""
        current = RuntimeLoadId.parse(self._runtime_load_id)
        return str(RuntimeLoadId(current.incarnation, current.sequence + 1))

    def _publication_target(self, marker: Mapping[str, Any] | None) -> str:
        if marker is not None:
            target = marker.get("target_runtime_load_id")
            if isinstance(target, str) and target:
                return target
        target = self._next_runtime_load_id()
        if marker is not None:
            # Persist intent before any bytes leave the sender. An uncertain
            # partial transfer and process restart must reuse the same target.
            write_marker(self._marker_path(), {**marker, "target_runtime_load_id": target})
        return target

    def _update_serving(
        self, *, force_full: bool = False, scenario: str | None = None, runtime_load_id: str | None = None
    ) -> str:
        """Publish the group's weights; ``scenario`` names the adapter a LoRA publication belongs to.

        A per-scenario adapter publication loads a new versioned name into
        every engine, so the residency manager frees a slot first (evicting
        the publishing scenario's own current revision when nothing else
        fits: generation is paused, so no request observes the gap) and
        records the published revision afterwards.

        Admission runs before any weight leaves the trainer. A capacity
        rejection therefore means nothing was published and every engine still
        serves what it served, so it must not terminate them — that took down
        scenarios which were never part of the publication (#65). An eviction
        the engine refused is the opposite: its state is uncertain, so the
        terminate-and-recover path stays (#61).
        """
        residency = self._residency if scenario is not None else None
        try:
            self._phase = "publishing"
            target = runtime_load_id or self._next_runtime_load_id()
            if residency is not None and scenario is not None:
                residency.make_room(scenario, self._adapter_engine, supersede=True)
            if self._colocate:
                # Repeatable release also covers retry after a partial receive:
                # a released sender may need to reconstruct GPU workers.
                self._inference.offload(self._release_tags)
            self._training.prepare_weights(target, force_full=force_full)
            if self._colocate and self._release_tags is None:
                self._inference.onload_weights()
            raw_version = self._training.send_weights(target, force_full=force_full)
            if raw_version != target:
                raise RuntimeError(f"weight sender returned runtime load ID {raw_version!r}; expected {target!r}")
            if self._colocate:
                self._inference.onload_kv()
            observed = [str(value) for value in self._inference.runtime_load_ids()]
            if not observed or set(observed) != {raw_version}:
                raise RuntimeError(f"serving engines disagree after update: {observed!r}")
            if residency is not None and scenario is not None:
                residency.register(scenario, raw_version)
        except AdapterEvictionFailed:
            self._phase = "weight_sync_failed"
            with suppress(Exception):
                self._inference.abort()
            raise
        except AdapterCapacityExhausted:
            # Admission was refused before any weight left the trainer.
            self._phase = "serving"
            raise
        except BaseException:
            self._phase = "weight_sync_failed"
            with suppress(Exception):
                self._inference.abort()
            raise
        self._runtime_load_id = raw_version
        return raw_version

    def _pause_generation(self, *, reconcile: bool = False) -> None:
        if self._generation_paused and not reconcile:
            return
        self._inference.pause()
        self._generation_paused = True

    def _continue_generation(self) -> None:
        if not self._generation_paused:
            return
        self._inference.resume()
        self._generation_paused = False

    def _marker_path(self) -> Path:
        if self._save_hf_template is None:
            raise RuntimeError("training checkpoint path is not configured")
        return marker_path(self._save_hf_template)

    def _recover_marker(self) -> dict[str, Any] | None:
        marker = self._execution.recover()
        if marker is None:
            return None
        self._next_rollout_id = max(self._next_rollout_id, marker["rollout_id"] + 1)
        return marker


class _ScheduledTrainingBackend:
    """Apply Reef's resource barrier only after a backend admits the job."""

    def __init__(self, coordinator: TrainingCoordinator) -> None:
        self._coordinator = coordinator

    @contextmanager
    def prepare(
        self, payload: Mapping[str, Any], *, job_id: str, rollout_id: int, prior_marker: Mapping[str, Any] | None
    ) -> Iterator[PreparedTrainingJob | TrainingJobResult]:
        coordinator = self._coordinator
        scenario = payload.get("scenario") if coordinator._history is not None else None
        window = max_staleness(payload)
        metrics: Mapping[str, Any] = {}
        serving = coordinator._runtime_load_id
        decision = None
        if coordinator._history is not None:
            if not isinstance(scenario, str) or not scenario:
                raise ValueError("per-scenario LoRA training jobs must name their scenario")
            decision = _scenario_staleness_admission(
                payload,
                scenario=scenario,
                history=coordinator._history,
                serving_runtime_load_id=serving,
                max_staleness=window,
            )
        elif uses_staleness_admission(payload):
            if payload.get("expected_runtime_load_id") != serving:
                versions = [value for group in _admission_runtime_load_id_groups(payload) for value in group]
                decision = _stale_drop_decision(
                    payload,
                    serving_runtime_load_id=serving,
                    producing_runtime_load_ids=versions,
                    reason="execution_fence_mismatch",
                )
            else:
                decision = _staleness_admission(payload, serving_runtime_load_id=serving, max_staleness=window)
        elif payload.get("expected_runtime_load_id") != serving:
            yield TrainingJobResult(outcome="stale", runtime_load_id=serving)
            return
        if decision is not None:
            if decision.action == "drop":
                yield TrainingJobResult(outcome="stale", runtime_load_id=serving, metrics=decision.metrics)
                return
            metrics = decision.metrics
        with self._coordinator._training.prepare(
            payload, job_id=job_id, rollout_id=rollout_id, prior_marker=prior_marker
        ) as prepared:
            if isinstance(prepared, TrainingJobResult):
                yield prepared
            else:
                yield _ScheduledTrainingJob(self._coordinator, prepared, metrics)


class _ScheduledTrainingJob:
    def __init__(
        self, coordinator: TrainingCoordinator, prepared: PreparedTrainingJob, admission_metrics: Mapping[str, Any]
    ) -> None:
        self._coordinator = coordinator
        self._prepared = prepared
        self._admission_metrics = admission_metrics

    @property
    def checkpoint(self) -> TrainingCheckpoint:
        return self._prepared.checkpoint

    def train(self) -> TrainingMetrics:
        self._coordinator._prepare_training()
        metrics = self._prepared.train()
        return TrainingMetrics(training=metrics.training, durable={**self._admission_metrics, **metrics.durable})

    def save_checkpoint(self) -> None:
        self._prepared.save_checkpoint()


class _CoordinatedWeightPublisher:
    """Reef publication operations across independent sender and receiver interfaces."""

    def __init__(self, bridge: TrainingCoordinator) -> None:
        self._bridge = bridge

    def recover(self, marker: Mapping[str, Any] | None) -> None:
        bridge = self._bridge
        bridge._inference.recover()
        if bridge._history is not None:
            # Replacement engines boot without adapters. Restore other scenarios
            # before this job's complete transfer, and release dead residency slots.
            bridge._require_residency().reconcile((), bridge._adapter_engine)
            bridge._recover_scenario_adapters(marker)

    def pause(self) -> None:
        # Reassert the owner barrier even if the bridge cached a prior pause;
        # replacement controllers/engines may not have observed that RPC.
        self._bridge._pause_generation(reconcile=True)

    def republish(self, runtime_load_id: str, marker: Mapping[str, Any] | None) -> str:
        bridge = self._bridge
        try:
            return bridge._update_serving(
                force_full=True, scenario=bridge._marker_scenario(marker), runtime_load_id=runtime_load_id
            )
        finally:
            # A failed/mismatched transfer must not replace the retry identity.
            bridge._runtime_load_id = runtime_load_id

    def publish(self, marker: Mapping[str, Any], *, force_full: bool) -> str:
        bridge = self._bridge
        if bridge._history is not None:
            bridge._training.activate_scenario(str(marker["scenario"]))
        published = bridge._update_serving(
            force_full=force_full,
            scenario=bridge._marker_scenario(marker),
            runtime_load_id=bridge._publication_target(marker),
        )
        if bridge._history is not None:
            scenario = str(marker["scenario"])
            bridge._history.record_publication(scenario, published, adapter_name(scenario, published))
        return published

    def resume(self) -> None:
        self._bridge._continue_generation()

    def restore_incumbent(self) -> None:
        self._bridge._restore_incumbent_serving()

    def abort(self) -> None:
        self._bridge._inference.abort()
