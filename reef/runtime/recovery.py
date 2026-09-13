"""Durable runtime state, startup reconstruction, and inference failure recovery."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread
from time import monotonic
from typing import Any, Literal

from reef.runtime.interfaces import (
    EngineHealthChecks,
    InferenceEngines,
    InferenceMonitor,
    MarkerStatus,
    RuntimeLoadId,
    ScenarioHistoryStore,
    TrainingJobResult,
    TrainingJobStore,
    WeightUpdateConnection,
)
from reef.runtime.publication import BackendWeightPublisher, TrainingPublication


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    mkdir_durable(path.parent)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name.lstrip('.')}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(value, file, allow_nan=False, separators=(",", ":"), sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def mkdir_durable(path: Path) -> None:
    missing = []
    current = path
    while not current.exists():
        if current.is_symlink():
            raise OSError(f"refusing symlinked directory: {current}")
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise OSError(f"unsafe directory: {current}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if directory.is_symlink() or not directory.is_dir():
                raise OSError(f"unsafe directory: {directory}") from None
        fsync_dir(directory.parent)


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


write_marker = write_json


LATEST_JOB_MARKER_FILENAME = ".reef-latest-job.json"


MarkerDisposition = Literal["replay", "resume", "conflict", "fresh"]


_MARKER_TRANSITIONS: dict[MarkerStatus, frozenset[MarkerStatus]] = {
    "RUNNING": frozenset({"CHECKPOINT"}),
    "CHECKPOINT": frozenset({"UPDATING_WEIGHTS", "REJECTING"}),
    "UPDATING_WEIGHTS": frozenset({"READY_TO_COMMIT"}),
    "READY_TO_COMMIT": frozenset({"HEAD_COMMITTED"}),
    "HEAD_COMMITTED": frozenset({"COMPLETE"}),
    "COMPLETE": frozenset(),
    "REJECTING": frozenset({"REJECTED"}),
    "REJECTED": frozenset(),
}


def marker_path(hf_template: str) -> Path:
    """The single marker location derived from the HF checkpoint template."""
    return Path(hf_template.format(rollout_id=0)).expanduser().parent / LATEST_JOB_MARKER_FILENAME


def read_marker(path: Path) -> dict[str, Any] | None:
    try:
        value = read_json(path)
    except ValueError as exc:
        raise RuntimeError(f"invalid training marker: {path}") from exc
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("status") not in {
        "RUNNING",
        "CHECKPOINT",
        "UPDATING_WEIGHTS",
        "READY_TO_COMMIT",
        "HEAD_COMMITTED",
        "COMPLETE",
        "REJECTING",
        "REJECTED",
    }:
        raise RuntimeError(f"invalid training marker: {path}")
    if (
        not isinstance(value.get("job_id"), str)
        or not value["job_id"]
        or not isinstance(value.get("rollout_id"), int)
        or isinstance(value["rollout_id"], bool)
        or value["rollout_id"] < 0
    ):
        raise RuntimeError(f"invalid training marker: {path}")
    commit_acknowledged = value.get("commit_acknowledged")
    if commit_acknowledged is not None and not isinstance(commit_acknowledged, bool):
        raise RuntimeError(f"invalid training marker commit acknowledgement: {path}")
    if "target_runtime_load_id" in value and (
        not isinstance(value["target_runtime_load_id"], str) or not value["target_runtime_load_id"]
    ):
        raise RuntimeError(f"invalid training marker target runtime load ID: {path}")
    if value["status"] == "HEAD_COMMITTED" and commit_acknowledged is not True:
        raise RuntimeError(f"training marker state requires a commit acknowledgement: {path}")
    if value["status"] != "RUNNING" and (
        not isinstance(value.get("checkpoint_path"), str)
        or not value["checkpoint_path"]
        or Path(value["checkpoint_path"]).is_symlink()
        or not Path(value["checkpoint_path"]).is_dir()
    ):
        raise RuntimeError(f"training marker has no checkpoint: {path}")
    if value["status"] in {"READY_TO_COMMIT", "HEAD_COMMITTED", "COMPLETE"} and (
        not isinstance(value.get("runtime_load_id"), str) or not value["runtime_load_id"]
    ):
        raise RuntimeError(f"invalid training marker: {path}")
    return value


def transition_marker(
    path: Path,
    marker: dict[str, Any],
    status: MarkerStatus,
    **updates: Any,
) -> dict[str, Any]:
    """Durably advance one edge of the training-job state machine."""
    current = marker.get("status")
    if current not in _MARKER_TRANSITIONS or status not in _MARKER_TRANSITIONS[current]:
        raise RuntimeError(f"invalid training marker transition {current!r} -> {status!r}")
    updated = {**marker, **updates, "status": status}
    write_marker(path, updated)
    marker.update(updated)
    return marker


def marker_disposition(marker: Mapping[str, Any] | None, job_id: str) -> MarkerDisposition:
    """Classify a recovered marker against an incoming job identity.

    - ``replay``: the same job already completed; return its recorded result.
    - ``resume``: the same job trained and checkpointed; only the serving
      publication remains.
    - ``conflict``: a different job is mid-flight; operator recovery required.
    - ``fresh``: nothing blocks running this job from the start.
    """
    status = None if marker is None else marker["status"]
    match (marker is not None and marker["job_id"] == job_id, status):
        case (True, "READY_TO_COMMIT" | "HEAD_COMMITTED" | "COMPLETE"):
            return "replay"
        case (True, "CHECKPOINT" | "REJECTED"):
            return "resume"
        case (
            _,
            "RUNNING" | "CHECKPOINT" | "UPDATING_WEIGHTS" | "READY_TO_COMMIT" | "HEAD_COMMITTED" | "REJECTING",
        ):
            return "conflict"
        case _:
            return "fresh"


def marker_result(marker: Mapping[str, Any], *, metrics: Mapping[str, Any] | None = None) -> TrainingJobResult:
    merged_metrics = _marker_metrics(marker)
    if metrics:
        merged_metrics.update(metrics)
    return TrainingJobResult(
        outcome="complete",
        runtime_load_id=str(marker["runtime_load_id"]),
        checkpoint_path=str(marker["checkpoint_path"]),
        metrics=merged_metrics or None,
        training_job_id=str(marker["job_id"]),
    )


def marker_checkpoint_result(marker: Mapping[str, Any]) -> TrainingJobResult:
    """Return a typed result for a job waiting on its serving-weight update."""
    return TrainingJobResult(
        outcome="checkpoint",
        runtime_load_id=str(marker.get("runtime_load_id", "pending")),
        checkpoint_path=str(marker["checkpoint_path"]),
        metrics=_marker_metrics(marker) or None,
        training_job_id=str(marker["job_id"]),
    )


def marker_rollouts(marker: Mapping[str, Any] | None) -> set[int]:
    rollout_id = None if marker is None else marker.get("rollout_id")
    return {rollout_id} if isinstance(rollout_id, int) else set()


def _marker_metrics(marker: Mapping[str, Any]) -> dict[str, Any]:
    """Merge durable method telemetry with backend training metrics."""
    durable = marker.get("metrics")
    worker = marker.get("train_metrics")
    return {
        **(dict(durable) if isinstance(durable, Mapping) else {}),
        **(dict(worker) if isinstance(worker, Mapping) else {}),
    }


SCENARIO_HISTORY_FILENAME = "reef_scenarios.json"


HISTORY_FILENAME = SCENARIO_HISTORY_FILENAME


HISTORY_FORMAT = 1


class ScenarioHistory(ScenarioHistoryStore):
    """Publication history and latest checkpoint per training scenario."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, dict[str, Any]] = {}
        value = read_json(path)
        if value is None:
            return
        if not isinstance(value, Mapping) or value.get("format") != HISTORY_FORMAT:
            raise RuntimeError(f"invalid scenario history: {path}")
        scenarios = value.get("scenarios")
        if not isinstance(scenarios, Mapping):
            raise RuntimeError(f"invalid scenario history: {path}")
        for scenario, entry in scenarios.items():
            if not isinstance(scenario, str) or not scenario or not isinstance(entry, Mapping):
                raise RuntimeError(f"invalid scenario history entry: {path}")
            publications = entry.get("publications", [])
            if not isinstance(publications, list) or not all(isinstance(item, str) for item in publications):
                raise RuntimeError(f"invalid scenario history publications: {path}")
            self._entries[scenario] = {
                "publications": list(publications),
                "adapter": entry.get("adapter"),
                "rollout_id": entry.get("rollout_id"),
                "steps": int(entry.get("steps", 0)),
            }

    @property
    def path(self) -> Path:
        return self._path

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def entry(self, scenario: str) -> Mapping[str, Any] | None:
        entry = self._entries.get(scenario)
        return None if entry is None else dict(entry)

    def adapter(self, scenario: str) -> str | None:
        entry = self._entries.get(scenario)
        adapter = None if entry is None else entry.get("adapter")
        return adapter if isinstance(adapter, str) and adapter else None

    def last_publication(self, scenario: str) -> str | None:
        entry = self._entries.get(scenario)
        if entry is None or not entry["publications"]:
            return None
        return str(entry["publications"][-1])

    def lag(self, scenario: str, producing: RuntimeLoadId) -> int | None:
        """How many of ``scenario``'s publications postdate ``producing``.

        ``None`` means the producing version belongs to another incarnation
        (a previous training-group lifetime), which is never admissible.
        """
        entry = self._entries.get(scenario)
        publications = [] if entry is None else entry["publications"]
        lag = 0
        for value in publications:
            published = RuntimeLoadId.parse(value)
            if published.incarnation != producing.incarnation:
                continue
            if published.sequence > producing.sequence:
                lag += 1
        if publications:
            latest = RuntimeLoadId.parse(publications[-1])
            if latest.incarnation != producing.incarnation:
                return None
        return lag

    def protected_rollouts(self) -> set[int]:
        """Global rollout ids whose checkpoints carry a scenario's latest adapter."""
        return {
            int(entry["rollout_id"])
            for entry in self._entries.values()
            if isinstance(entry.get("rollout_id"), int) and not isinstance(entry["rollout_id"], bool)
        }

    def record_checkpoint(self, scenario: str, rollout_id: int) -> None:
        entry = self._entries.setdefault(
            scenario, {"publications": [], "adapter": None, "rollout_id": None, "steps": 0}
        )
        entry["rollout_id"] = int(rollout_id)
        entry["steps"] = int(entry.get("steps", 0)) + 1
        self._write()

    def record_publication(self, scenario: str, runtime_load_id: str, adapter: str) -> None:
        entry = self._entries.setdefault(
            scenario, {"publications": [], "adapter": None, "rollout_id": None, "steps": 0}
        )
        if not entry["publications"] or entry["publications"][-1] != runtime_load_id:
            entry["publications"].append(runtime_load_id)
        entry["adapter"] = adapter
        self._write()

    def status(self) -> dict[str, dict[str, Any]]:
        return {
            scenario: {
                "runtime_load_id": self.last_publication(scenario),
                "adapter": self.adapter(scenario),
                "publications": len(entry["publications"]),
                "rollout_id": entry.get("rollout_id"),
                "steps": entry.get("steps", 0),
            }
            for scenario, entry in sorted(self._entries.items())
        }

    def _write(self) -> None:
        write_json(self._path, {"format": HISTORY_FORMAT, "scenarios": self._entries})


def history_path(hf_template: str) -> Path:
    """The history sits beside the job marker, in the HF checkpoint directory."""
    return Path(hf_template.format(rollout_id=0)).expanduser().parent / HISTORY_FILENAME


logger = logging.getLogger(__name__)


class InferenceControl:
    """Coordinate pause, recovery and reconnect without routing weight tensors.

    Calls must be serialized by the owning actor. A successful recovery of a
    paused publication leaves both generation and monitoring paused. Only the
    training publication coordinator may resume after the durable commit gate.
    """

    def __init__(
        self,
        engines: InferenceEngines,
        connection: WeightUpdateConnection,
        monitor: InferenceMonitor,
    ) -> None:
        self._engines = engines
        self._connection = connection
        self._monitor = monitor
        self.paused = False
        self.reconnect_required = False

    def pause(self) -> object:
        # Preserve pause intent even if only some engines cross the barrier.
        self.paused = True
        self._monitor.pause()
        return self._engines.pause()

    def resume(self) -> object:
        result = self._engines.resume()
        self.paused = False
        self._monitor.resume()
        return result

    def terminate(self) -> int:
        self.paused = True
        self._monitor.pause()
        return self._engines.terminate() if self._engines.owned else 0

    def recover(self) -> None:
        try:
            self._monitor.pause()
            try:
                usable = self._connection.is_usable()
            except Exception:
                usable = False
            if not usable:
                if not self._engines.owned:
                    raise RuntimeError("uncertain external weight update requires restarting the external deployment")
                self._connection.replace()
                self.reconnect_required = True
            self._engines.recover()
            if self.paused:
                self._engines.pause()
            else:
                self._monitor.resume()
        except BaseException:
            # A failed recovery cannot authorize a later legacy monitor-resume
            # call to replace engines behind an unfinished publication.
            self.paused = True
            raise

    def prepare_training_connection(self) -> None:
        """Fence a new trainer attachment even when all engines are healthy.

        The deployment owner must retire the previous training workers first.
        This is a serialized attachment handshake, not leader election. Engine
        and lock recovery keep their existing ownership rules.
        """
        self.paused = True
        self.reconnect_required = True
        self.recover()

    def acknowledge_reconnect(self) -> None:
        """Called only after training workers have attached to the current targets."""
        self.reconnect_required = False


@dataclass(frozen=True)
class HealthMonitorConfig:
    """Probe timings in seconds, independent of an inference framework."""

    interval: float
    timeout: float
    first_wait: float = 0

    def __post_init__(self) -> None:
        for name, value in (("interval", self.interval), ("timeout", self.timeout), ("first_wait", self.first_wait)):
            if not math.isfinite(value) or value < 0 or (name != "first_wait" and value == 0):
                raise ValueError(
                    f"health monitor {name} must be finite and {'non-negative' if name == 'first_wait' else 'positive'}"
                )


class EngineHealthMonitor(InferenceMonitor):
    """Schedule probes; pause/stop drain any active probe or retirement.

    The owner serializes lifecycle calls. Engine replacement and offload must
    wait for a successful pause. A timeout leaves scheduling disabled and must
    prevent replacement; it never claims the in-flight operation was cancelled.
    Backend targets must bound their check/retire operations. No model framework
    or executor is imported here.
    """

    def __init__(self, checks: EngineHealthChecks, config: HealthMonitorConfig) -> None:
        self._checks = checks
        self._config = config
        self._condition = Condition()
        self._thread: Thread | None = None
        self._enabled = False
        self._stopped = False
        self._active = False
        self._epoch = 0
        self._next_check = 0.0
        self._failure: BaseException | None = None

    def start(self) -> bool:
        """Start once, initially paused; resume explicitly after engine readiness."""
        with self._condition:
            self._raise_failure()
            if self._stopped:
                raise RuntimeError("health monitor is stopped")
            if self._thread is not None:
                return False
            self._thread = Thread(target=self._run, name="reef-engine-health", daemon=True)
            try:
                self._thread.start()
            except BaseException:
                self._thread = None
                raise
            return True

    def resume(self) -> None:
        with self._condition:
            self._raise_failure()
            if self._thread is None or self._stopped:
                raise RuntimeError("health monitor is not running")
            self._epoch += 1
            self._enabled = True
            self._next_check = monotonic() + self._config.first_wait
            self._condition.notify_all()

    def _drain_deadline(self, timeout: float | None) -> float:
        seconds = 2 * self._config.timeout + 1 if timeout is None else timeout
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("health monitor drain timeout must be finite and non-negative")
        return monotonic() + seconds

    def _disable_and_drain(self, deadline: float) -> None:
        # Called with the condition held; wait releases it so probe completion
        # can observe the epoch change and discard its now-stale failure.
        self._enabled = False
        self._epoch += 1
        self._condition.notify_all()
        while self._active:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("health monitor still has an in-flight probe or retirement")
            self._condition.wait(remaining)

    def pause(self, timeout: float | None = None) -> None:
        deadline = self._drain_deadline(timeout)
        with self._condition:
            self._disable_and_drain(deadline)

    def stop(self, timeout: float | None = None) -> None:
        """Drain and join; retain a live thread handle if shutdown times out."""
        deadline = self._drain_deadline(timeout)
        with self._condition:
            self._stopped = True
            self._disable_and_drain(deadline)
            thread = self._thread
        if thread is not None:
            thread.join(max(0.0, deadline - monotonic()))
            if thread.is_alive():
                raise TimeoutError("health monitor thread has not stopped")
            with self._condition:
                self._thread = None

    def is_checking_enabled(self) -> bool:
        with self._condition:
            return self._enabled and not self._stopped and self._failure is None

    def check_health(self) -> None:
        with self._condition:
            self._raise_failure()
            if self._thread is None or self._stopped:
                raise RuntimeError("health monitor is not running")

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError("engine health monitor failed") from self._failure

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._stopped:
                        delay = self._next_check - monotonic()
                        if self._enabled and delay <= 0:
                            break
                        self._condition.wait(max(0, delay) if self._enabled else None)
                    if self._stopped:
                        return
                    epoch = self._epoch
                    self._active = True
                try:
                    self._check_targets(epoch)
                except BaseException as exc:
                    # Make failure visible before releasing the drain barrier.
                    with self._condition:
                        self._failure = exc
                        self._enabled = False
                    raise
                finally:
                    with self._condition:
                        self._active = False
                        if self._epoch == epoch:
                            self._next_check = monotonic() + self._config.interval
                        self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._failure = exc
                self._enabled = False
                self._condition.notify_all()
            logger.exception("Engine health monitor stopped after an internal failure")

    def _check_targets(self, epoch: int) -> None:
        for target in self._checks.targets():
            with self._condition:
                if not self._enabled or self._stopped or self._epoch != epoch:
                    return
            try:
                target.check(self._config.timeout)
            except Exception:
                with self._condition:
                    if not self._enabled or self._stopped or self._epoch != epoch:
                        return
                # _active remains true through retirement. A concurrent pause
                # cannot return until all mutations to this snapshot finish.
                logger.warning("Inference engine health probe failed; retiring the captured engine")
                target.retire(self._config.timeout)


class TrainingRecovery:
    """Restore training state and serving weights behind the durable commit gate."""

    def __init__(self, publication: TrainingPublication, publisher: BackendWeightPublisher) -> None:
        self.publication = publication
        self.publisher = publisher

    def restore(self, marker: dict[str, Any] | None) -> str:
        if marker is not None:
            context = self.publisher.context
            context.next_rollout_id = max(context.next_rollout_id, marker["rollout_id"] + 1)
        with self.publication.recovery(marker):
            return self.restore_serving(marker)

    def restore_serving(self, marker: dict[str, Any] | None) -> str:
        """Reconstruct backend state inside Reef's startup recovery barrier."""
        publisher = self.publisher
        marker_status = None if marker is None else str(marker["status"])
        if marker_status == "UPDATING_WEIGHTS":
            # The previous fan-out may have updated only some engines. Recover
            # dead actors first, keep every engine paused, and force a complete
            # tensor transfer from the durable checkpoint-backed actor state.
            publisher.inference.recover()
        if publisher.config.colocate and publisher.config.lora:
            # Cold startup releases everything before training initializes.
            # Restore the frozen base before registering scenario adapters,
            # including runs that only release KV/graphs on later steps.
            publisher.inference.onload_weights()
        if marker_status == "REJECTING":
            if marker is None:
                raise RuntimeError("REJECTING marker status has no marker payload")
            self.publication.reject(str(marker["job_id"]))
            marker["status"] = marker_status = "REJECTED"
            publisher.pause_generation(reconcile=True)
        inference_url = publisher.inference.inference_url()
        versions = publisher.inference.runtime_load_ids()
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
        if publisher.history is not None:
            publisher.restore_scenario_adapters(marker)
        if (
            publisher.config.save_hf_template is not None
            and marker_status != "REJECTED"
            and not (publisher.config.lora and marker is None)
        ):
            # The Megatron checkpoint can be newer than the HF checkpoint used
            # to boot inference. Publish actor weights before construction returns
            # so the first Reef inference uses the actual training version. A
            # LoRA bridge that never trained has nothing to publish: the frozen
            # base inference booted from is exactly what every fresh adapter
            # computes, and the history replay above restored trained ones.
            publisher.runtime_load_id = publisher.update_serving(
                # A fresh sender may only capture a delta baseline unless a
                # real full transfer is requested. Startup must load bytes
                # and acknowledge Reef's identity on every receiver.
                force_full=True,
                scenario=publisher.marker_scenario(marker),
                runtime_load_id=recovered_runtime_load_id or publisher.publication_target(marker),
            )
        elif publisher.config.lora and marker is None:
            # Nothing to publish, but the engines still need Reef's canonical
            # version token (they boot with a backend default), and colocated
            # engines boot released: give them their weights and KV back
            # before the first request.
            if publisher.config.colocate:
                publisher.inference.onload_weights()
                publisher.inference.onload_kv()
            publisher.initialize_version()
        elif marker_status == "REJECTED":
            if marker is None:
                raise RuntimeError("REJECTED status requires a durable job marker")
            publisher.runtime_load_id = str(marker.get("parent_runtime_load_id") or versions[0])
            publisher.initialize_version()
        elif publisher.config.save_hf_template is None:
            publisher.initialize_version()
        self.publication.finish_recovery(marker, publisher.runtime_load_id)

        return inference_url


class FileTrainingJobStore(TrainingJobStore):
    """Atomically persist job transitions before exposing their updated state."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> dict[str, Any] | None:
        return read_marker(self.path)

    def write(self, marker: Mapping[str, Any]) -> None:
        write_marker(self.path, marker)

    def transition(self, marker: dict[str, Any], status: MarkerStatus, **updates: Any) -> dict[str, Any]:
        return transition_marker(self.path, marker, status, **updates)
