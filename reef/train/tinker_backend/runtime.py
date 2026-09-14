"""Tinker's checkpoint store and training runtime.

One deployment shares a :class:`TinkerCheckpointStore`: the local manifests
under ``state_dir`` and the immutable remote snapshots they reference, each
served under a runtime load ID of this process's incarnation. The training
runtime here branches candidates from the store's active checkpoint; the
inference runtime in :mod:`reef.train.tinker_backend.inference` samples from
the snapshot a frozen artifact resolves to and binds the head Reef publishes.
Neither runtime holds the other.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any

from reef.artifact.artifact import Artifact, LiveWeightArtifactRef
from reef.core.batches import TrainingBatch
from reef.core.evaluation import SelectionDecision
from reef.runtime.interfaces import ModelCandidate, PreparedTrainingStep, StaleCandidate, TrainingRuntime
from reef.train.tinker_backend.checkpoint import MANIFEST, TinkerCheckpoint
from reef.train.tinker_backend.client import TinkerClient
from reef.train.tinker_backend.config import TinkerConfig
from reef.train.tinker_backend.losses import resolve_tinker_loss, row_from_payload
from reef.train.tinker_backend.preparation import prepare_tinker_step


class TinkerCheckpointStore:
    """Local manifests and the remote snapshots one Tinker deployment trains from and serves.

    ``active`` is the checkpoint the next candidate branches from and new
    requests sample; a selected candidate stays ``pending`` until Reef commits
    its training job. Each snapshot is remembered once per incarnation, and a
    rollback to an older release mints a new runtime load ID. Reef's commit
    log remains authoritative; the store only follows it.
    """

    def __init__(self, base_model: str, config: TinkerConfig, client: TinkerClient) -> None:
        self.lock = RLock()
        self.model = base_model
        self.config = config
        self.client = client
        self.root = Path(config.state_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._state_lock = (self.root / ".lock").open("a")
        try:
            fcntl.flock(self._state_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._state_lock.close()
            raise ValueError("Tinker state_dir is already owned by another runtime") from None
        self._incarnation = uuid.uuid4().hex
        self._snapshots: dict[str, tuple[TinkerCheckpoint, str]] = {}
        self._loads: dict[str, TinkerCheckpoint] = {}
        self._releases: dict[str, tuple[TinkerCheckpoint, str]] = {}
        self.active_release: str | None = None
        #: Trained candidates by identity, with the incumbent version each branched from.
        self.candidates: dict[str, tuple[ModelCandidate, str]] = {}
        #: The activated candidate whose Reef commit has not been acknowledged yet.
        self.pending: ModelCandidate | None = None
        self._closed = False
        try:
            base = self.root / "base"
            if (base / MANIFEST).exists():
                self.base = TinkerCheckpoint.read(base)
                self.base.validate_model(base_model, config.lora_rank)
            else:
                self.base = client.initialize()
                self.base.validate_model(base_model, config.lora_rank)
                self.base.write(base)
            self.active, self.version = self.remember(self.base)
        except BaseException:
            self._state_lock.close()
            raise

    def remember(self, checkpoint: TinkerCheckpoint) -> tuple[TinkerCheckpoint, str]:
        """The load ID a snapshot already serves under in this incarnation, else a new one."""
        checkpoint.validate_model(self.model, self.config.lora_rank)
        existing = self._snapshots.get(checkpoint.sampler_path)
        if existing is not None:
            if existing[0] != checkpoint:
                raise ValueError("Tinker sampler path cannot identify different training checkpoints")
            return existing
        value = self.new_load(checkpoint)
        self._snapshots[checkpoint.sampler_path] = value
        return value

    def new_load(self, checkpoint: TinkerCheckpoint) -> tuple[TinkerCheckpoint, str]:
        version = f"{self._incarnation}:{len(self._loads)}"
        self._loads[version] = checkpoint
        return checkpoint, version

    def snapshot(self, artifact: Artifact) -> tuple[TinkerCheckpoint, str]:
        """Resolve exactly the frozen artifact; old in-flight calls keep their snapshot."""
        with self.lock:
            if isinstance(artifact.ref, LiveWeightArtifactRef):
                version = artifact.ref.runtime_load_id
                if version in self._loads:
                    return self._loads[version], version
                raise ValueError("Tinker live snapshot belongs to an unavailable serving incarnation")
            if artifact.ref.release_id in self._releases:
                return self._releases[artifact.ref.release_id]
            materialized = artifact.materialize()
            path = materialized.local_path
            if path is None:
                raise ValueError("Tinker requires a materialized checkpoint manifest")
            if (path / MANIFEST).exists():
                selected = self.remember(TinkerCheckpoint.read(path))
                self._releases[artifact.ref.release_id] = selected
                return selected
            # Reef's empty base tree represents the initial seeded adapter.
            # An arbitrary uploaded weight directory must not silently become base.
            contents = {entry.name for entry in path.iterdir()} - {".git", ".gitattributes", "reef-artifact.json"}
            if contents:
                raise ValueError("artifact is missing tinker-checkpoint.json")
            selected = self.remember(self.base)
            self._releases[artifact.ref.release_id] = selected
            return selected

    def bind(self, checkpoint: TinkerCheckpoint, version: str, release_id: str) -> None:
        """Serve ``checkpoint`` as the published release ``release_id``."""
        self.active, self.version = checkpoint, version
        self.active_release = release_id
        self._releases[release_id] = (checkpoint, version)

    def known_candidate(self, candidate: ModelCandidate) -> tuple[ModelCandidate, str]:
        """The trained candidate ``candidate`` names, with the version it branched from."""
        known = self.candidates.get(candidate.candidate_id)
        if known is None or known[0].checkpoint_path != candidate.checkpoint_path:
            raise ValueError("unknown Tinker candidate")
        return known

    def candidate_directory(self, identity: str) -> Path:
        return self.root / "candidates" / identity

    def close(self) -> None:
        """Close the SDK client and release the state directory; safe to repeat."""
        with self.lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.client.close()
        finally:
            self._state_lock.close()


class TinkerTrainingRuntime(TrainingRuntime):
    """Train candidates from the store's active checkpoint in separate remote sessions.

    Each attempt branches from the incumbent's weights AND optimizer.
    Repeating a failed attempt can consume API resources, but cannot apply its
    gradient twice to the incumbent.
    """

    def __init__(self, store: TinkerCheckpointStore) -> None:
        self._store = store

    def prepare_training_step(
        self,
        batch: TrainingBatch,
        step_preparer: str,
        algorithm_state: Mapping[str, Any],
        scenario_step: int,
        *,
        serving_runtime_load_id: str | None = None,
    ) -> PreparedTrainingStep:
        with self._store.lock:
            version = serving_runtime_load_id or self._store.version
        return prepare_tinker_step(
            batch,
            step_preparer,
            algorithm_state,
            scenario_step,
            runtime_load_id=version,
            batch_size=self._store.config.batch_size,
        )

    def train_candidate(self, payload: Mapping[str, Any]) -> ModelCandidate:
        store = self._store
        with store.lock:
            if store.pending is not None:
                raise RuntimeError("Tinker is waiting for the previous candidate's Reef commit")
            if payload["source_runtime_load_id"] != store.version or payload.get("stale"):
                raise StaleCandidate({"tinker_stale_samples": 1})
            incumbent, version = store.active, store.version
            identity = hashlib.sha256(json.dumps(dict(payload), sort_keys=True, allow_nan=False).encode()).hexdigest()
            known = store.candidates.get(identity)
            if known is not None:
                return known[0]
        loss = resolve_tinker_loss(payload["loss"])
        batches = [[row_from_payload(row) for row in batch] for batch in payload["batches"]]
        checkpoint, metrics = store.client.train(incumbent, batches, loss)
        checkpoint.validate_model(store.model, store.config.lora_rank)
        directory = store.candidate_directory(identity)
        checkpoint.write(directory)
        candidate = ModelCandidate(
            candidate_id=identity,
            training_job_id=identity,
            checkpoint_path=str(directory),
            current_runtime_load_id=version,
            training_metrics=dict(metrics),
            metadata={"scenario_step": payload["scenario_step"]},
        )
        with store.lock:
            if store.active != incumbent or store.version != version:
                raise StaleCandidate({"tinker_stale_samples": 1})
            store.remember(checkpoint)
            store.candidates[identity] = (candidate, version)
        return candidate

    def reject_candidate(self, candidate: ModelCandidate, decision: SelectionDecision) -> None:
        with self._store.lock:
            if self._store.pending is not None:
                raise RuntimeError("cannot reject an already activated Tinker candidate")
            self._store.known_candidate(candidate)
            self._store.candidates.pop(candidate.candidate_id)
            # The incumbent was never trained or swapped. Remote snapshots stay
            # available for the selection record and an external retention policy.

    def restore_checkpoint(self, artifact: Artifact) -> None:
        """Validate the rollback target; training state is whatever snapshot the head binds.

        Every candidate branches from the store's active checkpoint, which
        ``activate_checkpoint`` on the inference side points at the republished
        artifact once the rollback commit is final, optimizer state included.
        """
        self._store.snapshot(artifact)

    def shutdown(self) -> None:
        self._store.close()
