"""Synthetic CaseGraph -> Reef memory/harness adapter prototype.

The local state prototype is complemented by a thin Reef report/record bridge.
It does not implement a recipe or the observe/grow/commit lifecycle.
It never contains patient data, model prompts, or production service calls.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from itertools import pairwise
from typing import Any

FEEDBACK_TYPES = (
    "fact_error",
    "retrieval_omission",
    "execution_strategy_failure",
    "verifier_error",
)
UPDATE_SURFACES = ("episodic_memory", "procedural_harness")


def digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CaseEvent:
    """A synthetic, auditable event with temporal and feedback provenance."""

    case_id: str
    source_id: str
    event_type: str
    text_ref: str
    observed_at: str
    valid_from: str
    valid_to: str | None
    provenance: Mapping[str, Any]
    confidence: float
    feedback_type: str | None
    outcome: str
    update_surface: str
    artifact_version: str
    cost: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.case_id.startswith("syn-case-"):
            raise ValueError("fixture events must use synthetic case identifiers")
        if not self.source_id.startswith("synthetic-source-"):
            raise ValueError("fixture events must use synthetic source identifiers")
        if not self.text_ref.startswith("synthetic://"):
            raise ValueError("fixture events must reference synthetic text")
        if self.feedback_type not in (None, *FEEDBACK_TYPES):
            raise ValueError(f"unknown feedback type: {self.feedback_type}")
        if self.update_surface not in UPDATE_SURFACES:
            raise ValueError(f"unknown update surface: {self.update_surface}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        date.fromisoformat(self.observed_at)
        start = date.fromisoformat(self.valid_from)
        if self.valid_to is not None and date.fromisoformat(self.valid_to) < start:
            raise ValueError("valid_to must not precede valid_from")
        if self.provenance.get("kind") != "synthetic":
            raise ValueError("provenance.kind must be synthetic")

    def canonical(self) -> dict[str, Any]:
        value = asdict(self)
        value["provenance"] = dict(self.provenance)
        value["cost"] = dict(self.cost)
        return value

    @property
    def event_id(self) -> str:
        """Stable identity over the canonical event payload, excluding the ID."""

        return digest(self.canonical())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event with its explicit, verifiable ``event_id``."""

        value = self.canonical()
        value["event_id"] = self.event_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CaseEvent:
        """Deserialize and reject a tampered or stale event identity."""

        payload = dict(value)
        supplied_id = payload.pop("event_id", None)
        event = cls(**payload)
        if supplied_id is not None and supplied_id != event.event_id:
            raise ValueError("event_id does not match the canonical event payload")
        return event

    def serialize(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def deserialize(cls, value: str) -> CaseEvent:
        return cls.from_dict(json.loads(value))


class SyntheticCaseEventGenerator:
    """Generate deterministic cases partitionable by time and case identity."""

    def __init__(self, seed: int = 20260919, artifact_version: str = "casegraph-synthetic-v1") -> None:
        self.seed = seed
        self.artifact_version = artifact_version

    def generate(self, case_count: int = 8) -> tuple[CaseEvent, ...]:
        if case_count < 8:
            raise ValueError("the fixture needs at least 8 cases for all split arms")
        rng = random.Random(self.seed)
        base = date(2026, 1, 1)
        events: list[CaseEvent] = []
        for index in range(case_count):
            case_id = f"syn-case-{index:02d}"
            source_id = f"synthetic-source-{index:02d}"
            # Leave a one-day gap between cases so a feedback event cannot
            # share a temporal boundary with the next split's observation.
            observed = base + timedelta(days=2 * index)
            signal = rng.choice(("amber", "blue", "green"))
            provenance = {
                "kind": "synthetic",
                "generator": "reef_casegraph_adapter",
                "seed": self.seed,
                "source": "fixture",
            }
            events.append(
                CaseEvent(
                    case_id=case_id,
                    source_id=source_id,
                    event_type="observation",
                    text_ref=f"synthetic://{case_id}/observation/{signal}",
                    observed_at=observed.isoformat(),
                    valid_from=observed.isoformat(),
                    valid_to=None,
                    provenance=provenance,
                    confidence=0.90,
                    feedback_type=None,
                    outcome="observed",
                    update_surface="episodic_memory",
                    artifact_version=self.artifact_version,
                    cost={"compute_ms": 1.0, "reviewer_seconds": 0.0, "human_intervention_seconds": 0.0},
                )
            )
            feedback = FEEDBACK_TYPES[index % len(FEEDBACK_TYPES)]
            feedback_day = observed + timedelta(days=1)
            surface = "episodic_memory" if feedback in {"fact_error", "retrieval_omission"} else "procedural_harness"
            events.append(
                CaseEvent(
                    case_id=case_id,
                    source_id=source_id,
                    event_type="feedback",
                    text_ref=f"synthetic://{case_id}/feedback/{feedback}",
                    observed_at=feedback_day.isoformat(),
                    valid_from=feedback_day.isoformat(),
                    valid_to=None,
                    provenance=provenance,
                    confidence=0.95,
                    feedback_type=feedback,
                    outcome="corrective",
                    update_surface=surface,
                    artifact_version=self.artifact_version,
                    cost={"compute_ms": 2.0, "reviewer_seconds": 3.0, "human_intervention_seconds": 1.0},
                )
            )
        return tuple(events)

    def split(self, events: Iterable[CaseEvent]) -> dict[str, tuple[CaseEvent, ...]]:
        """Return replay/adapt/retained/drift arms with no case overlap."""

        items = tuple(events)
        cases: dict[str, list[CaseEvent]] = {}
        for event in items:
            cases.setdefault(event.case_id, []).append(event)
        if len(cases) < 8:
            raise ValueError("the fixture needs at least 8 cases for all split arms")
        ordered = sorted(cases, key=lambda case: (min(e.valid_from for e in cases[case]), case))
        # Preserve the original 4/2/1/1 split at eight cases, allocating every
        # additional case chronologically. Never iterate the caller's input twice.
        replay_end = len(ordered) // 2
        adapt_end = replay_end + len(ordered) // 4
        retained_end = adapt_end + max(1, len(ordered) // 8)
        groups = {
            "replay": set(ordered[:replay_end]),
            "adapt": set(ordered[replay_end:adapt_end]),
            "retained": set(ordered[adapt_end:retained_end]),
            "drift": set(ordered[retained_end:]),
        }
        result = {name: tuple(event for event in items if event.case_id in ids) for name, ids in groups.items()}
        validate_temporal_splits(result)
        return result


def validate_temporal_splits(splits: Mapping[str, tuple[CaseEvent, ...]]) -> None:
    """Reject temporal leakage between the ordered benchmark arms."""

    order = ("replay", "adapt", "retained", "drift")
    if set(splits) != set(order):
        raise ValueError("expected exactly replay/adapt/retained/drift splits")
    seen: set[str] = set()
    for name in order:
        if not splits[name]:
            raise ValueError(f"temporal split {name!r} must be non-empty")
        cases = {event.case_id for event in splits[name]}
        if seen & cases:
            raise ValueError("case overlap between splits")
        seen.update(cases)
    for left, right in pairwise(order):
        # Both effective time and knowledge availability must precede the next
        # arm on either clock. valid_to describes expiry, not availability.
        latest_left = max(
            max(date.fromisoformat(event.valid_from), date.fromisoformat(event.observed_at)) for event in splits[left]
        )
        earliest_right = min(
            min(date.fromisoformat(event.valid_from), date.fromisoformat(event.observed_at)) for event in splits[right]
        )
        if latest_left >= earliest_right:
            raise ValueError(f"temporal leakage between {left} and {right}")


@dataclass
class AdapterState:
    memory_nodes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    harness_failures: dict[str, int] = field(default_factory=dict)
    applied_event_ids: list[str] = field(default_factory=list)


class CaseGraphAdapter:
    """Map CaseEvents to episodic memory entries and harness counters."""

    def __init__(self, artifact_version: str = "casegraph-synthetic-v1") -> None:
        self.artifact_version = artifact_version
        self.state = AdapterState()

    def apply(self, event: CaseEvent) -> None:
        if event.artifact_version != self.artifact_version:
            raise ValueError("event belongs to a different artifact version")
        event_id = event.event_id
        if event_id in self.state.applied_event_ids:
            return
        if event.event_type == "observation":
            self.state.memory_nodes.setdefault(event.case_id, []).append(
                {"event_id": event_id, "text_ref": event.text_ref, "kind": "observation"}
            )
        if event.feedback_type in {"fact_error", "retrieval_omission"}:
            self.state.memory_nodes.setdefault(event.case_id, []).append(
                {"event_id": event_id, "text_ref": event.text_ref, "kind": event.feedback_type}
            )
        if event.feedback_type in {"execution_strategy_failure", "verifier_error"}:
            self.state.harness_failures[event.feedback_type] = (
                self.state.harness_failures.get(event.feedback_type, 0) + 1
            )
        self.state.applied_event_ids.append(event_id)

    def replay(self, events: Iterable[CaseEvent]) -> dict[str, Any]:
        for event in events:
            self.apply(event)
        return {
            "artifact_version": self.artifact_version,
            "event_count": len(self.state.applied_event_ids),
            "state_digest": self.state_digest(),
        }

    def state_digest(self) -> str:
        return digest(asdict(self.state))

    def candidate(self, version: str, parent_artifact_id: str | None = None) -> VersionedArtifact:
        return VersionedArtifact(
            artifact_id=f"artifact:{version}:{self.state_digest()[:12]}",
            version=version,
            parent_artifact_id=parent_artifact_id,
            state_digest=self.state_digest(),
            event_ids=tuple(self.state.applied_event_ids),
            metrics={"synthetic": True, "memory_nodes": len(self.state.memory_nodes)},
            status="candidate",
        )


@dataclass(frozen=True)
class VersionedArtifact:
    artifact_id: str
    version: str
    parent_artifact_id: str | None
    state_digest: str
    event_ids: tuple[str, ...]
    metrics: Mapping[str, Any]
    status: str


def select_or_rollback(
    current: VersionedArtifact,
    candidate: VersionedArtifact,
    *,
    retained_score: float,
    baseline_score: float,
    allowed_regression: float = 0.0,
) -> VersionedArtifact:
    """Select a candidate only when retained quality stays within the floor."""

    if retained_score + allowed_regression < baseline_score:
        return current
    return replace(candidate, status="selected")


def fixture_digest(events: Iterable[CaseEvent]) -> str:
    return digest([event.canonical() for event in events])
