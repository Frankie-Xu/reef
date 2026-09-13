"""Reported feedback: assemble existing records, group samples, and reserve batches.

Ingress validates references against storage before accepting reports. The
processor consumes records in append order, so references are already present.
Deduplication, consumed-source tracking, and group slots preserve retry behavior;
retention protects every live report and the inference records it references.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from reef.core.records_types import AgentRecord, RequestType
from reef.core.reports import ReportBase, ReportValidationError, validate_report_payload
from reef.train.processors.base import DataProcessor, RetentionDecision
from reef.train.processors.common import (
    make_multi_turn_policy_sample,
    make_policy_sample,
    report_score,
    sample_assembly_config_fields,
)
from reef.train.types import PolicySample, ProcessorContext, TrainingBatch

logger = logging.getLogger(__name__)

__all__ = [
    "BatchUnit",
    "Candidate",
    "GroupDecision",
    "ReportContext",
    "ReportSample",
    "ReportedFeedbackProcessor",
    "SampleAssembly",
]


# ---------------------------------------------------------------- value types


@dataclass(frozen=True)
class ReportSample:
    """Data assembled from a valid report, with optional grouping coordinates."""

    value: Any
    group_key: Hashable | None = None
    slot: Hashable | None = None


class GroupDecision(Enum):
    """The decision for a candidate group: ready, incomplete, or invalid."""

    READY = "ready"
    INCOMPLETE = "incomplete"
    DISCARD = "discard"


@dataclass(frozen=True)
class ReportContext:
    """A valid report and its inference records, in reference order."""

    report: AgentRecord
    score: float | None
    inferences: tuple[AgentRecord, ...]
    parsed_report: ReportBase | None = None

    @property
    def references(self) -> tuple[str, ...]:
        return self.report.references

    def require_score(self) -> float:
        """Read a finite reward for a training method that needs one."""
        if self.score is None or not math.isfinite(self.score):
            raise ValueError("training data requires a finite report score")
        return self.score


@dataclass(frozen=True)
class Candidate:
    """One accepted report's contribution to a batch, cached at assembly time."""

    order: int
    value: Any
    report_agent_record_id: str
    source_agent_record_ids: frozenset[str]
    slot: Hashable


@dataclass(frozen=True)
class BatchUnit:
    """One batch unit: a singleton candidate or one ready group."""

    group_key: Hashable | None
    candidates: tuple[Candidate, ...]


# ------------------------------------------------- reported-feedback processor


class ReportedFeedbackProcessor(DataProcessor, ABC):
    """Assemble valid reports and their existing inference records into batches.

    Recipes implement ``make_sample``, ``make_batch``, and optionally
    ``decide_group``. The engine owns deduplication, group slots, reservations,
    consumption, and retention. Invalid references raise immediately; training
    data failures propagate instead of silently dropping reports.
    """

    required_request_types = frozenset({RequestType.INFERENCE, RequestType.REPORT})

    def __init__(self, context: ProcessorContext) -> None:
        super().__init__(context)
        # The sample-assembly settings belong to every reported-feedback processor's config
        # surface: validate them here so a bad deployment fails at construction
        # even for recipes that never assemble a policy sample.
        sample_assembly_config_fields(context.config)

        # --- inference store ---
        self._inferences: dict[str, AgentRecord] = {}

        # --- report lifecycle ---
        self._reports: dict[str, AgentRecord] = {}  # live: assembled or failed, not consumed
        self._seen_reports: set[str] = set()  # dedup
        self._consumed: set[str] = set()  # acknowledged batch members
        self._terminal: set[str] = set()  # terminal reports

        # --- source ownership ---
        self._trained_sources: set[str] = set()  # consumed by an acknowledged batch
        self._terminal_owned_sources: set[str] = set()  # owned by terminal reports

        # --- candidate cache ---
        self._singletons: dict[str, Candidate] = {}  # report id → singleton candidate
        self._groups: dict[Hashable, dict[Hashable, Candidate]] = {}  # group key → slot → candidate
        self._ready_groups: set[Hashable] = set()
        self._discarded_groups: set[Hashable] = set()

        # --- pending batch ---
        self._pending_units: tuple[BatchUnit, ...] | None = None
        self._next_order = 0
        self._manual_limit_warned = False

    # ------------------------------------------------------- the recipe hooks

    #: The batch type ``make_batch`` returns; the trainer validates it.
    output_schema: type[TrainingBatch] = TrainingBatch
    #: Whether a terminal report owns its referenced sources outright.
    exclusive_sources: bool = False
    #: Batch ready groups in group-key order instead of arrival order.
    ordered_groups: bool = False
    #: Units held in manual mode beyond this many batches are released, oldest first.
    manual_unit_cap_batches: int = 4

    @abstractmethod
    def make_sample(self, context: ReportContext) -> ReportSample:
        """Assemble valid feedback into data; raise on a broken training contract.

        Runs on the trainer thread without network or model calls. Every
        valid report produces a sample; this hook does not accept/reject reports.
        """

    def decide_group(self, key: Hashable, candidates: tuple[Candidate, ...]) -> GroupDecision:
        """Decide whether an accepted candidate group is ready, incomplete, or invalid."""
        raise NotImplementedError(
            f"{type(self).__name__} produced a grouped candidate without a decide_group override"
        )

    @abstractmethod
    def make_batch(self, units: tuple[BatchUnit, ...], batch_number: int) -> TrainingBatch:
        """Shape the selected units into the recipe's typed batch."""

    def ingest(self, item: AgentRecord) -> None:
        if item.scenario != self.scenario:
            raise ReportValidationError("records must belong to the processor's scenario")
        if item.request_type is RequestType.TRAIN:
            super().ingest(item)
            return
        if item.request_type is RequestType.INFERENCE:
            self._inferences[item.agent_record_id] = item
            return
        if item.request_type is not RequestType.REPORT or item.agent_record_id in self._seen_reports:
            return
        validate_report_payload(item.payload)
        if not item.references or len(set(item.references)) != len(item.references):
            raise ReportValidationError("report references must be non-empty and unique")
        if any(ref in self._trained_sources for ref in item.references):
            self._seen_reports.add(item.agent_record_id)
            self._terminate(item)
            return
        context = self._report_context(item)
        # Retain the report before assembly: a contract failure must not let
        # compaction delete its inputs or turn a retry into a successful no-op.
        self._reports[item.agent_record_id] = item
        sample = self.make_sample(context)
        if not isinstance(sample, ReportSample):
            raise TypeError(f"{type(self).__name__}.make_sample must return ReportSample")
        self._seen_reports.add(item.agent_record_id)
        key = sample.group_key
        slot = item.agent_record_id if sample.slot is None else sample.slot
        if key is not None and (key in self._discarded_groups or slot in self._groups.get(key, {})):
            self._terminate(item)
            return
        self._next_order += 1
        candidate = Candidate(
            order=self._next_order,
            value=sample.value,
            report_agent_record_id=item.agent_record_id,
            source_agent_record_ids=frozenset(item.references),
            slot=slot,
        )
        if key is None:
            self._singletons[item.agent_record_id] = candidate
        else:
            self._groups.setdefault(key, {})[slot] = candidate
            self._refresh_group(key)
        self._cap_manual_units()

    def set_training_mode(self, training_mode: str) -> None:
        super().set_training_mode(training_mode)
        # The base selects the initial mode inside __init__, before any unit store exists; a switch trims at once.
        if hasattr(self, "_singletons"):
            self._cap_manual_units()

    def _cap_manual_units(self) -> None:
        """Manual mode batches on instructions, not units, so the pile is bounded here instead."""
        limit = self._batch_size * self.manual_unit_cap_batches
        if self.training_mode != "manual" or self._ready_count() <= limit:
            return
        # The reserved batch is handed out until acknowledged, so its units stay put.
        reserved = {c.report_agent_record_id for unit in self._pending_units or () for c in unit.candidates}
        for unit in self._ordered_units():
            if self._ready_count() <= limit:
                return
            if unit.candidates[0].report_agent_record_id in reserved:
                continue
            self._release_unit(unit)
            if not self._manual_limit_warned:
                logger.warning(
                    "%s scenario %r released report %s beyond the manual limit %d (further releases are not logged)",
                    type(self).__name__,
                    self.scenario,
                    unit.candidates[0].report_agent_record_id,
                    limit,
                )
                self._manual_limit_warned = True

    def _release_unit(self, unit: BatchUnit) -> None:
        """Drop one held unit: its reports turn terminal and own their sources, so retention frees both."""
        if unit.group_key is not None:
            self._discard_group(unit.group_key)
        for candidate in unit.candidates:
            report = self._reports.pop(candidate.report_agent_record_id, None)
            self._singletons.pop(candidate.report_agent_record_id, None)
            if report is not None:
                self._terminate(report)
            self._terminal_owned_sources.update(candidate.source_agent_record_ids)

    def _terminate(self, report: AgentRecord) -> None:
        """Drop a report from the live set and mark it terminal.

        Terminal reports release their own record, and the sources they own
        outright: a report that claims more than one inference,
        or any report when the recipe declares
        ``exclusive_sources``.
        """
        report_id = report.agent_record_id
        self._reports.pop(report_id, None)
        self._terminal.add(report_id)
        if not report.references:
            return
        if self.exclusive_sources or len(report.references) > 1:
            self._terminal_owned_sources.update(report.references)

    def _report_context(self, report: AgentRecord) -> ReportContext:
        missing = [ref for ref in report.references if ref not in self._inferences]
        if missing:
            raise ReportValidationError(f"report references unavailable inference records: {missing!r}")
        inferences = tuple(self._inferences[ref] for ref in report.references)
        report_type = self.context.report_type
        parsed_report = None if report_type is None else report_type.from_dict(report.payload)
        return ReportContext(report, report_score(report), inferences, parsed_report)

    # ---------------------------------------------------------------- groups

    def _group_candidates(self, key: Hashable) -> tuple[Candidate, ...]:
        return tuple(sorted(self._groups[key].values(), key=lambda c: c.order))

    def _refresh_group(self, key: Hashable) -> None:
        decision = self.decide_group(key, self._group_candidates(key))
        if decision is GroupDecision.READY:
            self._ready_groups.add(key)
        elif decision is GroupDecision.INCOMPLETE:
            self._ready_groups.discard(key)
        else:
            self._discard_group(key)

    def _discard_group(self, key: Hashable) -> None:
        self._ready_groups.discard(key)
        self._discarded_groups.add(key)
        group = self._groups.pop(key)
        # Remove every member from the live set first, so the wholesale
        # release is not blocked by siblings of the same discarded group.
        members = [
            report
            for candidate in group.values()
            if (report := self._reports.pop(candidate.report_agent_record_id, None)) is not None
        ]
        for report in members:
            self._terminate(report)

    # ----------------------------------------------------------- batch cycle
    #
    # A unit is one accepted singleton candidate or one ready group; the
    # engine's half of the shared cycle in base.py is the three methods below.

    def _ready_count(self) -> int:
        return len(self._singletons) + len(self._ready_groups)

    def _make_pending(self, batch_number: int) -> TrainingBatch:
        units = self._select_units()
        self._pending_units = units
        return self.make_batch(units, batch_number)

    def _select_units(self) -> tuple[BatchUnit, ...]:
        """Select up to ``batch_size`` batch units in priority order."""
        return tuple(self._ordered_units()[: self._batch_size])

    def _ordered_units(self) -> list[BatchUnit]:
        """Every held unit in priority order.

        By default everything shares arrival order. With ``ordered_groups``,
        singletons batch first (arrival order), then groups in key order.
        """
        singletons = [BatchUnit(None, (candidate,)) for candidate in self._singletons.values()]
        if self.ordered_groups:
            # ordered_groups requires sortable group keys (step indices, say).
            ordered = sorted(cast("set[Any]", self._ready_groups))
            groups = [BatchUnit(key, self._group_candidates(key)) for key in ordered]
            units = singletons + groups
        else:
            units = singletons + [BatchUnit(key, self._group_candidates(key)) for key in self._ready_groups]
            units.sort(key=lambda unit: unit.candidates[0].order)
        return units

    def _consume_pending(self) -> frozenset[str]:
        if self._pending_units is None:
            raise RuntimeError("cannot consume a batch before units are pending")
        consumed_reports: set[str] = set()
        trained_sources: set[str] = set()
        for unit in self._pending_units:
            group = self._groups.get(unit.group_key) if unit.group_key is not None else None
            for candidate in unit.candidates:
                report_id = candidate.report_agent_record_id
                self._consumed.add(report_id)
                consumed_reports.add(report_id)
                self._reports.pop(report_id, None)
                self._singletons.pop(report_id, None)
                trained_sources |= candidate.source_agent_record_ids
                if group is not None:
                    group.pop(candidate.slot, None)
            if unit.group_key is not None and group is not None:
                if group:
                    self._refresh_group(unit.group_key)
                else:
                    self._groups.pop(unit.group_key)
                    self._ready_groups.discard(unit.group_key)
        self._trained_sources.update(trained_sources)
        self._pending_units = None
        return frozenset(consumed_reports | trained_sources)

    # -------------------------------------------------------------- retention

    def _live_references(self) -> set[str]:
        return {ref for report in self._reports.values() for ref in report.references}

    def retention_decision(self) -> RetentionDecision:
        """Derive retention from live state — a pure read, nothing mutates.

        The releasable-source set is recomputed here every time: a source is
        releasable while a terminal report owns it (or a batch consumed it)
        and no live report references it.  Live claims and terminal ownership
        both move between reads, so the answer is never latched at event time.
        """
        live_references = self._live_references()
        releasable_sources = (self._terminal_owned_sources | self._trained_sources) - live_references
        releasable = self._consumed | self._terminal | releasable_sources
        protected = set(self._reports) | live_references
        protected.update(inference_id for inference_id in self._inferences if inference_id not in releasable_sources)
        return RetentionDecision(
            protected_agent_record_ids=frozenset(protected | self._training_requests.keys()),
            releasable_agent_record_ids=frozenset(releasable | self._consumed_requests),
        )

    def compaction_applied(self, agent_record_ids: frozenset[str]) -> None:
        super().compaction_applied(agent_record_ids)
        # --- scalar id sets ---
        self._consumed -= agent_record_ids
        self._terminal -= agent_record_ids
        self._trained_sources -= agent_record_ids
        self._terminal_owned_sources -= agent_record_ids
        self._seen_reports -= agent_record_ids

        # Stored records: only inferences can be here. The trainer compacts
        # ``releasable - protected``, and every live report, every reference
        # a live report holds, and every candidate's report are protected —
        # so a compacted id is never in _reports, _singletons, or a
        # group.
        for agent_record_id in agent_record_ids:
            self._inferences.pop(agent_record_id, None)


# --------------------------------------------------------- shared helpers


@dataclass(frozen=True)
class SampleAssembly:
    """Shape a resolved report into a policy sample, without recipe policy.

    One model call becomes one sample via ``make_sample`` (default: the shared
    tensor reader); an ordered multi-reference report becomes one assembled
    multi-turn sample. ``accept_multi_turn`` gates consumption only —
    assembly always runs first. Unsupported or unassemblable trajectories
    raise a training data error and retain their source records.
    """

    accept_multi_turn: bool = False
    realign_threshold: int = 1024
    scaffold_tolerance: int = 0
    make_sample: Callable[[AgentRecord, float], PolicySample] | None = None

    @classmethod
    def from_config(
        cls,
        context: ProcessorContext,
        make_sample: Callable[[AgentRecord, float], PolicySample] | None = None,
    ) -> SampleAssembly:
        accept_multi_turn, realign_threshold, scaffold_tolerance = sample_assembly_config_fields(context.config)
        return cls(accept_multi_turn, realign_threshold, scaffold_tolerance, make_sample)

    def build(self, context: ReportContext, score: float) -> PolicySample:
        """Build training data, raising if the recorded trajectory cannot be used."""
        inferences = context.inferences
        if inferences is None:
            raise RuntimeError("sample assembly requires resolved inferences")
        if len(inferences) == 1:
            sample: PolicySample | None = (
                make_policy_sample(inferences[0], score)
                if self.make_sample is None
                else self.make_sample(inferences[0], score)
            )
        else:
            sample = make_multi_turn_policy_sample(
                inferences,
                score,
                source_agent_record_id=context.report.agent_record_id,
                realign_threshold=self.realign_threshold,
                scaffold_tolerance=self.scaffold_tolerance,
            )
        if sample is None:
            raise ValueError("training data cannot assemble the recorded multi-turn trajectory")
        if sample.is_multi_turn and not self.accept_multi_turn:
            raise ValueError("training data requires accept_multi_turn_policy_samples for this trajectory")
        return sample
