"""Reef policy admission against serving versions and scenario history."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from reef.core.artifact_ref import parse_runtime_load_spans
from reef.runtime.training_job.execution import uses_staleness_admission as _uses_staleness_admission
from reef.runtime.training_job.scenarios import ScenarioHistory


@dataclass(frozen=True, slots=True)
class _StalenessDecision:
    action: Literal["admit", "drop"]
    metrics: Mapping[str, Any]


def _source_agent_record_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    samples = payload.get("samples")
    if not isinstance(samples, Sequence) or isinstance(samples, str | bytes):
        return ()
    return tuple(
        str(row[0]) for row in samples if isinstance(row, Sequence) and not isinstance(row, str | bytes) and row
    )


def _producing_runtime_load_ids(payload: Mapping[str, Any]) -> Sequence[Any]:
    versions = payload.get("producing_runtime_load_ids")
    if not isinstance(versions, Sequence) or isinstance(versions, str | bytes) or not versions:
        raise ValueError("bounded staleness admission requires producing_runtime_load_ids")
    source_ids = _source_agent_record_ids(payload)
    if len(versions) != len(source_ids):
        raise ValueError(
            "bounded staleness admission requires one producing runtime load ID "
            f"per sample: {len(versions)} versions for {len(source_ids)} samples"
        )
    return versions


def _admission_runtime_load_id_groups(payload: Mapping[str, Any]) -> list[list[Any]]:
    """Return each sample's exact span versions for bounded admission."""
    versions = _producing_runtime_load_ids(payload)
    raw_groups = payload.get("producing_runtime_load_spans")
    if raw_groups is None:
        return [[version] for version in versions]
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, str | bytes) or len(raw_groups) != len(versions):
        raise ValueError("producing_runtime_load_spans must contain one span list per sample")
    samples = payload["samples"]
    groups: list[list[Any]] = []
    for sample_index, (raw_spans, scalar) in enumerate(zip(raw_groups, versions, strict=True)):
        if not raw_spans:
            groups.append([scalar])
            continue
        row = samples[sample_index]
        response_length = len(row[2]) if isinstance(row, Sequence) and len(row) > 2 else None
        spans = parse_runtime_load_spans(
            raw_spans,
            field_name=f"producing_runtime_load_spans[{sample_index}]",
            response_length=response_length,
        )
        group = [span.runtime_load_id for span in spans]
        span_versions = set(group)
        if scalar is not None and span_versions != {scalar}:
            raise ValueError(f"producing runtime load ID for sample {sample_index} disagrees with its token spans")
        groups.append(group)
    return groups


def _stale_drop_decision(
    payload: Mapping[str, Any],
    *,
    serving_runtime_load_id: str,
    producing_runtime_load_ids: Sequence[Any],
    reason: str,
    policy_lags: Sequence[int] = (),
) -> _StalenessDecision:
    source_ids = _source_agent_record_ids(payload)
    metrics: dict[str, Any] = {
        "staleness/samples_dropped": len(source_ids) or len(producing_runtime_load_ids),
        "staleness/drop_reason": reason,
        "staleness/source_agent_record_ids": list(source_ids),
        "staleness/producing_runtime_load_ids": [
            None if version is None else str(version) for version in producing_runtime_load_ids
        ],
        "staleness/serving_runtime_load_id": serving_runtime_load_id,
    }
    if policy_lags:
        metrics["staleness/drop_policy_lags"] = list(policy_lags)
    return _StalenessDecision(action="drop", metrics=metrics)


def _staleness_admission(
    payload: Mapping[str, Any],
    *,
    serving_runtime_load_id: str,
    max_staleness: int,
) -> _StalenessDecision:
    from reef.runtime.weights.version import RuntimeLoadId

    try:
        serving = RuntimeLoadId.parse(serving_runtime_load_id)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot classify staleness from serving runtime load ID {serving_runtime_load_id!r}"
        ) from exc
    if str(serving) != serving_runtime_load_id:
        raise RuntimeError(f"cannot classify staleness from non-canonical serving version {serving_runtime_load_id!r}")
    producing_groups = _admission_runtime_load_id_groups(payload)
    producing_versions = [version for group in producing_groups for version in group]

    lags: list[int] = []
    sample_lags: list[int] = []

    def drop(reason: str) -> _StalenessDecision:
        return _stale_drop_decision(
            payload,
            serving_runtime_load_id=serving_runtime_load_id,
            producing_runtime_load_ids=producing_versions,
            reason=reason,
            policy_lags=lags,
        )

    for group in producing_groups:
        group_lags: list[int] = []
        previous_sequence: int | None = None
        for value in group:
            if not isinstance(value, str) or not value:
                return drop("missing_producing_runtime_load_id")
            try:
                producing = RuntimeLoadId.parse(value)
            except (TypeError, ValueError):
                return drop("malformed_producing_runtime_load_id")
            if str(producing) != value:
                return drop("malformed_producing_runtime_load_id")
            if producing.incarnation != serving.incarnation:
                return drop("cross_incarnation")
            if previous_sequence is not None and producing.sequence <= previous_sequence:
                return drop("non_monotonic_producing_runtime_load_ids")
            previous_sequence = producing.sequence
            lag = serving.sequence - producing.sequence
            lags.append(lag)
            group_lags.append(lag)
            if lag < 0:
                return drop("future_producing_runtime_load_id")
            if lag > max_staleness:
                return drop("policy_lag_exceeded")
        sample_lags.append(max(group_lags))
    return _StalenessDecision(
        action="admit",
        metrics={
            "staleness/samples_fresh": sum(lag == 0 for lag in sample_lags),
            "staleness/samples_admitted_stale": sum(lag > 0 for lag in sample_lags),
        },
    )


def _scenario_staleness_admission(
    payload: Mapping[str, Any],
    *,
    scenario: str,
    history: ScenarioHistory,
    serving_runtime_load_id: str,
    max_staleness: int,
) -> _StalenessDecision:
    """Bounded admission against one scenario's own publication history.

    The engine's runtime load ID advances on every scenario's publication, so
    the global sequence gap overstates this scenario's staleness. A sample's
    lag is the number of *this* scenario's publications that postdate the
    version its tokens were produced under.
    """
    from reef.runtime.weights.version import RuntimeLoadId

    if _uses_staleness_admission(payload):
        producing_groups = _admission_runtime_load_id_groups(payload)
    else:
        expected = payload.get("expected_runtime_load_id")
        producing_groups = [[expected]]
    producing_versions = [version for group in producing_groups for version in group]
    lags: list[int] = []
    sample_lags: list[int] = []

    def drop(reason: str) -> _StalenessDecision:
        return _stale_drop_decision(
            payload,
            serving_runtime_load_id=serving_runtime_load_id,
            producing_runtime_load_ids=producing_versions,
            reason=reason,
            policy_lags=lags,
        )

    for group in producing_groups:
        group_lags: list[int] = []
        for value in group:
            if not isinstance(value, str) or not value:
                return drop("missing_producing_runtime_load_id")
            try:
                producing = RuntimeLoadId.parse(value)
            except (TypeError, ValueError):
                return drop("malformed_producing_runtime_load_id")
            lag = history.lag(scenario, producing)
            if lag is None:
                return drop("cross_incarnation")
            lags.append(lag)
            group_lags.append(lag)
            if lag > max_staleness:
                return drop("policy_lag_exceeded")
        sample_lags.append(max(group_lags))
    return _StalenessDecision(
        action="admit",
        metrics={
            "staleness/samples_fresh": sum(lag == 0 for lag in sample_lags),
            "staleness/samples_admitted_stale": sum(lag > 0 for lag in sample_lags),
            "staleness/scenario": scenario,
        },
    )
