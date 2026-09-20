# Synthetic CaseGraph report/record adapter

This contribution contains deterministic synthetic CaseEvents, a local memory/harness
state prototype, and a thin bridge to the checked-out Reef report and record APIs.
All data is fabricated. No model calls, PHI, GPU work or production writes are involved.

## Implemented boundary

`reef_record_bridge.py` defines a recipe-local `CaseEventReport(ReportBase)`.
Its metadata contains the canonical serialized event, preserving provenance,
validity dates, observation time, feedback, artifact version and cost estimates.
`event_to_record()` returns a real `AgentRecord` of type `REPORT`. Feedback
references an observation receipt from the same case, scenario and artifact version.
Record IDs include scenario and event identity. This is report-to-report linkage;
there is no fabricated inference receipt or rollout score.

The bridge does not append records. The integration test appends only replay/adapt
to a disposable real `SQLiteRecordStore`, closes/reopens it, and checks receipt
linkage, idempotency, conflict rejection, scenario isolation and adapter replay.
Retained/drift events must stay outside training-visible storage; callers own this
boundary. The bridge alone does not enforce split admission.

No Reef recipe registration, HTTP ingress, processor, memory node, reader surface,
training, candidate evaluation plugin, artifact publication or observe/grow/commit
lifecycle is implemented. `CaseGraphAdapter` memory entries and harness counters are
local Python state. `VersionedArtifact` and numeric `select_or_rollback()` are local
prototypes, not Reef artifacts or measured selection outcomes.

## Split and report protocol

The splitter materializes an iterable once, groups complete cases chronologically,
and allocates half to replay, a quarter to adapt, an eighth (at least one) to
retained, and the remainder to drift. Fractions round down. At least eight cases
are required; the original 4/2/1/1 split is preserved. All input events are assigned
exactly once. Both valid_from and observed_at in an earlier arm must precede both
clocks in the next arm. `valid_to` describes expiry and may extend across arms;
this is a case-disjoint protocol, not an as-of clinical validity evaluator.

`build_report.py` verifies event hashes, fixture digest and split manifest, then
rebuilds `retained_evaluation_report.json`. Only replay/adapt update candidate
state. The report measures serialization round trips and synthetic-reference
coverage. Costs are sums of fixture estimates, not measured runtime or review time.
Quality and negative transfer remain null; selection is `not_evaluated`.
These protocol proxies are not clinical or model benchmark results.

Earlier hand-filled report scores and duplicated JSON under `docs/` were removed.
The root fixture and generated report in this directory are the only current copies.
The previous brief was replaced with an implementation boundary note.

## Reproduce from the repository root

Use the repository Python 3.12 environment with Reef dependencies and dev tools
installed, as described in the root contribution guide:

```bash
.venv/bin/python contrib/casegraph-memory-harness-adapter/build_report.py
.venv/bin/python -m pytest -q contrib/casegraph-memory-harness-adapter
.venv/bin/python -m mypy contrib/casegraph-memory-harness-adapter/reef_casegraph_adapter.py contrib/casegraph-memory-harness-adapter/reef_record_bridge.py contrib/casegraph-memory-harness-adapter/build_report.py
```

See `verification_report.md` for the checks actually run and their limitations.
