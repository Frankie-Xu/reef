# Reef CaseGraph adapter draft

This is an independent, projectless contribution draft for Reef. It uses only fabricated `syn-case-*` records and synthetic provenance. It does not copy Reef source code, call a production service, or modify the upstream checkout.

The adapter demonstrates:

- deterministic `CaseEvent` generation with a fixed seed;
- temporal and case-disjoint replay/adapt/retained/drift splits;
- feedback credit assignment across fact, retrieval, execution, and verifier failures;
- episodic-memory versus procedural-harness update surfaces;
- versioned candidate artifacts with retained-score rollback;
- idempotent replay and stable state digests.

Each event has a stable `event_id = sha256(canonical_event_json)`. The JSON
fixture includes that ID, and `CaseEvent.from_dict()` / `deserialize()` reject a
payload whose ID no longer matches. The canonical payload carries
`provenance`, `valid_from`/`valid_to`, `feedback_type`, `update_surface`,
`artifact_version`, and the compute/reviewer/human cost fields.

`retained_evaluation_report.json` is a synthetic report for the four arms. It
records quality, traceability, review time, compute, human cost, and negative
transfer. Its retained-floor decision selects a candidate at equal retained
quality while recording drift degradation as a follow-up signal; a future
upstream gate should rollback when retained quality falls below the allowed
floor.

## Mapping to Reef surfaces

| Draft contract | Reef surface to discuss with maintainers | Deliberate boundary |
| --- | --- | --- |
| `CaseEvent` append/replay and provenance | `reef/storage/records.py` and report references | Do not change `RecordStore` persistence or make retained data training-visible |
| memory entries vs harness failure counters | #522 memory node / recipe surface | Keep schema and reader recipe-owned until the RFC is accepted |
| retained-floor evaluation | `reef/train/evaluation` and #514 | Use candidate `evaluate/decide`; keep retained cases independent and read-only |
| parented candidate, select/rollback | `reef/artifact` and `reef/surface` | Stage candidates, publish only after the gate, discard rejected artifacts |
| training schedule | #532 `TrainingTrigger` | Do not couple feedback attribution to when training starts |

The table is an integration plan, not copied Reef implementation. The draft
does not call Reef, start a service, or alter a production model endpoint.

Run from this directory:

```bash
uv run --with pytest --python python3 -m pytest -q
```

The current draft passes eleven tests. Upstream integration remains blocked until a maintainer-approved checkout and contribution boundary are available.
