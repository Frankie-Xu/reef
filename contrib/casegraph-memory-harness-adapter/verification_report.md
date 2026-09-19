# Reef repository and contribution verification

- Public repository: https://github.com/Human-Agent-Society/reef
- License: Apache-2.0.
- Visible contribution surfaces: `CONTRIBUTING.md`, `docs/`, `recipes/`, `reef/`, `tests/`, `train/evaluation/`, `artifact/`, and `surface/`.
- This projectless draft does not modify or push to the upstream repository.
- Data policy: synthetic identifiers, synthetic text references, and synthetic provenance only; no PHI.
- Current main snapshot was read through the GitHub UI on 2026-09-19. The
  visible RFC alignment is #522 (memory node), #514 (retained harness
  evaluation, draft #515 pending acceptance), and #532 (TrainingTrigger,
  `status: needs-triage`).

## Validation

`uv run --with pytest --python python3 -m pytest -q` → 11 passed.

The failure regressions cover invalid provenance, tampered event IDs, temporal
leakage, artifact-version mismatch, candidate rollback, and duplicate replay.
`reef-caseevent-fixture.json` has a stable digest
`aec88a050f4812a672d40670cc5dcd2bb2e11e07784886ac90b3333c4adebd0d`; the
retained report is `retained_evaluation_report.json`.

The upstream mapping is intentionally descriptive:

| Draft | Upstream discussion point |
| --- | --- |
| Event/replay and report references | `reef/storage/records.py` and report references |
| Memory/harness update surfaces | #522 and recipe-owned reader/writer |
| Retained quality gate | `reef/train/evaluation`, #514/#515 |
| Versioned candidate and rollback | `reef/artifact`, `reef/surface` |
| Training timing | #532 `TrainingTrigger` |

## Remaining boundary

The adapter is intentionally independent of Reef APIs. A maintainer-approved checkout and placement decision are required before mapping it into Reef recipes, records, evaluation, artifact, or surface modules. No upstream checkout was found in the projectless workspace, so no source files were changed and no external message was sent.
