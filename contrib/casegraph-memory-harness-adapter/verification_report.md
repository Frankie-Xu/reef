# Phase 4 verification

Baseline: `ba4a85f68f95af2cd081ab7ddca728ca23cd8464`. Local and remote SHA matched
and the working tree was clean before implementation. All method changes stay
in the existing contrib directory; the implementation plan is saved under
`docs/superpowers/plans/2026-09-20-casegraph-recipe-lifecycle.md`. No core changes.

## Environment

Created ignored `.venv-phase4` with Python **3.12.13**, keeping `.venv` unchanged.
Installed editable CPU Reef/dev dependencies and the pinned reef-client submodule;
Git LFS **3.8.0** supplies the real local artifact repository. No GPU stack, paid
model API, PHI, cloud resource or production endpoint was used.

Before changing implementation, the existing **24 contribution + 44 record tests
passed on Python 3.12**. The new bridge regression first failed on the legacy
report-to-report reference, and the first lifecycle test failed before its module
existed. Development then used real scenario, trainer, SQLite, commit-log and Git
LFS implementations. Fault injection changes only a learner output, evaluator call,
commit-log append or artifact-pointer write inside the relevant test.

## Verified behavior

- Preparation leaves both scenario and artifact heads unchanged; a selected real
  artifact publishes only through Scenario.commit and survives close/reopen.
- Actual retained task judgments drive selection. The rejection case drops from
  baseline 1.0 to candidate 0.5, preserves the old policy/release, logs rejection
  and does not retrain consumed records after restart.
- Evaluation failure discards candidate bytes. Journal failure keeps unconsumed
  inputs and the old version. Both recover and complete after reopening.
- A durable commit with a failed backend-pointer update is repaired on restart.
- The entire admission batch is checked before append. Held-out, altered training
  identities and a substituted manifest are refused; the processor checks again.
- Frozen input identity is stored with the initial artifact; changing retained
  inputs before the first commit is refused on reopen.
- Actual Dispatcher ingress rejects Phase 3 native report-to-report references and
  accepts metadata.source_report_id. Storage-only testing had missed this distinction.
- The lifecycle report is rebuilt from actual tasks and persisted artifacts in two
  fresh local repositories, checked against committed JSON. Non-deterministic Git
  IDs are verified at runtime rather than included in the semantic report.

The source fixture checksum stays
`aec88a050f4812a672d40670cc5dcd2bb2e11e07784886ac90b3333c4adebd0d`.
Retained has **one case, two correlated event tasks**. Its selected-candidate and
baseline accuracies are both 1.0; this is a small-sample lifecycle diagnostic, not
proof of improved generalization or clinical/model quality. Phase 3 cost estimates
and serialization proxies remain labeled separately. Full HTTP deployment,
external inference, neural training and serving adapters are outside this example.

## Check results

- Python 3.12 focused suite: **78 passed** (44 upstream record tests + 34 contribution tests), with `GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null`.
- Command: `.venv-phase4/bin/python -m pytest -q tests/reef_service/test_records.py contrib/casegraph-memory-harness-adapter`.
- `.venv-phase4/bin/python -m mypy`: **313 source files passed**.
- Explicit mypy of the five contribution implementation modules: **5 files passed**.
- `.venv-phase4/bin/ruff check contrib/casegraph-memory-harness-adapter`: passed.
- `PATH="$PWD/.venv-phase4/bin:$PATH" .venv-phase4/bin/pre-commit run --all-files`: all applicable hooks passed, including policy, JSON, Ruff, isort and Black.
- The report reproduction test executes two independent full Git/SQLite runs and compares both to committed `lifecycle_report.json`.
- Local `.venv-phase4` logs and demo repositories remain ignored. The reef-client editable-install egg-info generated this round was removed from its submodule.

The bounded full-suite attempt on Python 3.12,
`.venv-phase4/bin/python -m pytest tests/ -q --maxfail=3`, stopped during collection:
`ray` is missing in two SGLang test modules, and `slime` is missing in a plugin
runtime test. Those unrelated inference/training stacks were not installed.
Full suite/coverage and GPU CI are not claimed.
