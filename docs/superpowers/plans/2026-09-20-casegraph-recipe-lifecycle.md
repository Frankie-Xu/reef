# CaseGraph Recipe Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.
> Execution override: the user requires this existing task and branch. Execute inline;
> the named execution subskills are not installed. No delegation or new task is needed.

**Goal:** Run a CPU-only synthetic feedback-routing recipe through Reef's actual candidate evaluation, durable scenario commit and artifact publication/recovery.

**Architecture:** Keep method policy in the existing contrib directory. A Recipe builds a DataProcessor, CandidateBackend and CandidateEvaluationPlugin; ScenarioFactory/Scenario own persistence, training-step preparation, commit and recovery. Frozen retained cases score candidate and incumbent routing policies by exact expected action; the backend learns only from replay/adapt records.

**Tech Stack:** Python 3.12, Reef checked out at ba4a85f6, pytest, SQLiteScenarioStorage, GitLFSRepositoryBackend, local JSON artifacts. No core changes.

## Global Constraints

- GPT-6 Astra / medium only; no other model or downgrade.
- Existing work/fork and codex/casegraph-memory-harness-adapter branch only.
- No force push, PR, issue/comment, external messages, PHI, paid model calls, GPU stack or cloud resources.
- Preserve .venv; create ignored .venv-phase4 with Python 3.12 and CPU dependencies.
- Report deterministic task accuracy as synthetic only; serialization rates are not quality.
- Frozen retained/drift input cannot be appended by the recipe's admission path or processor.
- Public references in Reef reports identify inference receipts only; report-to-report lineage must use recipe metadata, never pretend an observation is inference.

---

### Task 1: Verify environment and repair report admission semantics

**Files:** modify contrib/casegraph-memory-harness-adapter/reef_record_bridge.py and test_reef_record_bridge.py; create ignored .venv-phase4.
**Interfaces:** event_to_record(event, *, scenario, observation=None) -> AgentRecord continues to preserve CaseEvent; feedback observation ID moves to metadata.source_report_id. Native references remain empty because no model inference occurred.

- [x] Create Python 3.12 environment with `uv venv --python 3.12 .venv-phase4`; install editable CPU Reef and dev dependencies plus local reef-client.
- [x] Run the existing 24 contribution + 44 storage tests before changes; expect 68 passed.
- [x] Test native references are empty and metadata names the observation receipt; retain conflict, round-trip and scenario rejection tests.
- [x] Update the bridge metadata and run the 68 tests again.

### Task 2: Real recipe and durable lifecycle

**Files:** create contrib/casegraph-memory-harness-adapter/casegraph_recipe.py and test_casegraph_recipe.py.
**Interfaces:** CaseGraphRecipe(Recipe) builds a Trainer; CaseGraphProcessor(DataProcessor) emits TrainingBatch; CaseGraphBackend(CandidateBackend) prepares a local Artifact and settles a TrainStepResult; RetainedRoutingEvaluator(CandidateEvaluationPlugin) measures and selects; open_scenario(work_dir: Path, recipe: CaseGraphRecipe) -> Scenario uses real ScenarioFactory with Git LFS and SQLite.

- [x] Write the acceptance test around real APIs:
  ```python
  scenario = open_scenario(tmp_path, recipe)
  old = scenario.current_artifact_ref()
  ingest_training(scenario, recipe.fixture)
  result = scenario.prepare_training_step()
  assert result is not None and result.artifact is not None
  assert scenario.current_artifact_ref() == old
  scenario.commit(result)
  assert scenario.current_artifact_ref() != old
  scenario.close()
  ```
- [x] Implement the processor with public ingest/ready/build_batch/acknowledge/release_batch/retention_decision methods. Allow only exact frozen training event identities; never accept retained or drift.
- [x] Learn a feedback-category -> corrective surface lookup from training examples. Baseline routes feedback to procedural_harness and observations to episodic_memory; candidate uses training labels. Materialize policy.json in a local Artifact; do not move repository refs in backend preparation.
- [x] Freeze the entire fixture as serialized immutable strings, derive distinct train and retained values, and compare exact predicted update_surface against fixture labels. Select only if candidate retained accuracy is at least incumbent accuracy. Return actual per-task predictions, labels and scores.
- [x] Reject with unchanged algorithm state and no Artifact. Let Trainer call abort_step on evaluator failures; discard temporary candidate bytes.
- [x] Test successful publication/restart, rejected candidate preserving current release, evaluation failure with restart/retry, journal failure before commit, pointer synchronization recovery, and held-out rejection before storage and in processor.
- [x] Use narrowly scoped fault injection only in tests; all storage, repositories, trainer and scenario objects remain real implementations.

### Task 3: Reproducible lifecycle report, checks, delivery

**Files:** create contrib/casegraph-memory-harness-adapter/run_lifecycle.py, lifecycle_report.json; modify README.md, verification_report.md and report tests. Preserve build_report.py's explicitly labeled Phase 3 protocol report.
**Interfaces:** run_lifecycle(work_dir: Path) -> dict[str, object] executes one real run and returns deterministic semantic results; CLI writes report to --output with --work-dir required.

- [x] Report measured routing accuracy, per-case outcomes, baseline/candidate comparison, frozen input hashes, training record IDs, selection and restart checks. Do not put nondeterministic Git release IDs/timestamps in the reproducible semantic report; verify them in runtime assertions/tests.
- [x] Rebuild in two fresh work directories and assert reports equal; verify committed JSON matches.
- [x] Document observe -> record admission, grow -> prepare_training_step (Trainer prepare/evaluate/settle), commit -> Scenario.commit; explain that no external inference or neural training occurs.
- [x] Run relevant tests on Python 3.12, explicit contrib mypy, root mypy, Ruff, pre-commit --all-files and a bounded full pytest collection attempt. Do not install missing GPU stacks for unrelated tests.
- [x] Review scoped diff; commit and push existing branch; compare local HEAD with git ls-remote and require clean status.

## Acceptance

Working real recipe lifecycle with selected/rejected/recovered outcomes; task-derived synthetic scores; immutable held-out input excluded from training storage; reproducible report; documented actual checks and limitations; remote SHA verified. No generic core changes or deployment.

## Execution notes

Implemented inline without subagents. Python 3.12 baseline: 68 passed. Final scoped suite: 78 passed. Root mypy: 313 files; contribution mypy: 5 files. Retained is one case/two correlated tasks and the report explicitly limits interpretation. A persisted fixture identity additionally prevents changing held-out data across a pre-commit restart. Native report lineage moved to metadata, with a real Dispatcher ingress regression. Full suite collection is blocked by absent ray/slime; no GPU dependencies were installed. Final commit/push and remote SHA are reported in the task result.
