# Synthetic CaseGraph recipe

This contribution runs a CPU-only synthetic corrective-action task through the
checked-out Reef recipe, Trainer, candidate evaluation, durable scenario commit
and artifact publication/recovery APIs. It uses fabricated CaseEvents, no external
model, no neural training, no PHI and no cloud resources.

## What the task measures

Given an observation or a feedback category, a policy chooses `episodic_memory`
or `procedural_harness`. The initial policy sends observations to memory and all
feedback to the procedural harness. The learner updates that finite routing table
from replay/adapt examples. The verifier executes the table on frozen retained
inputs and compares its chosen action to the fixture's expected `update_surface`.
This is actual synthetic task accuracy, not serialization success.

The default retained arm is **one case with two correlated event tasks**. Its
baseline and selected candidate both score 1.0. This tiny diagnostic establishes
lifecycle behavior, not a generalization, model-quality or clinical benefit.
The rejection test supplies a regressing candidate at the learner boundary:
retained accuracy is 0.5 and the existing release remains selected. Evaluation
failures produce no selection. No evaluator result is hard-coded by the recipe.

## Actual interfaces and responsibilities

| Concept | Current Reef API | Contribution policy |
| --- | --- | --- |
| Recipe | `Recipe.build()` → real `Trainer` | `CaseGraphRecipe`, programmatic configuration |
| Observe | `RecordStore.append()` after recipe admission | Exact frozen training events only |
| Grow | `Scenario.prepare_training_step()` | Trainer calls `CandidateBackend.prepare_step`, `CandidateEvaluationPlugin.evaluate/decide`, `CandidateBackend.settle_step` |
| Candidate | `Artifact.local()` → `TrainStepResult` | Real `policy.json` bytes, not a local fake release |
| Commit | `Scenario.commit()` | Reef stages/publishes bytes, durably logs the decision, compacts consumed records and advances the release |
| Restore | `ScenarioFactory.load_or_create()` | Reef reconciles the Git artifact head and scenario commit log |

There are no `observe()` or `grow()` methods to emulate in this checkout.
`open_scenario()` composes the real `ScenarioFactory`, `SQLiteScenarioStorage`
and `GitLFSRepositoryBackend`. It starts no HTTP server or background dispatcher
worker. The selected release can be read with `Scenario.artifact_for_version()`.
No inference engine, model endpoint, Harbor rollout, weight optimizer or serving
adapter is implemented. This is a programmatic contrib recipe, not a YAML cookbook
deployment or clinical CaseGraph system. No generic core API was changed.

`CaseGraphProcessor` declares a typed `RoutingBatch(TrainingBatch)` carrying the
synthetic events; it does not invent token trajectories or Harbor tasks. The
backend receives training examples and fixture hashes only. A separate retained
evaluator owns the held-out inputs. The evaluation plugin is mandatory; the
backend's default evaluation entry point refuses to bypass it.

## Frozen input and admission

`FrozenFixture.load()` checks event identities, the fixture checksum and split
manifest, then stores nested event data as immutable JSON strings. The original
splitter allocates every case once: half replay, a quarter adapt, an eighth
(at least one) retained and the remainder drift, rounding down. It accepts one-shot
iterators and at least eight cases. Both effective and observed times must precede
both clocks in the next arm. `valid_to` is expiry, not knowledge availability.

`ingest_training()` validates the whole submitted batch against the scenario's
exact frozen training manifest before appending. Split labels alone are not trusted.
The processor independently rejects held-out or modified content. Both retained
and drift remain outside training storage and consumed record IDs. An identity
covering all frozen arms is persisted in the base and candidate artifacts;
`open_scenario()` rejects changed fixtures even before the first training commit.
Use this admission path: direct writes to a raw RecordStore bypass its pre-storage
check, and a generic HTTP endpoint is not configured as a secure deployment here.

## Report linkage correction

`CaseEventReport(ReportBase)` stores the canonical event under metadata.
`AgentRecord` has type `REPORT`; feedback links its observation report through
`metadata.source_report_id`, with native `references` empty. Reef reserves native
references for actual inference receipts; no inference occurred in this example.

Phase 3 used report-to-report native references. SQLite accepted that shape, so
its storage-only round-trip test did not reveal that actual Dispatcher ingress
rejects it. A new real `Dispatcher.accept_record()` test verifies that the old
shape is rejected and the corrected shape accepted. Existing Phase 3 databases
are not migrated; reproduce with fresh work directories to avoid conflicting
record IDs whose corrected payload has changed.

## Reproduce from the repository root

Python 3.12 and Git LFS are required for these lifecycle checks. The isolated
`.venv-phase4` leaves the existing `.venv` unchanged. Use CPU dependencies only:

```bash
uv venv --python 3.12 .venv-phase4
git submodule update --init third_party/reef-client
uv pip install --python .venv-phase4/bin/python -e '.[dev]' -e ./third_party/reef-client
GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null .venv-phase4/bin/python -m pytest -q tests/reef_service/test_records.py contrib/casegraph-memory-harness-adapter
GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null .venv-phase4/bin/python contrib/casegraph-memory-harness-adapter/run_lifecycle.py --work-dir .venv-phase4/fresh-run --output .venv-phase4/lifecycle-report.json
```

Choose a fresh work directory for each run. `lifecycle_report.json` is the committed
semantic result; tests reproduce it in two independent real repositories. Random
local Artifact IDs, Git release hashes and timestamps are intentionally absent
from this deterministic report; tests and the runner verify actual head equality,
head changes, consumed IDs, materialized policy bytes and restart behavior.
The runner refuses to overwrite an existing run and performs no failure injection.
Failure and rejection cases are exercised by integration tests.

`retained_evaluation_report.json` remains the Phase 3 protocol report, rebuilt by
`build_report.py`. Its serialization/provenance rates are proxies, costs are fixture
estimates, and quality/negative transfer stay unmeasured there. Use the lifecycle
report for the synthetic routing task results. Neither report is a model benchmark.

See `verification_report.md` for commands, results and environment limitations.
