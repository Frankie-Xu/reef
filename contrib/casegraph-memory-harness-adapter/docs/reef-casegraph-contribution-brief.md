# CaseGraph contribution boundary

The current implementation, actual Reef API mapping, corrected report lineage,
frozen input policy and reproducible commands are in [the README](../README.md).

Phase 4 adds a real CPU-only recipe lifecycle using ScenarioFactory, Trainer,
SQLite scenario storage and Git LFS artifacts. The candidate is a deterministic
feedback-routing table, evaluated on frozen held-out synthetic tasks before the
real Scenario commit. No model, clinical, neural-training or deployed-inference
benefit is claimed. Retained has only one case.

The Phase 3 protocol report remains explicitly separate from the new measured
synthetic lifecycle report. Old hand-entered scores and duplicate JSON snapshots
were removed in Phase 3. No current RFC acceptance or upstream issue status is
asserted; this implementation is scoped to the existing fork contribution.
