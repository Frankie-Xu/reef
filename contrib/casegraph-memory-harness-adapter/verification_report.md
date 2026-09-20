# Verification of the synthetic report/record bridge

The implementation is in the existing fork contribution directory. All fixture
records are synthetic. No PHI, model calls, GPU jobs or production writes were used.
The bridge imports actual checked-out Reef types; its integration test uses a real
SQLiteRecordStore in a temporary directory, including persistence across reopening.

## Results

- Reused repository `.venv`: Python 3.14.6. This is not the Python 3.12 CI environment.
- `.venv/bin/python -m pytest -q tests/reef_service/test_records.py contrib/casegraph-memory-harness-adapter`: **68 passed** (44 upstream storage tests, 24 contribution tests).
- The contribution uses root pytest configuration and existing unit/integration markers.
- `.venv/bin/python -m mypy`: **313 source files passed**.
- Explicit mypy of `reef_casegraph_adapter.py`, `reef_record_bridge.py`, `build_report.py`: **3 files passed**.
- `PATH="$PWD/.venv/bin:$PATH" .venv/bin/pre-commit run --all-files`: all applicable hooks passed, including repository policy checks, Ruff, isort and Black.
- `.venv/bin/ruff check contrib/casegraph-memory-harness-adapter`: passed.
- The committed report is checked against deterministic reconstruction from the fixture.
- Full test attempt, `.venv/bin/python -m pytest tests/ -q --maxfail=3`, stopped during collection with missing `ray` and `slime`. Full CI is not claimed.

The fixture digest remains
`aec88a050f4812a672d40670cc5dcd2bb2e11e07784886ac90b3333c4adebd0d`.
Regressions cover one-shot iterators, 9/16/101-case inputs, complete disjoint
assignment, observation/effective time leakage, invalid dates/provenance, tampered
identities, fixture metadata mismatch, version mismatch, duplicate replay, and
wrong feedback receipts. Report tests verify that retained event IDs do not enter
the candidate and that the input fixture remains unchanged.

The old hand-filled report was replaced. Quality and negative transfer are
unmeasured; cost values are synthetic fixture estimates. No clinical benchmark,
model improvement or actual retained-gate selection is asserted. The full Reef
observe/grow/commit path remains unimplemented; see README for exact boundaries.
