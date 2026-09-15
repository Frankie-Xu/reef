"""Small, process-local measurements that remain readable during slow operations."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock


class OperationMetrics:
    """Measure serial operations without holding a lock while they execute.

    Counters belong to this instance and reset when its owner is rebuilt.
    Sampling performs no provider calls and never consumes recorded values.
    """

    def __init__(self, operations: tuple[str, ...]) -> None:
        self.lock = Lock()
        self.started_at_seconds = time.time()
        self.metrics: dict[str, float | int] = {}
        for operation in operations:
            for quantity in ("active", "elapsed_seconds", "completed_total", "failed_total", "duration_seconds_total"):
                self.metrics[f"{operation}/{quantity}"] = 0
        self.operation_start_seconds: dict[str, float] = {}

    @contextmanager
    def measure(self, operation: str) -> Iterator[None]:
        started_monotonic_seconds = time.monotonic()
        with self.lock:
            self.operation_start_seconds[operation] = started_monotonic_seconds
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            duration_seconds = time.monotonic() - started_monotonic_seconds
            with self.lock:
                self.operation_start_seconds.pop(operation, None)
                self.metrics[f"{operation}/active"] = 0
                self.metrics[f"{operation}/elapsed_seconds"] = 0.0
                self.metrics[f"{operation}/last_duration_seconds"] = duration_seconds
                outcome_metric = f"{operation}/completed_total" if succeeded else f"{operation}/failed_total"
                self.metrics[outcome_metric] = self.metrics.get(outcome_metric, 0) + 1
                duration_metric = f"{operation}/duration_seconds_total"
                self.metrics[duration_metric] = self.metrics.get(duration_metric, 0.0) + duration_seconds

    def snapshot(self) -> dict[str, float | int]:
        with self.lock:
            metrics = {"started_at_seconds": self.started_at_seconds, **self.metrics}
            for operation, started_monotonic_seconds in self.operation_start_seconds.items():
                metrics[f"{operation}/active"] = 1
                metrics[f"{operation}/elapsed_seconds"] = time.monotonic() - started_monotonic_seconds
            return metrics
