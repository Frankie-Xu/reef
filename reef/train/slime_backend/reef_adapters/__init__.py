"""Native Slime training operations and checkpoint storage.

Reef's runtime coordinator owns the training job API, admission, durable marker
transitions and the serving commit gate. Slime adapters supply sample packing,
optimizer execution, checkpoint I/O and tensor transport. Inference adapters
live in ``reef.inference`` and are assembled independently by Reef.
"""

from importlib import import_module
from typing import Any

from reef.train.slime_backend.reef_adapters.training_job.storage import RetentionConfig

__all__ = ["RetentionConfig", "SlimeTrainingOperations"]


def __getattr__(name: str) -> Any:
    if name != "SlimeTrainingOperations":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module(f"{__name__}.bridge").SlimeTrainingOperations
    globals()[name] = value
    return value
