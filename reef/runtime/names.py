"""Names shared across the processes of the Reef-to-backend bridge contract.

The Reef service and a training backend run as separate processes that find
each other only by name, so these values must agree on both sides:

- ``DEFAULT_ACTOR_NAME`` / ``DEFAULT_NAMESPACE`` locate the named Ray bridge
  coordinator (``connect_ray_runtime`` in the serving process and Reef's
  deployment owner and healthcheck in the driver process).
- ``LATEST_JOB_MARKER_FILENAME`` is the durable training-job marker the
  Reef writes next to the training checkpoints and must recognize across restarts.
"""

DEFAULT_ACTOR_NAME = "reef-train-bridge"
DEFAULT_NAMESPACE = "reef"
LATEST_JOB_MARKER_FILENAME = ".reef-latest-job.json"
#: Per-scenario publication history of a LoRA deployment; sits beside the marker.
SCENARIO_HISTORY_FILENAME = "reef_scenarios.json"
#: Rank-local adapter-slot snapshots of a LoRA training group; sits beside the Megatron checkpoint.
ADAPTER_SLOTS_DIRNAME = "reef_adapter_slots"

__all__ = [
    "ADAPTER_SLOTS_DIRNAME",
    "DEFAULT_ACTOR_NAME",
    "DEFAULT_NAMESPACE",
    "LATEST_JOB_MARKER_FILENAME",
    "SCENARIO_HISTORY_FILENAME",
]
