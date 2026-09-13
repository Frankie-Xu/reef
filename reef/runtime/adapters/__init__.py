"""Generic connections between Reef runtime contracts and model services.

``http`` owns provider requests and inference-proxy construction.
``executor_training`` and ``executor_inference`` adapt worker control to the
separate runtime interfaces. ``training_group`` defines their training RPC
contract; ``ray`` connects to existing named Ray workers. Connection config
lives in ``config``. Native model integrations belong in ``reef.inference``
or ``reef.train``.
"""

from reef.runtime.adapters.executor_inference import ExecutorInferenceRuntime
from reef.runtime.adapters.executor_training import ExecutorTrainingRuntime
from reef.runtime.adapters.http import InferenceProxyRuntime
from reef.runtime.adapters.ray import (
    RayRuntimeError,
    RayTrainGroupHandle,
    RemoteRayTrainGroupHandle,
    connect_ray_runtime,
)

__all__ = [
    "ExecutorInferenceRuntime",
    "ExecutorTrainingRuntime",
    "InferenceProxyRuntime",
    "RayRuntimeError",
    "RayTrainGroupHandle",
    "RemoteRayTrainGroupHandle",
    "connect_ray_runtime",
]
