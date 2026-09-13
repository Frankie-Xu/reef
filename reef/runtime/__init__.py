"""Reef's model scheduling contracts and coordination.

``InferenceRuntime`` owns request execution and admission. ``TrainingRuntime``
prepares training jobs and exports checkpoints. Neither inherits the other.
``scheduler.RuntimeScheduler`` coordinates candidate selection and durable
artifact commit acknowledgement across the two interfaces.

``deployment.ModelDeployment`` owns component startup, weight-transport
attachment, health and reverse-order cleanup. ``training_job.coordinator``
implements publication, recovery, LoRA residency and colocated resource
handoffs through separate ``TrainingOperations`` and ``InferenceOperations``
contracts. Backends supply native model operations and weight transport.

``model_config.ModelConfig`` is the in-memory model selection shared by a
scenario and its recipe. It has no file paths or persistence behavior.
``inference_control`` coordinates engine pause/recovery and transport reconnect;
``health_monitor`` drains engine probes before lifecycle changes;
``inference_memory`` pairs acknowledged memory release/resume operations;
``weight_update`` supplies transport-lock failure and phase semantics.

Concrete inference implementations live in ``reef.inference`` and training
implementations in ``reef.train``. This package imports neither integration
package, including from function bodies. Protocol adapters and executors here
connect Reef to those implementations without choosing a model framework.

Malformed results and missing capabilities surface as contract errors, never
as silent fallbacks. The default ``restore_checkpoint`` refuses rather than
moving the artifact head under an engine that kept newer weights. Surfaces
consume runtimes structurally through ``ServingRuntime`` and ``WeightRuntime``
in ``surface/base.py``.

To add a runtime kind, subclass ``RuntimeFactory``, set ``kind`` and register
it with ``@register_runtime_kind`` in a module imported at boot. A config
``type`` may also name a dotted ``package.module:factory_name`` reference.
"""

from reef.runtime.adapters.executor_inference import ExecutorInferenceRuntime
from reef.runtime.adapters.executor_runtime import ExecutorTrainingRuntime
from reef.runtime.adapters.inference_proxy import InferenceProxyRuntime
from reef.runtime.adapters.ray_runtime import (
    RayRuntimeError,
    RayTrainGroupHandle,
    RemoteRayTrainGroupHandle,
    connect_ray_runtime,
)
from reef.runtime.base import InferenceRuntime, PreparedTrainingStep, TrainingJobResult, TrainingRuntime
from reef.runtime.candidates import ActivatedModel, ModelCandidate
from reef.runtime.executor import Executor, ExecutorConfig, ExecutorFuture, WorkerSpec
from reef.runtime.proxy import resolve_proxy_runtime
from reef.runtime.registry import (
    RuntimeConfigError,
    RuntimeFactory,
    RuntimeRegistry,
    register_runtime_kind,
    runtime_factory_for,
    runtime_kinds,
)
from reef.runtime.training_group import ExecutorTrainGroupHandle, TrainingGroupHandle, TrainingRuntimeError

__all__ = [
    "ActivatedModel",
    "Executor",
    "ExecutorConfig",
    "ExecutorFuture",
    "ExecutorInferenceRuntime",
    "ExecutorTrainGroupHandle",
    "ExecutorTrainingRuntime",
    "InferenceProxyRuntime",
    "InferenceRuntime",
    "ModelCandidate",
    "PreparedTrainingStep",
    "RayRuntimeError",
    "RayTrainGroupHandle",
    "RemoteRayTrainGroupHandle",
    "RuntimeConfigError",
    "RuntimeFactory",
    "RuntimeRegistry",
    "TrainingGroupHandle",
    "TrainingJobResult",
    "TrainingRuntime",
    "TrainingRuntimeError",
    "WorkerSpec",
    "connect_ray_runtime",
    "register_runtime_kind",
    "resolve_proxy_runtime",
    "runtime_factory_for",
    "runtime_kinds",
]
