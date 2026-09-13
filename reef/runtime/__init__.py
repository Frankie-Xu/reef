"""Reef's model scheduling contracts and coordination.

``InferenceRuntime`` owns request execution and admission. ``TrainingRuntime``
prepares training jobs and exports checkpoints. Neither inherits the other.
``scheduler.RuntimeScheduler`` coordinates candidate selection and durable
artifact commit acknowledgement across the two interfaces.

``deployment.ModelDeployment`` owns component startup, weight-transport
attachment, health and reverse-order cleanup. ``training_job.coordinator``
implements publication, recovery, LoRA residency and colocated resource
handoffs through separate ``TrainingBackend`` and ``InferenceBackend``
contracts. Backends supply native model operations and weight transport.

``model_config.ModelConfig`` is the in-memory model selection shared by a
scenario and its recipe. It has no file paths or persistence behavior.

``control`` groups inference pause/recovery, health probes and memory handoffs.
``weights`` groups version identity, candidates, LoRA residency and transfer locks.
``backends`` defines the paired native backend contracts independently of the
coordinator. ``training_job`` owns job execution and publication.
``adapters`` implements generic HTTP and executor connections and their config.
``executor`` launches and controls workers without choosing a model framework.

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
from reef.runtime.adapters.executor_training import ExecutorTrainingRuntime
from reef.runtime.adapters.http import InferenceProxyRuntime, resolve_proxy_runtime
from reef.runtime.adapters.ray import (
    RayRuntimeError,
    RayTrainGroupHandle,
    RemoteRayTrainGroupHandle,
    connect_ray_runtime,
)
from reef.runtime.adapters.training_group import ExecutorTrainGroupHandle, TrainingGroupHandle
from reef.runtime.backends import InferenceBackend, TrainingBackend
from reef.runtime.base import (
    InferenceRuntime,
    PreparedTrainingStep,
    TrainingJobResult,
    TrainingRuntime,
    TrainingRuntimeError,
)
from reef.runtime.executor import Executor, ExecutorConfig, ExecutorFuture, WorkerSpec
from reef.runtime.registry import (
    RuntimeConfigError,
    RuntimeFactory,
    RuntimeRegistry,
    register_runtime_kind,
    runtime_factory_for,
    runtime_kinds,
)
from reef.runtime.weights.candidates import ActivatedModel, ModelCandidate

__all__ = [
    "ActivatedModel",
    "Executor",
    "ExecutorConfig",
    "ExecutorFuture",
    "ExecutorInferenceRuntime",
    "ExecutorTrainGroupHandle",
    "ExecutorTrainingRuntime",
    "InferenceBackend",
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
    "TrainingBackend",
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
