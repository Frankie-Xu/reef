"""Tinker's components for Reef's model driver: a GPU-less trainer beside a local engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from reef.runtime.deployment import DeploymentResources, InferenceResources, TrainingService, WeightTransferSession
from reef.runtime.executor import Executor
from reef.runtime.executor.placement import ModelGpuLayout, ModelGpuReservation, reserve_model_gpus
from reef.runtime.interfaces import TrainingBackend
from reef.train.tinker_backend.client import TinkerClient, TinkerSDKClient
from reef.train.tinker_backend.config import TinkerConfig

#: The SGLang control protocol the local engines speak; the trainer only loads adapters through it.
SGLANG_CONTROL_PROTOCOL = "slime-sglang-control-v2"


class TinkerDeploymentResources(InferenceResources):
    """One Ray session and the placement group the local engines run on; the trainer needs no GPU."""

    def __init__(self, layout: ModelGpuLayout, *, ray_address: str, namespace: str) -> None:
        if layout.training_gpus != 0:
            raise ValueError("a hosted trainer reserves inference GPUs only")
        self.layout = layout
        self.ray_address = ray_address
        self.namespace = namespace
        self._reservation: ModelGpuReservation | None = None
        self._started = False
        self._closed = False

    @property
    def inference_placement(self) -> Any:
        return None if self._reservation is None else self._reservation.inference

    def start(self) -> None:
        import ray

        if self._started or self._closed:
            raise RuntimeError("deployment resources can only be started once")
        if ray.is_initialized():
            raise RuntimeError("the model driver requires its own Ray client session")
        self._started = True
        ray.init(address=self.ray_address, namespace=self.namespace)
        self._reservation = reserve_model_gpus(self.layout)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._reservation is not None:
                self._reservation.release()
        finally:
            if self._started:
                import ray

                ray.shutdown()


class TinkerTrainingService(TrainingService):
    """Own the SDK client and hand the coordinator a backend once the engines are attached."""

    weight_transfer_protocol = SGLANG_CONTROL_PROTOCOL

    def __init__(self, base_model: str, config: TinkerConfig, api_key: str, *, client: TinkerClient | None = None):
        self._model = base_model
        self._config = config
        self._api_key = api_key
        self._client = client
        self._receiver: Executor | None = None
        self._backend: TrainingBackend | None = None
        self._started = False
        self._closed = False

    def start(self, resources: DeploymentResources) -> None:
        if self._started or self._closed:
            raise RuntimeError("training service can only be started once")
        self._started = True
        if self._client is None:
            self._client = TinkerSDKClient(self._model, self._config, self._api_key)

    def attach_weight_transport(self, session: WeightTransferSession) -> None:
        if not self._started or self._closed:
            raise RuntimeError("start the training service before attaching its weight transport")
        if session.protocol != self.weight_transfer_protocol:
            raise ValueError("Tinker serving through a local engine requires the SGLang control protocol")
        self._receiver = session.receiver

    def backend(self) -> TrainingBackend:
        if self._client is None or self._receiver is None or self._closed:
            raise RuntimeError("training backend requires a started service with attached engines")
        if self._backend is None:
            from reef.train.tinker_backend.backend import TinkerTrainingBackend

            self._backend = TinkerTrainingBackend(self._model, self._config, self._client, self._receiver)
        return self._backend

    def check_health(self) -> None:
        self.poll()

    def poll(self) -> None:
        if not self._started or self._closed:
            raise RuntimeError("training service is not running")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._client is not None and self._backend is None:
            self._client.close()


def sglang_inference_config(reef: Mapping[str, Any], config: TinkerConfig) -> dict[str, Any]:
    """The managed SGLang input for serving Tinker's adapters: single-node tensor-parallel engines."""
    num_gpus = int(reef["inference_num_gpus"])
    gpus_per_engine = int(reef["tensor_parallel_size"])
    options = {key.replace("-", "_"): value for key, value in dict(reef.get("inference_options") or {}).items()}
    options.update(
        model_path=reef["model_path"],
        trust_remote_code=True,
        skip_server_warmup=True,
        enable_metrics=True,
        enable_lora=True,
        max_lora_rank=config.lora_rank,
        max_loaded_loras=config.max_loaded_adapters,
        max_loras_per_batch=config.max_loaded_adapters,
    )
    options.setdefault("lora_target_modules", ["all"])
    return {
        "model_path": reef["model_path"],
        "num_gpus": num_gpus,
        "gpus_per_engine": gpus_per_engine,
        "gpus_per_node": max(num_gpus, gpus_per_engine),
        "options": options,
        "executor": "auto",
    }
