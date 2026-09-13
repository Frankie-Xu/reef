"""Compose separately selected training and inference runtime connections.

Generic executor and Ray config kinds retain their HTTP request default.
Native model runtimes are supplied by their integration factories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any

from reef.inference.http import HttpInferenceHandler
from reef.inference.runtime import ExecutorInferenceRuntime
from reef.runtime.deployment import ExecutorRuntimeConfig, RayRuntimeConfig, RuntimeConfigError, RuntimeFactory
from reef.runtime.executor import Executor, ExecutorConfig, WorkerSpec
from reef.runtime.executor.connection import (
    DEFAULT_ACTOR_NAME,
    DEFAULT_NAMESPACE,
    CoordinatorClient,
    ExecutorCoordinatorClient,
    connect_ray_coordinator,
)
from reef.runtime.interfaces import InferenceHandler, InferenceRuntime, TrainingRuntime
from reef.train.runtime import ExecutorTrainingRuntime


def connect_executor_runtimes(
    *,
    train_group_handle: CoordinatorClient,
    inference: InferenceRuntime | None = None,
    inference_url: str | None = None,
    model_path: str = "",
    inference_timeout_s: float = 300.0,
    max_staleness: int = 0,
    inference_handler_factory: type[InferenceHandler] = HttpInferenceHandler,
    inference_handler_config: Mapping[str, Any] | None = None,
) -> tuple[TrainingRuntime, InferenceRuntime]:
    """Assemble independent components over the existing deployment connection."""
    if inference is not None and inference_url is not None:
        raise ValueError("pass inference or inference_url, not both")
    training = ExecutorTrainingRuntime(train_group_handle, max_staleness=max_staleness)
    if inference is None:
        inference = ExecutorInferenceRuntime(
            control=train_group_handle,
            inference_url=inference_url,
            model_path=model_path,
            inference_timeout_s=inference_timeout_s,
            inference_handler_factory=inference_handler_factory,
            inference_handler_config=inference_handler_config,
        )
    return training, inference


def connect_ray_runtime(
    *,
    inference_url: str | None = None,
    actor_name: str = DEFAULT_ACTOR_NAME,
    namespace: str = DEFAULT_NAMESPACE,
    ray_address: str | None = None,
    model_path: str = "",
    inference_timeout_s: float = 300.0,
    train_timeout_s: float | None = None,
    max_staleness: int = 0,
    inference_handler_factory: type[InferenceHandler] = HttpInferenceHandler,
    inference_handler_config: Mapping[str, Any] | None = None,
) -> tuple[TrainingRuntime, InferenceRuntime]:
    """Connect to a named training actor and return separate training and inference runtimes.

    Reef and the backend run as separate services in one Ray cluster.
    ``namespace`` must match the namespace used when the backend actor was
    created. ``inference_url`` defaults to the address the actor reports.
    """
    return connect_executor_runtimes(
        train_group_handle=connect_ray_coordinator(
            actor_name=actor_name,
            namespace=namespace,
            ray_address=ray_address,
            train_timeout_s=train_timeout_s,
            inference_timeout_s=inference_timeout_s,
        ),
        inference_url=inference_url,
        model_path=model_path,
        inference_timeout_s=inference_timeout_s,
        max_staleness=max_staleness,
        inference_handler_factory=inference_handler_factory,
        inference_handler_config=inference_handler_config,
    )


def _executor_config(value: Mapping[str, Any]) -> ExecutorConfig:
    workers = value.get("workers", ())
    if not isinstance(workers, Sequence) or isinstance(workers, (str, bytes)):
        raise RuntimeConfigError("runtime.executor.workers must be a sequence of worker specifications")
    specs = []
    for worker in workers:
        if isinstance(worker, WorkerSpec):
            specs.append(worker)
        elif isinstance(worker, Mapping):
            try:
                specs.append(WorkerSpec(**dict(worker)))
            except (TypeError, ValueError) as exc:
                raise RuntimeConfigError(f"invalid runtime.executor worker: {exc}") from exc
        else:
            raise RuntimeConfigError("runtime.executor.workers entries must be WorkerSpec objects or mappings")
    try:
        return ExecutorConfig(
            backend=value.get("backend", "auto"),
            workers=tuple(specs),
            options=value.get("options", {}),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(f"invalid runtime.executor configuration: {exc}") from exc


class ExecutorTrainingRuntimeFactory(RuntimeFactory):
    """Create a training coordinator using a configured executor.

    The executor entry accepts an existing Executor, an ExecutorConfig, or a
    mapping with backend, workers, and options. The worker at coordinator_rank
    (default 0) implements CoordinatorClient's methods; it may manage its
    own model-parallel worker group.
    """

    kind = "executor_training"

    def config_type(self) -> type:
        return ExecutorRuntimeConfig

    def parse_config(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
        injected = {
            key: config[key] for key in ("executor", "inference", "inference_handler_factory") if key in config
        }
        values = super().parse_config({key: value for key, value in config.items() if key not in injected}, environ)
        return {**values, **injected}

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> tuple[TrainingRuntime, InferenceRuntime]:
        value = config.get("executor")
        if isinstance(value, Mapping):
            value = _executor_config(value)
        created = False
        if isinstance(value, ExecutorConfig):
            executor = Executor.create(value)
            created = True
        elif isinstance(value, Executor):
            executor = value
        else:
            raise RuntimeConfigError("runtime.executor must be an Executor, ExecutorConfig, or configuration mapping")
        try:
            handle = ExecutorCoordinatorClient(
                executor,
                rank=config.get("coordinator_rank", 0),
                timeout_s=(
                    config["train_timeout_s"]
                    if config.get("train_timeout_s") is not None
                    else config.get("inference_timeout_s", 300.0)
                ),
            )
            kwargs: dict[str, Any] = {"train_group_handle": handle, "model_path": model_path}
            for key in (
                "inference",
                "inference_url",
                "inference_timeout_s",
                "max_staleness",
                "inference_handler_factory",
                "inference_handler_config",
            ):
                if key in config:
                    kwargs[key] = config[key]
            return connect_executor_runtimes(**kwargs)
        except BaseException:
            if created:
                with suppress(Exception):
                    executor.shutdown()
            raise


class RayTrainingRuntimeFactory(RuntimeFactory):
    """Build separate training and inference runtimes from runtime configuration.

    The config mirrors :func:`connect_ray_runtime`'s keyword arguments. A
    ``connect`` entry may inject an alternative connector callable (tests use
    this to stub the Ray cluster); it defaults to :func:`connect_ray_runtime`.
    """

    kind = "ray_training"

    def config_type(self) -> type:
        return RayRuntimeConfig

    def parse_config(self, config: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
        # Existing Python assembly can inject these objects. They are not YAML
        # fields and must never be serialized through the argument parser.
        injected = {key: config[key] for key in ("connect", "inference_handler_factory") if key in config}
        values = super().parse_config({key: value for key, value in config.items() if key not in injected}, environ)
        return {**{key: value for key, value in values.items() if key in config}, **injected}

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> tuple[TrainingRuntime, InferenceRuntime]:
        connect = config.get("connect", connect_ray_runtime)
        if not callable(connect):
            raise RuntimeConfigError("runtime.connect must be callable")
        kwargs: dict[str, Any] = {"model_path": model_path}
        for key in (
            "inference_url",
            "actor_name",
            "namespace",
            "ray_address",
            "inference_timeout_s",
            "train_timeout_s",
            "max_staleness",
            "inference_handler_factory",
            "inference_handler_config",
        ):
            if key in config:
                kwargs[key] = config[key]
        runtime = connect(**kwargs)
        if not (
            isinstance(runtime, tuple)
            and len(runtime) == 2
            and isinstance(runtime[0], TrainingRuntime)
            and isinstance(runtime[1], InferenceRuntime)
        ):
            for component in runtime if isinstance(runtime, tuple) else (runtime,):
                with suppress(Exception):
                    component.shutdown()
            raise RuntimeConfigError(
                f"runtime connector returned {type(runtime).__name__}, not a (TrainingRuntime, InferenceRuntime) pair"
            )
        return runtime
