"""SGLang receiver results and failures crossing the runtime control boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from reef.inference.sglang.config import SGLangConfig
from reef.inference.sglang.operations import SGLangInferenceOperations
from reef.inference.sglang.service import INFERENCE_PROTOCOL, SGLangInferenceService
from reef.runtime.deployment import InferenceConnection
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.executor.uniproc import UniProcExecutor


class Receiver:
    def __init__(self, versions=("weights:1", "weights:1")):
        self.versions = versions
        self.paused = False
        self.closed = False

    def inference_url(self):
        return "http://inference:8000"

    def get_runtime_load_ids(self):
        return self.versions

    def pause_generation_for_update(self):
        self.paused = True

    def continue_generation_after_update(self):
        self.paused = False

    def prepare_training_connection(self):
        self.paused = True

    def shutdown(self):
        self.closed = True


def test_service_adapts_only_compatible_connections_without_owning_workers():
    receiver = Receiver()
    executor = UniProcExecutor.from_workers([receiver])
    service = SGLangInferenceService(SGLangConfig("model", 1, 1, 1))
    operations = service.operations(InferenceConnection(INFERENCE_PROTOCOL, executor))

    assert operations.inference_url() == "http://inference:8000"
    assert operations.runtime_load_ids() == ("weights:1", "weights:1")
    operations.pause()
    assert receiver.paused
    operations.resume()
    assert not receiver.paused
    service.close()
    assert not receiver.closed
    assert operations.runtime_load_ids() == ("weights:1", "weights:1")

    with pytest.raises(ValueError, match="incompatible SGLang"):
        service.operations(InferenceConnection("other-protocol", executor))


def test_preparing_weight_transfer_fences_only_a_compatible_receiver():
    receiver = Receiver()
    executor = UniProcExecutor.from_workers([receiver])
    service = SGLangInferenceService(SGLangConfig("model", 1, 1, 1))
    with pytest.raises(ValueError, match="incompatible SGLang"):
        service.prepare_weight_transfer(InferenceConnection("other-protocol", executor))
    assert not receiver.paused

    connection = InferenceConnection(INFERENCE_PROTOCOL, executor)
    service.prepare_weight_transfer(connection)
    assert receiver.paused
    assert not receiver.closed


@pytest.mark.parametrize("versions", ["weights:1", [None], [1], [""], {"engine": "weights:1"}])
def test_receiver_rejects_malformed_versions_before_publication(versions):
    operations = SGLangInferenceOperations(UniProcExecutor.from_workers([Receiver(versions)]))
    with pytest.raises(RuntimeError, match="invalid runtime load IDs"):
        operations.runtime_load_ids()


def test_receiver_preserves_mixed_versions_for_coordinator_consistency_check():
    operations = SGLangInferenceOperations(UniProcExecutor.from_workers([Receiver(["weights:1", "weights:2"])]))
    assert operations.runtime_load_ids() == ("weights:1", "weights:2")


def test_receiver_preserves_pause_failure_without_attempting_resume():
    class UncertainReceiver(Receiver):
        def pause_generation_for_update(self):
            self.paused = True
            raise TimeoutError("pause acknowledgement lost")

    receiver = UncertainReceiver()
    operations = SGLangInferenceOperations(UniProcExecutor.from_workers([receiver]))
    with pytest.raises(TimeoutError, match="acknowledgement lost"):
        operations.pause()
    assert receiver.paused


@pytest.mark.parametrize("fail_second", [False, True])
def test_initial_version_stamping_keeps_receiver_paused_on_partial_failure(monkeypatch, fail_second):
    versions = ["native:0", "native:0"]

    class EngineGroup:
        def collective_rpc(self, method, *, args, timeout):
            assert method == "set_runtime_load_id"
            versions[0] = args[0]
            if fail_second:
                raise TimeoutError("second engine acknowledgement lost")
            versions[1] = args[0]

    receiver = Receiver(versions)
    receiver.get_updatable_engines_and_lock = lambda: ([object(), object()], None, 0, [], [], [])
    monkeypatch.setattr(RayExecutor, "from_workers", lambda workers: EngineGroup())
    operations = SGLangInferenceOperations(UniProcExecutor.from_workers([receiver]))
    operations.pause()
    if fail_second:
        with pytest.raises(TimeoutError, match="acknowledgement lost"):
            operations.initialize_version("reef:initial")
        assert operations.runtime_load_ids() == ("reef:initial", "native:0")
    else:
        operations.initialize_version("reef:initial")
        assert operations.runtime_load_ids() == ("reef:initial", "reef:initial")
    assert receiver.paused


@pytest.mark.parametrize("last_result", [{"success": False}, {"success": True}, None])
def test_adapter_eviction_requires_every_engine_acknowledgement(monkeypatch, last_result):
    unloaded = []

    class Engine:
        def __init__(self, result):
            self.result = result

        def unload_lora_adapter(self, *, lora_name):
            unloaded.append(lora_name)
            return self.result

    engines = [Engine({"success": True}), Engine(last_result)]
    receiver = SimpleNamespace(get_updatable_engines_and_lock=lambda: (engines, None, 0, [], [], []))

    class EngineGroup:
        def collective_rpc(self, method, *, kwargs, timeout):
            assert method == "unload_lora_adapter"
            return [engine.unload_lora_adapter(**kwargs) for engine in engines]

    monkeypatch.setattr(RayExecutor, "from_workers", lambda workers: EngineGroup())
    operations = SGLangInferenceOperations(UniProcExecutor.from_workers([receiver]))
    if last_result == {"success": False}:
        with pytest.raises(RuntimeError, match="engine kept adapter"):
            operations.unload_adapter("scenario/weights:2")
    else:
        operations.unload_adapter("scenario/weights:2")
    assert unloaded == ["scenario/weights:2", "scenario/weights:2"]
