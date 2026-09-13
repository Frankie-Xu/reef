"""Explicit test assembly of Reef coordination and the two native adapters."""

from reef.inference.sglang.operations import SGLangInferenceOperations
from reef.runtime.executor.ray import RayExecutor
from reef.runtime.training_job.coordinator import TrainingCoordinator
from reef.train.slime_backend.reef_adapters.bridge import SlimeTrainingOperations


class FixtureCoordinator(TrainingCoordinator):
    """Preserve explicit fake-group versions in older wire contract fixtures.

    Allocation with canonical, advancing IDs is tested separately against the
    unmodified TrainingCoordinator in test_training_coordinator.py.
    """

    def _next_runtime_load_id(self):
        return self._training._group.next_runtime_load_id()


class FixtureInferenceOperations(SGLangInferenceOperations):
    def initialize_version(self, runtime_load_id):
        # These fixtures begin with the selected identity already loaded.
        pass


def build_slime_coordinator(actor_group, inference, **kwargs) -> TrainingCoordinator:
    training = SlimeTrainingOperations(actor_group, **kwargs)
    training.context.runtime_load_id = training.current_runtime_load_id()
    receiver = FixtureInferenceOperations(RayExecutor.from_workers([inference]))
    return FixtureCoordinator(training, receiver)
