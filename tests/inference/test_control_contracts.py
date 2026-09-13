"""Control integrations must implement the complete inherited interface."""

import pytest

from reef.runtime.control.health import EngineHealthChecks, EngineHealthTarget
from reef.runtime.control.inference import InferenceEngines, InferenceMonitor, WeightUpdateConnection
from reef.runtime.control.memory import InferenceMemoryOperations
from reef.runtime.executor.failure import ExecutorFailureListener
from reef.runtime.weights.residency import AdapterEngine


@pytest.mark.parametrize(
    "interface",
    [
        InferenceEngines,
        InferenceMonitor,
        WeightUpdateConnection,
        EngineHealthChecks,
        EngineHealthTarget,
        InferenceMemoryOperations,
        AdapterEngine,
        ExecutorFailureListener,
    ],
)
def test_incomplete_control_integration_cannot_be_constructed(interface):
    class IncompleteIntegration(interface):
        pass

    with pytest.raises(TypeError, match="abstract"):
        IncompleteIntegration()
