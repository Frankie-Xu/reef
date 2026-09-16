"""Abstract generation and validation hooks for processors that produce tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from reef.core.tasks import HarborTask, TaskGenerationRequest, TaskValidationResult
from reef.train.processors.base import DataProcessor


class TaskGenerationProcessor(DataProcessor, ABC):
    """The method contract for turning records into validated Harbor tasks.

    This ABC declares the two asynchronous hooks only. It does not yet supply
    background execution, task publication, retries, or record-to-batch assembly.
    Implementing the hooks alone retains DataProcessor's no-update lifecycle.

    A lifecycle implementation must run these hooks outside the trainer lock,
    preserve source records until acknowledgement, and expose only validated,
    complete task directories through TaskItem. Reserved directories must remain
    accessible to the consumer until consumption finishes. Its synchronous
    ingest/ready/build_batch methods must not wait for generation or validation.
    """

    @abstractmethod
    async def generate(self, request: TaskGenerationRequest) -> HarborTask:
        """Generate one task from records belonging to this processor's scenario.

        Return the task specification without publishing a directory or starting
        training. Preserve the request's ordered source record ids in the task's
        source_agent_record_ids. Model calls and other slow work belong here;
        generation failures propagate to the lifecycle implementation.
        """

    @abstractmethod
    async def validate(self, task_path: Path) -> TaskValidationResult:
        """Check a materialized candidate before it becomes a ready TaskItem.

        Run the method's required structural and execution checks without
        modifying the candidate. Return task defects as validation errors;
        raise execution failures so the lifecycle can handle retries separately.
        Validation alone does not publish a task or consume its source records.
        """
