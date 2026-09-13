"""Inference request handlers and provider response streams.

``InferenceRuntime`` (``base.py``) owns the lifecycle and *composes* an
``InferenceHandler``; this module defines the request contract and the streaming
wrapper the service forwards. HTTP execution and provider configuration live
in ``adapters.http``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.errors import ReefError


class UpstreamStatusError(ReefError):
    """An upstream service answered an inference request with an error status.

    Carries ``status`` so the service can hand the caller the upstream's own
    4xx and message rather than collapsing both into an opaque 500. That
    distinction is load-bearing: agents correct themselves from these bodies
    (shrinking ``max_tokens`` when the engine reports a context overflow, for
    instance), and an opaque 500 leaves them retrying the identical request
    until they give up.
    """

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


class InferenceHandler(ABC):
    """Execute inference for a selected artifact without implicitly materializing it."""

    @classmethod
    def from_config(
        cls,
        upstream_url: str,
        *,
        model_path: str,
        timeout_s: float,
        **config: Any,
    ) -> InferenceHandler:
        """Construct a configured handler; direct injection needs only inference()."""
        raise ValueError(f"{cls.__name__} does not support deployment configuration")

    def reconnect(self, upstream_url: str) -> None:
        """Retarget a managed endpoint, preserving handler-specific configuration."""
        raise RuntimeError(f"{type(self).__name__} does not support inference endpoint replacement")

    @abstractmethod
    async def inference(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the provider response for one native request payload."""

    async def inference_stream(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> InferenceStream:
        """Return a stream for one native request payload.

        Custom handlers that only implement buffered inference keep working: the
        default implementation exposes their JSON response as one chunk. HTTP
        handlers override this method to preserve provider-native streaming.
        """
        value = await self.inference(artifact, path, payload)

        async def chunks() -> AsyncIterator[bytes]:
            yield json.dumps(value, ensure_ascii=False).encode()

        return InferenceStream(
            status=200,
            headers={"Content-Type": "application/json"},
            chunks=chunks(),
        )


class InferenceStream:
    """One open provider response whose bytes can be forwarded incrementally."""

    def __init__(
        self,
        *,
        status: int,
        headers: Mapping[str, str],
        chunks: AsyncIterator[bytes],
        close: Callable[[], Awaitable[None]] | None = None,
        record_response: Mapping[str, Any] | None = None,
        record_response_pending: bool = False,
    ) -> None:
        self.status = status
        self.headers = dict(headers)
        self.chunks = chunks
        self._close = close
        self._closed = False
        self.record_response = None if record_response is None else dict(record_response)
        # Some custom handlers can only construct their exact, provider-neutral
        # recording response after the upstream stream reaches its terminal
        # event. RequestService keeps durable admission open for those streams
        # and validates the completed capture before accepting the record.
        self.record_response_pending = record_response_pending

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close is not None:
            await self._close()


__all__ = [
    "InferenceHandler",
    "InferenceStream",
    "UpstreamStatusError",
]
