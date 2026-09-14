"""Tinker's inference runtime: text chat over the immutable sampler a frozen artifact names.

:class:`TinkerInferenceRuntime` owns request admission and the serving
version, activates selected candidates behind Reef's commit gate and binds
the head Reef publishes; :class:`TinkerInferenceHandler` renders, samples and
shapes each request. The module-level helpers validate OpenAI chat requests
and shape the responses.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from reef_client.sse import synthesize_sse_events

from reef.artifact.artifact import Artifact
from reef.runtime.interfaces import (
    ActivatedModel,
    InferenceHandler,
    InferenceRuntime,
    InferenceStream,
    ModelCandidate,
    StaleCandidate,
    TrainingRuntimeError,
    UpstreamStatusError,
)
from reef.train.tinker_backend.checkpoint import TinkerCheckpoint
from reef.train.tinker_backend.client import SampleResult
from reef.train.tinker_backend.runtime import TinkerCheckpointStore

TINKER_URL = "https://tinker.thinkingmachines.ai"


class TinkerInferenceRuntime(InferenceRuntime):
    """Serve the store's active snapshot and follow Reef's publication of new ones."""

    def __init__(self, store: TinkerCheckpointStore, *, base_url: str = TINKER_URL) -> None:
        super().__init__(base_url=base_url, inference_timeout_s=store.config.inference_timeout_s)
        self._store = store
        self._handler = TinkerInferenceHandler(store)
        # The base or recovered checkpoint the store opened with is the published head.
        self.mark_published()

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._handler

    @property
    def model_path(self) -> str:
        return self._store.model

    def serving_runtime_load_id(self) -> str:
        with self._store.lock:
            return self._store.version

    def restore_checkpoint(self, artifact: Artifact) -> str:
        # Rollback validates the target first; activate_checkpoint binds the
        # republished artifact. No mutable remote serving slot needs restoring.
        return self._store.snapshot(artifact)[1]

    def activate_checkpoint(self, artifact: Artifact) -> str:
        """Bind the recovered or republished artifact's snapshot before the scenario serves it."""
        store = self._store
        with store.lock:
            checkpoint, version = store.snapshot(artifact)
            if store.pending is not None:
                pending = TinkerCheckpoint.read(Path(store.pending.checkpoint_path))
                if pending != checkpoint:
                    # Reload after a failed publication restores the durable head.
                    store.pending = None
            if store.pending is None and store.active_release != artifact.ref.release_id:
                # A rollback is a new serving update even if its immutable
                # sampler was served earlier in this incarnation.
                checkpoint, version = store.new_load(checkpoint)
            store.bind(checkpoint, version, artifact.ref.release_id)
            if store.pending is None:
                self.mark_published()
            return version

    def activate_candidate(self, candidate: ModelCandidate) -> ActivatedModel:
        store = self._store
        with store.lock:
            known, incumbent = store.known_candidate(candidate)
            if store.pending is not None and store.pending.candidate_id == candidate.candidate_id:
                return ActivatedModel(candidate.candidate_id, store.version)
            if incumbent != store.version:
                raise StaleCandidate
            checkpoint = TinkerCheckpoint.read(Path(known.checkpoint_path))
            store.active, store.version = store.remember(checkpoint)
            store.pending = known
            return ActivatedModel(candidate.candidate_id, store.version)

    def acknowledge_publication(self, training_job_id: str) -> None:
        """Release the activated candidate once Reef committed exactly its training job."""
        with self._store.lock:
            if self._store.pending is None:
                return
            if training_job_id != self._store.pending.training_job_id:
                raise TrainingRuntimeError("Tinker candidate does not match Reef's committed training job")
            self._store.pending = None

    def shutdown(self) -> None:
        self.pause_admission()
        self._store.close()


class TinkerInferenceHandler(InferenceHandler):
    """Serve text chat from the immutable sampler the frozen artifact resolves to."""

    def __init__(self, store: TinkerCheckpointStore) -> None:
        self._store = store

    async def inference(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path != "/v1/chat/completions":
            raise UpstreamStatusError("Tinker supports /v1/chat/completions", status=400)
        messages, params = chat_request(payload)
        return await asyncio.to_thread(self._sample, artifact, messages, params, payload)

    def _sample(
        self, artifact: Artifact, messages: list[dict[str, str]], params: dict[str, Any], payload: dict[str, Any]
    ) -> dict[str, Any]:
        checkpoint, version = self._store.snapshot(artifact)
        client = self._store.client
        prompt = client.render(messages, template_kwargs=payload.get("chat_template_kwargs") or {})
        if not prompt:
            raise ValueError("Tinker chat template produced an empty prompt")
        result = client.sample(checkpoint, prompt, params)
        content = client.decode(result.tokens)
        return chat_completion(payload.get("model", self._store.model), messages, prompt, result, content, version)

    async def inference_stream(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> InferenceStream:
        return chat_stream(await self.inference(artifact, path, payload), payload)


def chat_completion(
    model: str,
    messages: list[dict[str, str]],
    prompt: list[int],
    result: SampleResult,
    content: str,
    runtime_load_id: str,
) -> dict[str, Any]:
    """An OpenAI chat completion plus the private training capture Reef records."""
    message = {"role": "assistant", "content": content}
    finish = "length" if result.stop_reason == "length" else "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(result.tokens),
            "total_tokens": len(prompt) + len(result.tokens),
        },
        "training": {
            "tokens": [*prompt, *result.tokens],
            "loss_mask": [1] * len(result.tokens),
            "rollout_log_probs": list(result.logprobs),
            "prompt_length": len(prompt),
            "response_length": len(result.tokens),
            "runtime_load_id": runtime_load_id,
            "request_messages": messages,
            "response_message": message,
            "finish_reason": finish,
        },
    }


def chat_stream(response: dict[str, Any], payload: dict[str, Any]) -> InferenceStream:
    """Emit a completed response as buffered OpenAI SSE, keeping it as the record."""
    include_usage = bool((payload.get("stream_options") or {}).get("include_usage"))

    async def chunks() -> AsyncIterator[bytes]:
        for event in synthesize_sse_events(response, include_usage=include_usage):
            yield event.encode()

    return InferenceStream(
        status=200, headers={"Content-Type": "text/event-stream"}, chunks=chunks(), record_response=response
    )


def chat_request(payload: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Validate a text chat request into its messages and Tinker sampling parameters."""
    supported = {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "n",
        "return_meta_info",
        "chat_template_kwargs",
    }
    unknown = payload.keys() - supported
    if unknown or payload.get("n", 1) != 1:
        raise UpstreamStatusError(f"unsupported Tinker chat options: {sorted(unknown)}; n must be 1", status=400)
    raw = payload.get("messages")
    if not isinstance(raw, list) or not raw:
        raise UpstreamStatusError("Tinker requires non-empty text messages", status=400)
    messages = []
    for message in raw:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or message["role"] not in {"system", "user", "assistant"}
            or not isinstance(message["content"], str)
        ):
            raise UpstreamStatusError("Tinker currently supports text system/user/assistant messages", status=400)
        messages.append(dict(message))
    params: dict[str, Any] = {
        key: payload[key] for key in ("temperature", "top_p", "top_k", "seed", "stop") if key in payload
    }
    params["max_tokens"] = payload.get("max_completion_tokens", payload.get("max_tokens", 1024))
    maximum = params["max_tokens"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise UpstreamStatusError("Tinker max_tokens must be a positive integer", status=400)
    template = payload.get("chat_template_kwargs", {})
    if (
        not isinstance(template, dict)
        or template.keys() - {"enable_thinking"}
        or any(not isinstance(value, bool) for value in template.values())
    ):
        raise UpstreamStatusError("Tinker chat_template_kwargs supports only boolean enable_thinking", status=400)
    stream_options = payload.get("stream_options", {})
    if not isinstance(stream_options, dict) or stream_options.keys() - {"include_usage"}:
        raise UpstreamStatusError("unsupported Tinker stream_options", status=400)
    for field, minimum, maximum_value in (("temperature", 0, None), ("top_p", 0, 1)):
        value = params.get(field, 1.0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < minimum
            or (field == "top_p" and value == 0)
            or (maximum_value is not None and value > maximum_value)
        ):
            raise UpstreamStatusError(f"invalid Tinker {field}", status=400)
    top_k = params.get("top_k", -1)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or (top_k != -1 and top_k <= 0):
        raise UpstreamStatusError("Tinker top_k must be -1 or a positive integer", status=400)
    return messages, params
