"""Text chat facade over Tinker's token-native immutable sampler.

The runtime binds these helpers to its snapshots in ``TinkerInferenceBackend``;
this module only validates OpenAI chat requests and shapes the responses.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from reef_client.sse import synthesize_sse_events

from reef.runtime.inference import InferenceStream, UpstreamStatusError
from reef.train.tinker_backend.client import SampleResult


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
