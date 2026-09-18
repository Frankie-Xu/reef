"""Provider calls: the non-chat model routes Reef relays and records as a summary.

A harness reaches image, embedding, speech and decision models through the same
Reef service, scenario and token as its chat calls. Reef forwards the request body
unchanged to OpenRouter (see :mod:`reef.inference.openrouter`), whatever serves
the chat, and relays the response byte for byte; the record keeps a summary
instead of the body, so generated media and embedding vectors never enter record
storage or a proposer's prompt. The record names its route under
``metadata.reef_endpoint``, which is how trajectories and processors tell a
provider call from a chat exchange.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

#: The routes a client calls on Reef for a provider call.
PROVIDER_ROUTE_PATHS: tuple[str, ...] = ("/v1/images", "/v1/embeddings", "/v1/audio/speech", "/v1/decisions")

#: The record metadata key naming a provider call's route.
ENDPOINT_KEY = "reef_endpoint"
#: Strings longer than this (base64 media, data URLs) are recorded as their length and checksum.
MAX_RECORDED_TEXT_CHARS = 2048
#: Number lists longer than this (embedding vectors) are recorded as their length.
MAX_RECORDED_NUMBERS = 64


def compact(value: Any) -> Any:
    """``value`` with long strings and number vectors replaced by a description of what was there.

    Shape-agnostic on purpose: provider response formats differ and change (the
    decisions API is an alpha), so no field names are assumed.
    """
    if isinstance(value, str):
        if len(value) <= MAX_RECORDED_TEXT_CHARS:
            return value
        checksum = hashlib.sha256(value.encode()).hexdigest()
        return {"omitted": "text", "chars": len(value), "sha256": checksum}
    if isinstance(value, Mapping):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, list):
        is_vector = all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
        if is_vector and len(value) > MAX_RECORDED_NUMBERS:
            return {"omitted": "vector", "length": len(value)}
        return [compact(item) for item in value]
    return value


def provider_call_payload(path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """The record payload of a provider call: the compacted request body with its route in metadata."""
    recorded = compact(payload)
    metadata = recorded.get("metadata")
    recorded["metadata"] = {**(metadata if isinstance(metadata, Mapping) else {}), ENDPOINT_KEY: path}
    return recorded


def provider_call_response(
    status: int,
    headers: Mapping[str, str],
    body: bytes,
    *,
    complete: bool,
    error: str | None = None,
) -> dict[str, Any]:
    """The recorded response of a provider call: a JSON body compacted, any other body as its size and checksum."""
    lowered = {name.lower(): value for name, value in headers.items()}
    content_type = lowered.get("content-type", "")
    response: dict[str, Any] = {"status": status, "content_type": content_type, "complete": complete}
    if error is not None:
        response["error"] = error
    encoded = lowered.get("content-encoding", "identity").lower() != "identity"
    if "json" in content_type.lower() and not encoded:
        try:
            response["body"] = compact(json.loads(body))
            return response
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    response["bytes"] = len(body)
    response["sha256"] = hashlib.sha256(body).hexdigest()
    return response


def provider_call_endpoint(payload: Mapping[str, Any]) -> str | None:
    """The route of a recorded provider call, or ``None`` for any other record payload."""
    metadata = payload.get("metadata")
    endpoint = metadata.get(ENDPOINT_KEY) if isinstance(metadata, Mapping) else None
    return endpoint if endpoint in PROVIDER_ROUTE_PATHS else None


__all__ = [
    "ENDPOINT_KEY",
    "PROVIDER_ROUTE_PATHS",
    "compact",
    "provider_call_endpoint",
    "provider_call_payload",
    "provider_call_response",
]
