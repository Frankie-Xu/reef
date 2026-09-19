"""Provider calls: the non-chat model routes Reef relays and records as a summary.

The deployment's multimodal provider (a preset, its address and key; see
:func:`resolve_provider`) is described here too, as the values the service, the
training side and an agent proposer share; :mod:`reef.inference.multimodal`
relays the calls to it.

A harness reaches image, embedding, speech and decision models through the same
Reef service, scenario and token as its chat calls. Reef forwards the request body
unchanged to the deployment's multimodal provider (see
:mod:`reef.inference.multimodal`), whatever serves the chat, and relays the response byte for byte; the record keeps a summary
instead of the body, so generated media and embedding vectors never enter record
storage or a proposer's prompt. The record names its route under
``metadata.reef_endpoint``, which is how trajectories and processors tell a
provider call from a chat exchange.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ProviderPreset:
    """A kind of multimodal gateway: its default address, its path for each Reef route it serves, and where its
    models are listed (``{base_url}`` and ``{modality}`` are filled in)."""

    name: str
    base_url: str | None
    paths: Mapping[str, str]
    models_url: str


PRESETS = {
    "openrouter": ProviderPreset(
        name="openrouter",
        base_url="https://openrouter.ai/api",
        paths={
            "/v1/images": "/v1/images",
            "/v1/embeddings": "/v1/embeddings",
            "/v1/audio/speech": "/v1/audio/speech",
            # OpenRouter's structured decision API lives outside /v1.
            "/v1/decisions": "/alpha/decisions",
        },
        models_url="{base_url}/v1/models?output_modalities={modality}",
    ),
    # OpenAI's routes, as the OpenAI-compatible gateways (OrcaRouter, LiteLLM, ...) serve them; no decisions.
    "openai-compatible": ProviderPreset(
        name="openai-compatible",
        base_url=None,
        paths={
            "/v1/images": "/v1/images/generations",
            "/v1/embeddings": "/v1/embeddings",
            "/v1/audio/speech": "/v1/audio/speech",
        },
        models_url="{base_url}/v1/models",
    ),
}
DEFAULT_PRESET = "openrouter"


@dataclass(frozen=True)
class MultimodalProvider:
    """One deployment's multimodal gateway: a preset, its address (no ``/v1`` suffix) and its key."""

    preset: ProviderPreset
    base_url: str
    api_key: str

    def upstream_path(self, route: str) -> str | None:
        """The provider's path for one of Reef's provider routes, or ``None`` when it serves none."""
        return self.preset.paths.get(route)

    def models_url(self, modality: str) -> str:
        """Where the provider lists its models of one output modality (``speech``, ``image``, ``embeddings``)."""
        return self.preset.models_url.format(base_url=self.base_url, modality=modality)

    def environment(self) -> dict[str, str]:
        """The provider as the ``REEF_PROVIDER_CALLS_*`` variables the service hands its recipes."""
        return {
            "REEF_PROVIDER_CALLS_PRESET": self.preset.name,
            "REEF_PROVIDER_CALLS_URL": self.base_url,
            "REEF_PROVIDER_CALLS_API_KEY": self.api_key,
        }


def resolve_provider(
    preset_name: str | None,
    base_url: str | None,
    api_key: str | None,
    upstream_url: str | None = None,
    upstream_api_key: str | None = None,
) -> MultimodalProvider | None:
    """The configured provider, or ``None`` when it has no key.

    The key is the configured one, else the chat upstream's when the upstream
    is the same gateway. An unknown preset, or one with no default address and
    none configured, is a configuration error.
    """
    name = (preset_name or DEFAULT_PRESET).strip()
    preset = PRESETS.get(name)
    if preset is None:
        raise ValueError(f"unknown provider_calls preset {name!r}; known: {', '.join(sorted(PRESETS))}")
    address = (base_url or preset.base_url or "").strip().rstrip("/")
    if not address:
        raise ValueError(f"the {name} provider preset needs provider_calls_url: the gateway's address, no /v1")
    key = api_key or None
    if key is None and upstream_url and upstream_url.strip().rstrip("/") == address:
        key = upstream_api_key or None
    return None if key is None else MultimodalProvider(preset=preset, base_url=address, api_key=key)


def provider_from_environment(environ: Mapping[str, str]) -> MultimodalProvider | None:
    """The provider the service handed a recipe as ``REEF_PROVIDER_CALLS_*``, or ``None`` when it has none."""
    return resolve_provider(
        environ.get("REEF_PROVIDER_CALLS_PRESET"),
        environ.get("REEF_PROVIDER_CALLS_URL"),
        environ.get("REEF_PROVIDER_CALLS_API_KEY"),
    )


__all__ = [
    "DEFAULT_PRESET",
    "ENDPOINT_KEY",
    "PRESETS",
    "PROVIDER_ROUTE_PATHS",
    "MultimodalProvider",
    "ProviderPreset",
    "compact",
    "provider_call_endpoint",
    "provider_call_payload",
    "provider_call_response",
    "provider_from_environment",
    "resolve_provider",
]
