"""The multimodal provider: where Reef sends provider calls (images, embeddings, speech, decisions).

A provider call goes to one gateway that serves every modality behind a single
key, whatever serves the scenario's chat: a deployment can chat through a local
Ollama and still generate speech. Which gateway is a deployment setting
(``inference.provider_calls_*``); a preset names its paths, and OpenRouter is the
default. The bodies pass through unchanged, so a client speaks the provider's
own format; the preset only decides where each of Reef's routes lands and how an
agent can list the provider's models. A deployment whose chat upstream is the
same gateway reuses that upstream's key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reef.artifact.artifact import Artifact
from reef.inference.http import HttpInferenceHandler, RequestHeadersFactory


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


class ProviderRequestHeaders(RequestHeadersFactory):
    """The provider's key, and an unencoded body so Reef can record a JSON response; no Reef identity headers."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("a provider api_key must be non-empty")
        self._api_key = api_key

    def headers(self, artifact: Artifact, path: str) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Accept-Encoding": "identity"}


class ProviderCallHandler(HttpInferenceHandler):
    """Relay provider calls to the deployment's multimodal provider at its own path for each route."""

    def __init__(self, provider: MultimodalProvider, *, timeout_s: float = 300.0) -> None:
        super().__init__(
            provider.base_url,
            request_headers=ProviderRequestHeaders(provider.api_key),
            timeout_s=timeout_s,
            error_label=f"the {provider.preset.name} provider",
        )
        self.provider = provider

    def _post_arguments(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        upstream_path = self.provider.upstream_path(path)
        if upstream_path is None:
            raise ValueError(f"the {self.provider.preset.name} provider serves no {path}")
        return {**super()._post_arguments(artifact, path, payload), "url": f"{self._upstream_url}{upstream_path}"}


__all__ = [
    "DEFAULT_PRESET",
    "PRESETS",
    "MultimodalProvider",
    "ProviderCallHandler",
    "ProviderPreset",
    "ProviderRequestHeaders",
    "provider_from_environment",
    "resolve_provider",
]
