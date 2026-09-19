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
from typing import Any

from reef.artifact.artifact import Artifact
from reef.core.provider_calls import MultimodalProvider
from reef.inference.http import HttpInferenceHandler, RequestHeadersFactory


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


__all__ = ["ProviderCallHandler", "ProviderRequestHeaders"]
