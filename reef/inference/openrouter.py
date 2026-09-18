"""OpenRouter: where Reef sends provider calls (images, embeddings, speech, decisions).

OpenRouter is the one provider that serves every one of these modalities behind a
single key, so provider calls go there whatever serves the scenario's chat: a
deployment can chat through a local Ollama and still generate speech. The key is
``inference.openrouter_api_key`` (``OPENROUTER_API_KEY``); a deployment whose chat
upstream is OpenRouter reuses that upstream's key. The calls carry only the
key: Reef's artifact identity headers stay inside the deployment.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from reef.artifact.artifact import Artifact
from reef.core.provider_calls import PROVIDER_ROUTE_PATHS
from reef.inference.http import HttpInferenceHandler, RequestHeadersFactory

OPENROUTER_BASE_URL = "https://openrouter.ai/api"
#: OpenRouter's path for each provider route; its structured decision API lives outside ``/v1``.
OPENROUTER_PATHS = {**{path: path for path in PROVIDER_ROUTE_PATHS}, "/v1/decisions": "/alpha/decisions"}


class OpenRouterRequestHeaders(RequestHeadersFactory):
    """The key, and an unencoded body so Reef can record a JSON response."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("an OpenRouter api_key must be non-empty")
        self._api_key = api_key

    def headers(self, artifact: Artifact, path: str) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Accept-Encoding": "identity"}


class OpenRouterHandler(HttpInferenceHandler):
    """Relay provider calls to OpenRouter at its own path for each route."""

    def __init__(self, api_key: str, *, base_url: str = OPENROUTER_BASE_URL, timeout_s: float = 300.0) -> None:
        super().__init__(
            base_url, request_headers=OpenRouterRequestHeaders(api_key), timeout_s=timeout_s, error_label="OpenRouter"
        )

    def _post_arguments(self, artifact: Artifact, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        upstream_path = OPENROUTER_PATHS.get(path)
        if upstream_path is None:
            raise ValueError(f"OpenRouter serves no provider call at {path}")
        return {**super()._post_arguments(artifact, path, payload), "url": f"{self._upstream_url}{upstream_path}"}


def openrouter_api_key(configured: str | None, upstream_url: str | None, upstream_api_key: str | None) -> str | None:
    """The key provider calls use: the configured one, else the chat upstream's when that upstream is OpenRouter."""
    if configured:
        return configured
    if upstream_url and urlparse(upstream_url).hostname == "openrouter.ai":
        return upstream_api_key or None
    return None


__all__ = [
    "OPENROUTER_BASE_URL",
    "OPENROUTER_PATHS",
    "OpenRouterHandler",
    "OpenRouterRequestHeaders",
    "openrouter_api_key",
]
