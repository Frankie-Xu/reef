"""HTTP request execution and inference proxy runtime construction.

This protocol adapter supports external providers and managed inference
endpoints without importing a native model framework. Request payloads and
streamed responses retain the provider's format.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from reef.artifact.artifact import Artifact
from reef.core.config import config_option
from reef.runtime.base import InferenceRuntime
from reef.runtime.inference import InferenceHandler, InferenceStream, UpstreamStatusError
from reef.runtime.registry import RuntimeFactory, RuntimeRegistry, config_secret, config_string, register_runtime_kind


class RequestHeadersFactory(Protocol):
    """Produce request headers for one artifact-bound upstream call."""

    def __call__(self, artifact: Artifact, path: str) -> Mapping[str, str]: ...


def content_identity_headers(artifact: Artifact) -> dict[str, str]:
    """Reef identity headers for one selected, optionally materialized artifact."""
    headers = {"x-reef-release-id": artifact.ref.release_id}
    if artifact.local_path is not None:
        headers["x-reef-artifact-path"] = str(artifact.local_path)
    return headers


def default_artifact_request_headers(artifact: Artifact, path: str = "") -> Mapping[str, str]:
    """The default :data:`RequestHeadersFactory`: identity headers for every path."""
    return content_identity_headers(artifact)


def provider_request_headers(api_key: str) -> RequestHeadersFactory:
    """Build a RequestHeadersFactory that adds provider-native auth.

    Reef artifact identity headers are always included. For Anthropic
    (/v1/messages) the api key is sent as x-api-key + anthropic-version;
    for OpenAI-compatible routes it is sent as a Bearer token.
    """
    if not api_key:
        raise ValueError("api_key must be non-empty")

    def headers_for(artifact: Artifact, path: str) -> Mapping[str, str]:
        headers = content_identity_headers(artifact)
        if path in ("/v1/messages", "/v1/messages/count_tokens"):
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    return headers_for


class HttpInferenceHandler(InferenceHandler):
    """POST native inference requests to an HTTP provider.

    The request headers are produced by a callable, so callers can inject
    any combination of reef artifact identity headers, provider auth,
    or custom headers without subclassing.
    """

    def __init__(
        self,
        upstream_url: str,
        *,
        request_headers: RequestHeadersFactory = default_artifact_request_headers,
        timeout_s: float = 300.0,
        error_label: str = "inference upstream",
    ) -> None:
        upstream = upstream_url.rstrip("/")
        if not upstream:
            raise ValueError("upstream_url must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self._upstream_url = upstream
        self._request_headers = request_headers
        self._timeout_s = timeout_s
        self._error_label = error_label

    def reconnect(self, upstream_url: str) -> None:
        if not isinstance(upstream_url, str) or not upstream_url.startswith(("http://", "https://")):
            raise ValueError("inference endpoint must be an HTTP URL")
        self._upstream_url = upstream_url.rstrip("/")

    async def inference(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        from aiohttp import ClientSession, ClientTimeout

        async with (
            ClientSession(timeout=ClientTimeout(total=self._timeout_s)) as session,
            session.post(
                f"{self._upstream_url}{path}",
                json=payload,
                headers=dict(self._request_headers(artifact, path)),
            ) as response,
        ):
            body = await response.text()
            if response.status >= 400:
                raise UpstreamStatusError(
                    f"{self._error_label} returned {response.status}: {body[:400]}",
                    status=response.status,
                )
            value = await response.json()
        if not isinstance(value, dict):
            raise TypeError(f"{self._error_label} response must be a JSON object")
        return value

    async def inference_stream(
        self,
        artifact: Artifact,
        path: str,
        payload: dict[str, Any],
    ) -> InferenceStream:
        from aiohttp import ClientSession, ClientTimeout

        session = ClientSession(
            timeout=ClientTimeout(total=self._timeout_s),
            auto_decompress=False,
        )
        try:
            response = await session.post(
                f"{self._upstream_url}{path}",
                json=payload,
                headers=dict(self._request_headers(artifact, path)),
            )
        except Exception:
            await session.close()
            raise

        if response.status >= 400:
            try:
                body = (await response.read()).decode(errors="replace")
            finally:
                response.close()
                await session.close()
            raise UpstreamStatusError(
                f"{self._error_label} returned {response.status}: {body[:400]}",
                status=response.status,
            )

        excluded_headers = {
            "connection",
            "content-length",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
        }
        response_headers = {
            name: value for name, value in response.headers.items() if name.lower() not in excluded_headers
        }

        async def close() -> None:
            response.close()
            await session.close()

        return InferenceStream(
            status=response.status,
            headers=response_headers,
            chunks=response.content.iter_any(),
            close=close,
        )


def build_http_inference_handler(
    upstream_url: str,
    *,
    model_path: str,
    timeout_s: float,
    **config: Any,
) -> InferenceHandler:
    """Default factory; ``model_path`` is reserved for tokenizer-aware handlers."""

    if config:
        raise ValueError(f"default HTTP inference handler does not accept config keys: {sorted(config)}")
    return HttpInferenceHandler(upstream_url, timeout_s=timeout_s)


#: Provider API dialects the proxy can describe to the training side.
PROVIDER_APIS = ("openai", "responses", "anthropic")


class InferenceProxyRuntime(InferenceRuntime):
    """Inference runtime that wraps an HTTP inference handler.

    Returned by the ``inference_proxy`` runtime type for no-update recipes.
    Holds an HttpInferenceHandler with provider-native auth
    and exposes it via inference_handler. Does not implement training
    lifecycle methods.
    """

    def __init__(
        self,
        *,
        model_path: str = "",
        base_url: str,
        api_key: str | None = None,
        api: str = "openai",
        inference_timeout_s: float = 300.0,
    ) -> None:
        super().__init__(
            base_url=base_url,
            inference_timeout_s=inference_timeout_s,
        )
        if api not in PROVIDER_APIS:
            raise ValueError(f"inference proxy api must be one of {PROVIDER_APIS}, got {api!r}")
        self._model_path = model_path
        self._api_key = api_key
        self._api = api
        request_headers: RequestHeadersFactory = default_artifact_request_headers
        if api_key:
            request_headers = provider_request_headers(api_key)
        self._inference_handler = HttpInferenceHandler(
            self.base_url,
            request_headers=request_headers,
            timeout_s=self.inference_timeout_s,
            error_label="inference provider",
        )

    @classmethod
    def from_model_config(cls, value: object) -> InferenceProxyRuntime | None:
        """Validate a per-scenario model override and build its proxy runtime."""
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) - {"url", "model", "api", "api_key"}:
            raise ValueError("model must be null or an object with url, model, api and api_key")
        for name in ("url", "model"):
            item = value.get(name)
            if not isinstance(item, str) or not item.strip() or any(ord(c) < 32 for c in item):
                raise ValueError(f"model.{name} must be a non-empty string without control characters")
        parsed = urlparse(value["url"])
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("model.url must be an HTTP(S) URL without credentials, query or fragment")
        api = value.get("api", "openai")
        if api not in PROVIDER_APIS:
            raise ValueError("model.api must be openai, responses or anthropic")
        key = value.get("api_key")
        if key is not None and (not isinstance(key, str) or any(ord(c) < 32 for c in key)):
            raise ValueError("model.api_key must be a string without control characters")
        return cls(base_url=value["url"].strip(), model_path=value["model"].strip(), api=api, api_key=key)

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def api_key(self) -> str | None:
        return self._api_key

    @property
    def api(self) -> str:
        """The provider's API dialect (``openai``, ``responses``, or
        ``anthropic``). The proxy forwards whatever path a client calls; this
        tells the training side which dialect to speak when it calls the model
        itself."""
        return self._api

    @property
    def inference_handler(self) -> InferenceHandler:
        return self._inference_handler


@dataclass(frozen=True)
class InferenceProxyConfig:
    """Connection configuration owned by the inference proxy adapter."""

    base_url: str = config_option("", help="Inference provider base URL.")
    api: str = config_option("openai", help="Provider API format.")
    api_key: str | None = config_option(None, help="Provider credential.")
    api_key_env: str | None = config_option(None, help="Environment variable containing the provider credential.")
    timeout_s: float = config_option(300.0, help="Inference request timeout in seconds.")

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("runtime.base_url must be a non-empty string")
        if self.timeout_s <= 0:
            raise ValueError("runtime.timeout_s must be positive")


@register_runtime_kind
class InferenceProxyRuntimeFactory(RuntimeFactory):
    """Build an :class:`InferenceProxyRuntime` from a runtime config section."""

    kind = "inference_proxy"

    def config_type(self) -> type:
        return InferenceProxyConfig

    def __call__(
        self,
        config: Mapping[str, Any],
        model_path: str,
        recipe_config: Mapping[str, Any],
        environ: Mapping[str, str],
    ) -> InferenceRuntime:
        api_key = config_secret(config, environ, "api_key", "api_key_env")
        return InferenceProxyRuntime(
            model_path=model_path,
            base_url=config_string(config, "base_url"),
            api_key=api_key,
            api=config["api"],
            inference_timeout_s=config["timeout_s"],
        )


def resolve_proxy_runtime(
    values: Mapping[str, str],
    runtime: InferenceRuntime | None,
) -> InferenceRuntime | None:
    """Prefer an injected runtime, else build a proxy from environment values.

    Construction goes through the runtime registry's ``inference_proxy`` kind,
    the same factory the YAML config path uses.
    """
    if runtime is not None:
        return runtime
    base_url = _env_value(values, "REEF_UPSTREAM_URL")
    if base_url is None:
        return None
    config: dict[str, object] = {"type": "inference_proxy", "base_url": base_url}
    # REEF_API_KEY is the pre-rename spelling; drop it once no deployment sets it.
    api_key = _env_value(values, "REEF_UPSTREAM_API_KEY") or _env_value(values, "REEF_API_KEY")
    if api_key is not None:
        config["api_key"] = api_key
    timeout_raw = _env_value(values, "REEF_INFERENCE_TIMEOUT_S")
    if timeout_raw is not None:
        config["timeout_s"] = float(timeout_raw)
    built = RuntimeRegistry().build(
        config,
        model_path=_env_value(values, "REEF_MODEL_PATH") or "",
        environ=values,
    )

    if not isinstance(built, InferenceRuntime):
        raise TypeError("inference proxy factory must return an InferenceRuntime")
    return built


def _env_value(values: Mapping[str, str], name: str) -> str | None:
    return values.get(name, "").strip() or None


__all__ = [
    "PROVIDER_APIS",
    "HttpInferenceHandler",
    "InferenceProxyConfig",
    "InferenceProxyRuntime",
    "InferenceProxyRuntimeFactory",
    "RequestHeadersFactory",
    "build_http_inference_handler",
    "content_identity_headers",
    "default_artifact_request_headers",
    "provider_request_headers",
    "resolve_proxy_runtime",
]
