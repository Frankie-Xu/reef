"""The loopback gateway an agent proposer and its trial runs reach models through.

The agent and every candidate harness it tries hold one URL and nothing else: no
provider key and no Reef token. The URL's path carries a random token, so the
candidate's own extensions reach the gateway as they would reach Reef, with no
credential of their own. Behind it:

- the model routes go to the served model with the served binding's key, one
  call from the step's budget each, recorded in the step's ``proposer.json``;
  the request's ``model`` is always the served one;
- the provider-call routes (images, embeddings, speech, decisions) go to
  OpenRouter, recorded as the summaries Reef records;
- ``/check`` and ``/trial`` run the agent's workspace through admission and
  through a real run of the candidate harness (:class:`WorkspaceTools`);
- everything else, Reef's own routes included, is 404.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from reef.core.provider_calls import PROVIDER_ROUTE_PATHS, provider_call_payload, provider_call_response
from reef.harness.episodes.model_binding import ModelBinding, usage_of
from reef.inference.openrouter import OPENROUTER_BASE_URL, OPENROUTER_PATHS
from reef.train.cordis_backend.strategies import ProposerCalls

logger = logging.getLogger(__name__)

MODEL_PATHS = ("/v1/chat/completions", "/v1/responses", "/v1/messages")
#: The most of one reply the gateway keeps to record; the client always gets all of it.
MAX_RECORDED_BYTES = 2 * 1024 * 1024
#: How much of a provider's refusal a trial reports back to the agent.
MAX_ERROR_CHARS = 600
ANTHROPIC_VERSION = "2023-06-01"


class WorkspaceTools(ABC):
    """What ``/check`` and ``/trial`` do with the agent's workspace."""

    @abstractmethod
    def check(self) -> dict[str, Any]:
        """The workspace through admission, as the step would admit it."""

    @abstractmethod
    def trial(self, task: str) -> dict[str, Any]:
        """One real run of the candidate harness on ``task``."""


class AgentGateway:
    """A loopback HTTP server for one agent run; ``base_url`` is the only thing the agent is given."""

    def __init__(
        self,
        served: ModelBinding,
        calls: ProposerCalls,
        tools: WorkspaceTools,
        openrouter_key: str | None,
        *,
        host: str = "127.0.0.1",
    ) -> None:
        self._served = served
        self._calls = calls
        self._tools = tools
        self._openrouter_key = openrouter_key
        self._host = host
        self._token = secrets.token_urlsafe(24)
        self._lock = threading.Lock()
        self._provider_calls: list[dict[str, Any]] = []
        self._server: ThreadingHTTPServer | None = None

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("the agent gateway is not running")
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        """The address the agent and its trials use in place of Reef's: no ``/v1`` suffix."""
        return f"http://{self._host}:{self.port}/t/{self._token}"

    def provider_calls_since(self, index: int) -> list[dict[str, Any]]:
        """The provider calls made after the first ``index``: what one trial's harness asked of OpenRouter."""
        with self._lock:
            return list(self._provider_calls[index:])

    def provider_call_count(self) -> int:
        with self._lock:
            return len(self._provider_calls)

    def start(self) -> None:
        server = ThreadingHTTPServer((self._host, 0), self.handler_class())
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, name="reef-agent-gateway", daemon=True).start()
        self._server = server

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def handler_class(self) -> type[BaseHTTPRequestHandler]:
        gateway = self
        prefix = f"/t/{self._token}"

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                if not secrets.compare_digest(path[: len(prefix)], prefix):
                    self._answer(404, {"error": "not found"})
                    return
                route = path[len(prefix) :]
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    self._answer(400, {"error": "the body must be JSON"})
                    return
                if not isinstance(payload, dict):
                    self._answer(400, {"error": "the body must be a JSON object"})
                    return
                if route in MODEL_PATHS:
                    gateway.relay_model_call(self, route, payload)
                elif route in PROVIDER_ROUTE_PATHS:
                    gateway.relay_provider_call(self, route, payload)
                elif route == "/check":
                    self._answer(200, gateway._tools.check())
                elif route == "/trial":
                    task = payload.get("task")
                    if not isinstance(task, str) or not task.strip():
                        self._answer(400, {"error": "a trial needs a task"})
                        return
                    self._answer(200, gateway._tools.trial(task.strip()))
                else:
                    self._answer(404, {"error": f"the gateway serves no {route}"})

            def do_GET(self) -> None:
                self._answer(404, {"error": "not found"})

            def _answer(self, status: int, body: Mapping[str, Any]) -> None:
                data = json.dumps(body, default=str).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("agent gateway: " + format, *args)

        return Handler

    def relay_model_call(self, client: BaseHTTPRequestHandler, route: str, payload: dict[str, Any]) -> None:
        try:
            self._calls.spend()
        except RuntimeError as exc:
            refuse(client, 429, str(exc))
            return
        # The served model is the only one the agent reaches: a request naming another is sent to it anyway.
        payload = {**payload, "model": self._served.model}
        headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
        if route == "/v1/messages":
            if self._served.api_key:
                headers["x-api-key"] = self._served.api_key
            headers["anthropic-version"] = client.headers.get("anthropic-version") or ANTHROPIC_VERSION
            beta = client.headers.get("anthropic-beta")
            if beta:
                headers["anthropic-beta"] = beta
        elif self._served.api_key:
            headers["Authorization"] = f"Bearer {self._served.api_key}"
        url = f"{self._served.base_url.rstrip('/')}{route}"
        started = time.monotonic()
        status, body, _ = relay(client, url, headers, payload, timeout_s=self._served.timeout_s)
        text = body.decode("utf-8", errors="replace")
        entry: dict[str, Any] = {
            "source": "agent",
            "model": self._served.model,
            "path": route,
            "status": status,
            "messages": len(payload.get("messages") or payload.get("input") or []),
            "last_message": (payload.get("messages") or [None])[-1],
            "reply": reply_text(text),
            "seconds": round(time.monotonic() - started, 3),
        }
        usage = sse_usage(text)
        if usage is not None:
            entry["usage"] = usage
        self._calls.record(entry)

    def relay_provider_call(self, client: BaseHTTPRequestHandler, route: str, payload: dict[str, Any]) -> None:
        if self._openrouter_key is None:
            refuse(client, 501, f"{route} needs an OpenRouter key on the Reef deployment")
            return
        try:
            self._calls.spend()
        except RuntimeError as exc:
            refuse(client, 429, str(exc))
            return
        headers = {
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
            "Authorization": f"Bearer {self._openrouter_key}",
        }
        url = f"{OPENROUTER_BASE_URL}{OPENROUTER_PATHS[route]}"
        started = time.monotonic()
        status, body, response_headers = relay(client, url, headers, payload, timeout_s=300.0)
        summary = provider_call_response(status, response_headers, body, complete=True)
        call: dict[str, Any] = {"path": route, "model": payload.get("model"), "status": status}
        if status >= 400:
            call["error"] = body.decode("utf-8", errors="replace")[:MAX_ERROR_CHARS]
        with self._lock:
            self._provider_calls.append(call)
        self._calls.record(
            {
                "source": "agent",
                "provider_call": provider_call_payload(route, {**payload, "response": summary}),
                "seconds": round(time.monotonic() - started, 3),
            }
        )


def refuse(client: BaseHTTPRequestHandler, status: int, message: str) -> None:
    data = json.dumps({"error": {"message": message}}).encode()
    client.send_response(status)
    client.send_header("Content-Type", "application/json")
    client.send_header("Content-Length", str(len(data)))
    client.end_headers()
    client.wfile.write(data)


def relay(
    client: BaseHTTPRequestHandler,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    *,
    timeout_s: float,
) -> tuple[int, bytes, dict[str, str]]:
    """Forward ``payload`` and stream the answer to ``client`` as it arrives: the status, what was kept of the
    body for the record, and the upstream headers."""
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=dict(headers), method="POST")
    try:
        upstream = urllib.request.urlopen(request, timeout=timeout_s)
    except urllib.error.HTTPError as error:
        # A refusal is short and whole: relay it as the provider wrote it.
        body = error.read()
        upstream_headers = dict(error.headers.items())
        client.send_response(error.code)
        client.send_header("Content-Type", error.headers.get("Content-Type") or "application/json")
        client.send_header("Content-Length", str(len(body)))
        client.end_headers()
        client.wfile.write(body)
        return error.code, body[:MAX_RECORDED_BYTES], upstream_headers
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        refuse(client, 502, f"the upstream did not answer: {error}")
        return 502, b"", {}
    kept = bytearray()
    with upstream:
        upstream_headers = dict(upstream.headers.items())
        client.send_response(upstream.status)
        for name in ("Content-Type", "Content-Disposition"):
            if upstream.headers.get(name):
                client.send_header(name, upstream.headers[name])
        client.end_headers()
        while True:
            # read1 returns what has arrived, so a streamed reply reaches the agent as it is written.
            chunk = upstream.read1(65536)
            if not chunk:
                break
            client.wfile.write(chunk)
            client.wfile.flush()
            if len(kept) < MAX_RECORDED_BYTES:
                kept.extend(chunk)
    return upstream.status, bytes(kept), upstream_headers


def reply_events(text: str) -> list[dict[str, Any]]:
    """The JSON body, or the JSON payload of every SSE ``data:`` line."""
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        return [value] if isinstance(value, dict) else []
    events = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            value = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def reply_text(text: str) -> str:
    """The text a model reply carries, whole or streamed, in the chat, Anthropic or Responses dialect."""
    parts: list[str] = []
    for event in reply_events(text):
        for choice in event.get("choices") or ():
            if isinstance(choice, dict):
                piece = (choice.get("delta") or choice.get("message") or {}).get("content")
                if isinstance(piece, str):
                    parts.append(piece)
        delta = event.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            parts.append(delta["text"])
        elif event.get("type") == "response.output_text.delta" and isinstance(delta, str):
            parts.append(delta)
        parts.extend(
            block["text"]
            for block in event.get("content") or ()
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    return "".join(parts)[:20_000]


def sse_usage(text: str) -> dict[str, int] | None:
    """Input and output tokens a reply reported, whole or streamed; ``None`` when it reported none."""
    input_tokens = output_tokens = 0
    seen = False
    for event in reply_events(text):
        for holder in (event, event.get("message") if isinstance(event.get("message"), dict) else None):
            if holder is None or not isinstance(holder.get("usage"), dict):
                continue
            usage = usage_of(holder)
            if usage is None:
                continue
            seen = True
            input_tokens = max(input_tokens, usage.get("input_tokens", 0))
            output_tokens = max(output_tokens, usage.get("output_tokens", 0))
    return {"input_tokens": input_tokens, "output_tokens": output_tokens} if seen else None


__all__ = ["MODEL_PATHS", "AgentGateway", "WorkspaceTools", "reply_text", "sse_usage"]
