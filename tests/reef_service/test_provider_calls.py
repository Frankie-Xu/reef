"""Provider calls: images, embeddings, speech and decisions relayed through Reef and recorded as summaries."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import textwrap
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from reef_service.runtime_stubs import StubTrainingRuntime, runtime_bindings

from reef.artifact import InMemoryRepositoryBackend
from reef.core import AgentRecord, RequestType
from reef.core.provider_calls import compact, provider_call_endpoint
from reef.core.trajectories import make_trajectory, provider_calls, recorded_payload
from reef.dispatcher import Dispatcher, build_default_dispatcher
from reef.harness.client.wrapper import CAPTURE_PATHS, run_agent
from reef.inference.http import HttpInferenceHandler, provider_request_headers
from reef.recipe.reefine.evolution import failures_text
from reef.service.app import create_app
from reef.storage.sqlite import SQLiteScenarioStorage
from reef.train.cordis_backend.processor import RecordDrivenTraceProcessor
from reef.train.types import ProcessorContext

from ._policy_recipe import TestPolicyRecipe

IMAGE_BASE64 = "iVBORw0KGgo" * 1000
AUDIO_BYTES = bytes(range(256)) * 64


def provider_upstream(received: list[dict]) -> web.Application:
    """A provider serving the four routes at their provider paths, remembering each request."""

    async def handle(request: web.Request) -> web.StreamResponse:
        received.append(
            {
                "path": request.path,
                "payload": await request.json(),
                "authorization": request.headers.get("Authorization"),
                "accept_encoding": request.headers.get("Accept-Encoding"),
            }
        )
        if request.path == "/v1/images":
            return web.json_response({"data": [{"b64_json": IMAGE_BASE64}], "usage": {"cost": 0.04}})
        if request.path == "/v1/embeddings":
            return web.json_response({"data": [{"embedding": [0.5] * 1536, "index": 0}]})
        if request.path == "/v1/audio/speech":
            return web.Response(body=AUDIO_BYTES, content_type="audio/mpeg")
        return web.json_response({"answers": {"route": {"choice": "billing"}}, "confidence": 0.93})

    app = web.Application()
    for path in ("/v1/images", "/v1/embeddings", "/v1/audio/speech", "/alpha/decisions"):
        app.router.add_post(path, handle)
    return app


async def reef_client(upstream: TestServer) -> tuple[TestClient, object]:
    handler = HttpInferenceHandler(
        str(upstream.make_url("")).rstrip("/"), request_headers=provider_request_headers("provider-key")
    )
    dispatcher = build_default_dispatcher(scenario_storage=SQLiteScenarioStorage())
    client = TestClient(TestServer(create_app(dispatcher, inference_handler=handler)))
    await client.start_server()
    return client, dispatcher


@pytest.mark.unit
def test_provider_calls_reach_their_provider_paths_and_are_recorded_as_summaries() -> None:
    async def run() -> None:
        received: list[dict] = []
        upstream = TestServer(provider_upstream(received))
        await upstream.start_server()
        client, dispatcher = await reef_client(upstream)
        headers = {"x-reef-scenario": "media", "x-reef-tag-session": "s-1"}
        try:
            image = await client.post(
                "/v1/images", headers=headers, json={"model": "google/image", "prompt": "a reef at dawn"}
            )
            assert image.status == 200
            assert (await image.json())["data"][0]["b64_json"] == IMAGE_BASE64
            image_record = image.headers["x-reef-agent-record-id"]

            speech = await client.post(
                "/v1/audio/speech", headers=headers, json={"model": "openai/tts", "input": "hi", "voice": "alloy"}
            )
            assert speech.status == 200
            assert speech.headers["Content-Type"] == "audio/mpeg"
            assert await speech.read() == AUDIO_BYTES

            embedding = await client.post(
                "/v1/embeddings", headers=headers, json={"model": "openai/embed", "input": "reef"}
            )
            assert len((await embedding.json())["data"][0]["embedding"]) == 1536

            decision = await client.post(
                "/v1/decisions",
                headers=headers,
                json={"model": "~typesafe/jev-latest", "state": "refund?", "questions": {"route": {}}},
            )
            assert (await decision.json())["answers"]["route"]["choice"] == "billing"

            assert [request["path"] for request in received] == [
                "/v1/images",
                "/v1/audio/speech",
                "/v1/embeddings",
                "/alpha/decisions",
            ]
            assert received[0]["payload"] == {"model": "google/image", "prompt": "a reef at dawn"}
            assert {request["authorization"] for request in received} == {"Bearer provider-key"}
            assert {request["accept_encoding"] for request in received} == {"identity"}

            records = {
                record.agent_record_id: record.payload
                for record in dispatcher.get_or_create_scenario("media").records.replay("media")
            }
            image_payload = records[image_record]
            assert image_payload["prompt"] == "a reef at dawn"
            assert image_payload["metadata"] == {"tags": {"session": "s-1"}, "reef_endpoint": "/v1/images"}
            assert image_payload["response"]["status"] == 200
            assert image_payload["response"]["body"]["usage"] == {"cost": 0.04}
            assert image_payload["response"]["body"]["data"][0]["b64_json"] == {
                "omitted": "text",
                "chars": len(IMAGE_BASE64),
                "sha256": hashlib.sha256(IMAGE_BASE64.encode()).hexdigest(),
            }
            by_endpoint = {payload["metadata"]["reef_endpoint"]: payload for payload in records.values()}
            assert by_endpoint["/v1/audio/speech"]["response"] == {
                "status": 200,
                "content_type": "audio/mpeg",
                "complete": True,
                "bytes": len(AUDIO_BYTES),
                "sha256": hashlib.sha256(AUDIO_BYTES).hexdigest(),
            }
            assert by_endpoint["/v1/embeddings"]["response"]["body"]["data"][0]["embedding"] == {
                "omitted": "vector",
                "length": 1536,
            }
            assert by_endpoint["/v1/decisions"]["response"]["body"]["confidence"] == 0.93
        finally:
            await client.close()
            await upstream.close()

    asyncio.run(run())


@pytest.mark.unit
def test_a_provider_call_does_not_stream() -> None:
    async def run() -> None:
        received: list[dict] = []
        upstream = TestServer(provider_upstream(received))
        await upstream.start_server()
        client, _ = await reef_client(upstream)
        try:
            response = await client.post(
                "/v1/images",
                headers={"x-reef-scenario": "media"},
                json={"model": "google/image", "prompt": "reef", "stream": True},
            )
            assert response.status == 400
            assert "does not stream" in await response.text()
            assert received == []
        finally:
            await client.close()
            await upstream.close()

    asyncio.run(run())


@pytest.mark.unit
def test_a_training_runtime_does_not_serve_provider_calls(tmp_path) -> None:
    async def run() -> None:
        received: list[dict] = []
        upstream = TestServer(provider_upstream(received))
        await upstream.start_server()
        initial = tmp_path / "initial"
        initial.mkdir()
        dispatcher = Dispatcher(
            TestPolicyRecipe(**runtime_bindings(StubTrainingRuntime(base_url="http://trainer")), batch_size=1),
            InMemoryRepositoryBackend.factory(initial, root=tmp_path / "repository"),
            local_artifact_dir=tmp_path / "staged",
            scenario_storage=SQLiteScenarioStorage(),
        )
        handler = HttpInferenceHandler(str(upstream.make_url("")).rstrip("/"))
        client = TestClient(TestServer(create_app(dispatcher, inference_handler=handler)))
        await client.start_server()
        try:
            response = await client.post(
                "/v1/images", headers={"x-reef-scenario": "math"}, json={"model": "google/image", "prompt": "reef"}
            )
            assert response.status == 400
            assert "provider-backed scenario" in await response.text()
            assert received == []
        finally:
            await client.close()
            await upstream.close()

    asyncio.run(run())


@pytest.mark.unit
def test_compact_keeps_short_values_and_describes_long_text_and_vectors() -> None:
    value = {"prompt": "short", "flags": [True] * 100, "tokens": list(range(10)), "image": "x" * 3000}
    compacted = compact(value)
    assert compacted["prompt"] == "short"
    assert compacted["flags"] == [True] * 100
    assert compacted["tokens"] == list(range(10))
    assert compacted["image"]["chars"] == 3000
    assert compact({"embedding": [0.1] * 65})["embedding"] == {"omitted": "vector", "length": 65}


def inference(agent_record_id: str, payload: dict) -> AgentRecord:
    return AgentRecord.create(
        scenario="s", request_type=RequestType.INFERENCE, payload=payload, agent_record_id=agent_record_id
    )


FIRST_TURN = {
    "model": "chat-model",
    "messages": [{"role": "user", "content": "draw a reef"}],
    "response": {"choices": [{"message": {"role": "assistant", "content": "drawing"}}]},
}
IMAGE_CALL = {
    "model": "google/image",
    "prompt": "a reef",
    "metadata": {"reef_endpoint": "/v1/images"},
    "response": {"status": 200, "content_type": "application/json", "complete": True, "body": {"data": []}},
}
SECOND_TURN = {
    "model": "chat-model",
    "messages": [
        {"role": "user", "content": "draw a reef"},
        {"role": "assistant", "content": "drawing"},
        {"role": "user", "content": "done?"},
    ],
    "response": {"choices": [{"message": {"role": "assistant", "content": "done"}}]},
}


@pytest.mark.unit
def test_a_provider_call_is_one_agent_step_between_chat_turns() -> None:
    sample = make_trajectory(
        [inference("chat-1", FIRST_TURN), inference("image-1", IMAGE_CALL), inference("chat-2", SECOND_TURN)]
    )
    steps = sample.trajectory["steps"]
    assert [(step["source"], step["message"]) for step in steps] == [
        ("user", "draw a reef"),
        ("agent", "drawing"),
        ("agent", ""),
        ("user", "done?"),
        ("agent", "done"),
    ]
    (call,) = steps[2]["tool_calls"]
    assert call == {
        "tool_call_id": "image-1",
        "function_name": "/v1/images",
        "arguments": {"model": "google/image", "prompt": "a reef"},
    }
    assert json.loads(steps[2]["observation"]["results"][0]["content"])["body"] == {"data": []}
    assert sample.trajectory["agent"]["model_name"] == "chat-model"
    assert recorded_payload(sample) == SECOND_TURN
    assert provider_calls(sample) == (IMAGE_CALL,)
    assert provider_call_endpoint(IMAGE_CALL) == "/v1/images"
    assert provider_call_endpoint({"metadata": {"reef_endpoint": "/v1/unknown"}}) is None


@pytest.mark.unit
def test_a_sample_of_provider_calls_alone_reads_its_last_call() -> None:
    sample = make_trajectory([inference("image-1", IMAGE_CALL)])
    assert recorded_payload(sample) == IMAGE_CALL


@pytest.mark.unit
def test_the_record_driven_processor_does_not_batch_provider_calls() -> None:
    processor = RecordDrivenTraceProcessor(ProcessorContext("s", {"batch_size": 1}))
    processor.ingest(inference("image-1", IMAGE_CALL))
    assert not processor.ready()
    assert processor.retention_decision().releasable_agent_record_ids == frozenset({"image-1"})
    processor.ingest(inference("chat-1", FIRST_TURN))
    assert processor.ready()


@pytest.mark.unit
def test_the_proposer_reads_the_chat_turn_with_the_provider_calls_beside_it() -> None:
    sample = make_trajectory([inference("chat-1", FIRST_TURN), inference("image-1", IMAGE_CALL)], 0.0, "no image")
    (view,) = json.loads(failures_text([sample]))
    assert view == {
        "request": FIRST_TURN,
        "provider_calls": [IMAGE_CALL],
        "score": 0.0,
        "feedback": "no image",
    }
    (chat_only,) = json.loads(failures_text([make_trajectory([inference("chat-1", FIRST_TURN)], 0.0)]))
    assert "provider_calls" not in chat_only


@pytest.mark.unit
def test_wrapper_exports_the_proxy_so_an_extensions_provider_calls_are_captured(tmp_path) -> None:
    import http.server
    import threading

    assert "/v1/images" in CAPTURE_PATHS

    class FakeReef(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("x-reef-agent-record-id", "image-receipt")
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), FakeReef)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    compose = tmp_path / "compose"
    compose.mkdir()
    reef_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    provider = {"api": "openai-completions", "apiKey": "dummy", "baseUrl": reef_url, "models": [{"id": "chat"}]}
    (compose / "models.json").write_text(json.dumps({"providers": {"reef": provider}}))
    # An extension's own call: straight to the address the wrapper exports, with no proxy of the environment.
    binary = tmp_path / "pi"
    binary.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, os, urllib.request
            request = urllib.request.Request(
                os.environ["REEF_INFERENCE_URL"] + "/v1/images",
                data=json.dumps({"model": "google/image", "prompt": "reef"}).encode(),
                headers={"content-type": "application/json"},
            )
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request).read()
            """
        )
    )
    binary.chmod(0o755)
    with (
        patch.dict(os.environ, {"REEF_HARNESS_CAPTURES_DIR": str(tmp_path)}),
        contextlib.suppress(SystemExit),
    ):
        run_agent(str(binary), str(compose), "media", "pi", "PI_CODING_AGENT_DIR", [])
    server.shutdown()
    (captures,) = tmp_path.glob("*.pending.json")
    turns = json.loads(captures.read_text())["turns"]
    assert [(turn["path"], turn["receipt"]) for turn in turns] == [("/v1/images", "image-receipt")]
