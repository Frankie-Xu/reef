"""The page per filed harness request: the step's state while it runs, the verdict once its row lands.

``GET /reef/harness/requests/{record_id}/page`` renders it from the request's
agent record, the catalog and the running step's progress, and a browser opens
it by a link that carries the scenario and the token as query parameters. The
live chain here runs in ``training_mode: manual`` with a proposer that holds
its step open until the test has read the page mid-step.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from threading import Event

from aiohttp.test_utils import TestClient, TestServer
from reef_service.test_harness_proposals import _dispatcher, _recipe

from reef.core import AgentRecord, RequestType
from reef.service.app import create_app
from reef.service.request_page import REFRESH_SECONDS, build_request_page, settled_step
from reef.train.cordis_backend import Mutation, StepProgress

MODULE = Path(__file__).parents[2] / "reef" / "service" / "request_page.py"
REFRESH = f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">'
RECORD_ID = "3f1c2a9d0b7e4c5d8e9f0a1b2c3d4e5f"
TEXT = "text me when the run is blocked"
SESSION = "3f1c2a9d0b7e"
MARKER = Mutation("create", "r1", {"name": "rules", "config": {"text": "marker rules"}})
SCENARIO = "agents"
QUERY = {"scenario": SCENARIO, "token": "secret"}


def _record(compacted_at: float | None = None, text: str = TEXT, requires: list | None = None) -> dict:
    """The agent record as ``Dispatcher.read_record`` answers it for a ``POST /reef/train`` instruction."""
    payload = {"text": text, "session": SESSION, "release_id": "rel-0", "requires": requires or []}
    return {
        "sequence": 1,
        "agent_record_id": RECORD_ID,
        "request_type": "train",
        "created_at": 1_000.0,
        "compacted_at": compacted_at,
        "references": [],
        "artifact_ref": None,
        "score": None,
        "payload": payload,
    }


def _row(metrics: dict, *, release_id: str = "rel-1", parent: str | None = "rel-0", **rest) -> dict:
    return {
        "release_id": release_id,
        "parent_release_id": parent,
        "operation": "training",
        "pending": False,
        "recorded_at": 1_050.0,
        "metrics": metrics,
        **rest,
    }


CREATION = _row({}, release_id="rel-0", parent=None, operation="creation")
MUTATION = {"op": "create", "id": "r1", "options": {"name": "rules", "config": {"text": "marker rules"}}}


def _answered(**extra) -> dict:
    """The metrics of the row that answered the request, the trainer's ``training_request`` stamp included."""
    request = {"id": RECORD_ID, "text": TEXT, "session": SESSION, "release_id": "rel-0", "requires": []}
    return {"steps": 1, "training_request": request, **extra}


def _sections(page: str) -> list[str]:
    return [line[4:-5] for line in page.splitlines() if line.startswith("<h2>")]


def _section(page: str, name: str) -> str:
    _, _, tail = page.partition(f"<h2>{name}</h2>")
    body, _, _ = tail.partition("<h2>")
    return body


def _sub(page: str) -> str:
    return next(line for line in page.splitlines() if line.startswith('<p class="sub">'))


def test_a_queued_request_reloads_and_says_no_step_has_taken_it() -> None:
    page = build_request_page(_record(), [CREATION], now=1_042.0)
    page.encode("ascii")
    assert page.startswith(REFRESH + "\n<title>Harness request 3f1c2a9d</title>")
    assert "<h1>Harness request 3f1c2a9d</h1>" in page
    assert f'request <span class="id">{RECORD_ID}</span> | <span class="queued">queued</span> | filed at 1000' in _sub(
        page
    )
    assert _sections(page) == ["Request", "Progress"]
    assert f'<p class="text">{TEXT}</p>' in _section(page, "Request")
    assert f'from session <span class="id">{SESSION}</span> on release <span class="id">rel-0</span>' in page
    progress = _section(page, "Progress")
    assert "no step has taken the request yet" in progress
    assert "<tr><th>filed</th><td>42 s ago</td></tr>" in progress
    assert f"reloads every {REFRESH_SECONDS} seconds until the step settles" in progress
    assert "step time" not in progress and "<h2>Verdict</h2>" not in page
    assert settled_step([CREATION], RECORD_ID) is None


def test_a_running_request_shows_the_steps_phase_its_elapsed_time_and_the_gates_size() -> None:
    gating = StepProgress(RECORD_ID, "gating", started_at=900.0, step_record="/work/steps/1", episodes_total=2)
    page = build_request_page(_record(), [CREATION], progress=gating, now=1_100.0)
    assert REFRESH in page and '<span class="gating">gating</span>' in _sub(page)
    progress = _section(page, "Progress")
    assert "the gate is running the candidate through its episodes" in progress
    assert "<tr><th>step time</th><td>3 min 20 s into the step</td></tr>" in progress
    assert "<tr><th>episodes</th><td>2 in the gate</td></tr>" in progress
    assert '<tr><th>step record</th><td class="id">/work/steps/1</td></tr>' in progress
    assert "filed</th>" not in progress

    proposing = StepProgress(RECORD_ID, "proposing", started_at=1_058.0, step_record=None)
    page = build_request_page(_record(), [CREATION], progress=proposing, now=1_100.0)
    progress = _section(page, "Progress")
    assert '<span class="proposing">proposing</span>' in progress and "the proposer is writing the change" in progress
    assert "<tr><th>step time</th><td>42 s into the step</td></tr>" in progress
    assert "episodes</th>" not in progress and "step record</th>" not in progress

    # Another request's step says nothing about this one, which still waits.
    other = replace(gating, request_id="another")
    page = build_request_page(_record(), [CREATION], progress=other, now=1_100.0)
    assert '<span class="queued">queued</span>' in page and "100 s ago" in page

    # The trainer holds the request and the backend reports no phase: between settlement and the commit.
    page = build_request_page(_record(), [CREATION], consumed=True, now=1_100.0)
    assert '<span class="running">running</span>' in page and "its row follows" in page and REFRESH in page

    # Compacted and no row yet: the commit that consumed the record is landing, so the page keeps reloading.
    page = build_request_page(_record(compacted_at=1_099.0), [CREATION], now=1_100.0)
    assert '<span class="settling">settling</span>' in page and "committing its row" in page and REFRESH in page


def test_a_settled_selected_request_carries_the_verdict_the_mutation_and_the_link_with_its_query() -> None:
    rows = [CREATION, _row(_answered(selected=True, published=True, mutation=MUTATION))]
    page = build_request_page(_record(compacted_at=1_050.0), rows, link_query=QUERY, now=1_100.0)
    page.encode("ascii")
    assert REFRESH not in page and page.startswith("<title>Harness request 3f1c2a9d</title>")
    assert '<span class="selected">selected</span> | filed at 1000' in _sub(page)
    assert _sections(page) == ["Request", "Verdict", "What changed"]
    verdict = _section(page, "Verdict")
    assert '<tr><th>verdict</th><td><span class="selected">selected</span></td></tr>' in verdict
    assert "published as release rel-1; restart reef-pi to install it (the update notice offers it)" in verdict
    assert '<tr><th>release</th><td class="id">rel-1</td></tr>' in verdict
    assert (
        'href="/reef/harness/releases/1/page?scenario=agents&amp;token=secret">step 1: the version page</a>' in verdict
    )
    assert "error</th>" not in verdict and "proposer failure" not in verdict
    changed = _section(page, "What changed")
    assert '<li><span class="tag">create</span>r1 <span class="tag">rules</span></li>' in changed
    assert settled_step(rows, RECORD_ID) == 1

    # Opened with headers, the page links the version page the same way: no query.
    bare = build_request_page(_record(compacted_at=1_050.0), rows, now=1_100.0)
    assert 'href="/reef/harness/releases/1/page">step 1' in bare

    # A composite proposal lists every mutation; a rejected step names the gate's reason and the next action.
    second = {"op": "update", "id": "ext", "options": {"name": "code_extension", "config": {"code": "x"}}}
    rejected = _row(
        _answered(
            selected=False,
            mutations=[MUTATION, second],
            selection={"reason": "candidate missed the floor on 1 of 1 tasks"},
        )
    )
    page = build_request_page(_record(compacted_at=1_050.0), [CREATION, rejected], now=1_100.0)
    assert '<span class="rejected">rejected</span>' in _sub(page)
    assert (
        "did not pass the gate (candidate missed the floor on 1 of 1 tasks); nothing changed: rephrase or split "
        "the request" in page
    )
    assert (
        page.count("<li>") == 2
        and '<span class="tag">update</span>ext <span class="tag">code_extension</span>' in page
    )


def test_a_pending_request_names_the_promote_and_reads_promoted_once_a_promote_row_names_it() -> None:
    pending = _row(_answered(selected=True, mutation=MUTATION), pending=True)
    page = build_request_page(_record(compacted_at=1_050.0), [CREATION, pending], now=1_100.0)
    assert '<span class="pending">pending</span>' in _sub(page)
    assert (
        "ready as release rel-1 but changes an extension, so it waits for your review: /reef-versions 1, then "
        "/reef-versions 1 promote" in page
    )
    promote = _row({}, release_id="rel-2", parent="rel-0", operation="promote", rollback_target_release_id="rel-1")
    page = build_request_page(_record(compacted_at=1_050.0), [CREATION, pending, promote], now=1_100.0)
    assert '<span class="promoted">promoted at step 2</span>' in _sub(page)
    assert "won the gate and was promoted at step 2; the release that step published serves it" in page


def test_a_skipped_request_shows_why_the_proposer_produced_nothing_and_what_the_review_left_uncovered() -> None:
    notes = {
        "design": "one rules entry",
        "failure": "model call failed after 60.0 s (max_tokens=16384): timeout",
        "review": {"verdict": "partial", "covered": ["the trigger"], "uncovered": ["a way to turn it off"]},
    }
    skipped = _row(_answered(skipped="no proposal", proposal_notes=notes), release_id="rel-0")
    page = build_request_page(_record(compacted_at=1_050.0), [CREATION, skipped], now=1_100.0)
    assert REFRESH not in page and '<span class="skipped">skipped</span>' in _sub(page)
    assert _sections(page) == ["Request", "Verdict", "What changed", "Review"]
    verdict = _section(page, "Verdict")
    assert "produced no change (no proposal); nothing changed" in verdict
    assert (
        "<tr><th>proposer failure</th><td>model call failed after 60.0 s (max_tokens=16384): timeout</td></tr>"
        in verdict
    )
    assert '<p class="empty">nothing: the step recorded no mutation</p>' in _section(page, "What changed")
    review = _section(page, "Review")
    assert '<span class="partial">partial</span>' in review and "<li>a way to turn it off</li>" in review
    assert "the trigger" not in review

    # A review that left nothing uncovered says so; a step without a review has no such section.
    complete = {"review": {"verdict": "complete", "covered": ["all of it"], "uncovered": []}}
    row = _row(_answered(selected=True, mutation=MUTATION, proposal_notes=complete))
    page = build_request_page(_record(compacted_at=1_050.0), [CREATION, row], now=1_100.0)
    assert '<span class="complete">complete</span>' in page and "nothing left uncovered" in page
    row = _row(_answered(selected=True, mutation=MUTATION, proposal_notes={"design": "plan"}))
    assert "<h2>Review</h2>" not in build_request_page(_record(compacted_at=1_050.0), [CREATION, row], now=1_100.0)

    # An instruction whose step failed is committed with a skip row that carries the failure.
    failed = _row(_answered(skipped="instruction failed", error="RuntimeError: poison proposer"), release_id="rel-0")
    page = build_request_page(_record(compacted_at=1_050.0), [CREATION, failed], now=1_100.0)
    assert "produced no change (instruction failed)" in page
    assert "<tr><th>error</th><td>RuntimeError: poison proposer</td></tr>" in page


def test_the_page_module_is_ascii_and_the_builder_escapes_the_request_the_notes_and_the_link() -> None:
    MODULE.read_text(encoding="utf-8").encode("ascii")
    text = 'text me <script>alert(1)</script> & "quote" café'
    requires = [{"name": "TWILIO_SID", "kind": "env", "check": "TWILIO_SID"}, {"name": "<x>", "kind": "service"}]
    notes = {"failure": "<b>failed</b>", "review": {"verdict": "partial", "covered": [], "uncovered": ["<i>off</i>"]}}
    row = _row(_answered(skipped="no proposal", proposal_notes=notes), release_id="rel-0")
    page = build_request_page(
        _record(compacted_at=1_050.0, text=text, requires=requires),
        [CREATION, row],
        link_query={"scenario": "a b", "token": "t&<"},
        now=1_100.0,
    )
    page.encode("ascii")
    assert "<script>" not in page and "<b>" not in page and "<i>" not in page
    assert "text me &lt;script&gt;alert(1)&lt;/script&gt; &amp; &quot;quote&quot; caf&#233;" in page
    assert "needs from your machine: TWILIO_SID (env), &lt;x&gt; (service)" in page
    assert "&lt;b&gt;failed&lt;/b&gt;" in page and "<li>&lt;i&gt;off&lt;/i&gt;</li>" in page
    assert 'href="/reef/harness/releases/1/page?scenario=a+b&amp;token=t%26%3C"' in page
    queued = build_request_page(_record(text=text), [CREATION], now=1_100.0)
    queued.encode("ascii")
    assert "<script>" not in queued and "caf&#233;" in queued


def _propose_holding(entered: Event, release: Event):
    def propose(nodes, samples, models, *, requests=()):
        entered.set()
        release.wait(30)
        return MARKER

    return propose


def test_the_page_follows_a_filed_request_from_proposing_to_its_verdict_by_a_browser_link(tmp_path: Path) -> None:
    entered, release = Event(), Event()
    recipe = replace(_recipe(tmp_path, _propose_holding(entered, release)), training_mode="manual")
    dispatcher = _dispatcher(tmp_path, recipe)
    scenario = dispatcher.get_or_create_scenario(SCENARIO)
    assert scenario is not None
    headers = {"x-reef-scenario": SCENARIO, "Authorization": "Bearer secret"}

    async def run() -> None:
        client = TestClient(TestServer(create_app(dispatcher, tokens="secret")))
        await client.start_server()
        try:
            body = {"text": TEXT, "session": SESSION, "release_id": "rel-0"}
            response = await client.post("/reef/train", headers=headers, json=body)
            assert response.status == 200, await response.text()
            record_id = (await response.json())["agent_record_id"]
            link = f"/reef/harness/requests/{record_id}/page"
            assert await asyncio.to_thread(entered.wait, 10)

            # The link a browser opens: no header, the scenario and the token in the query.
            response = await client.get(link, params=QUERY)
            page = await response.text()
            assert response.status == 200 and response.headers["content-type"].startswith("text/html"), page
            assert response.headers["Cache-Control"] == "no-store"
            page.encode("ascii")
            assert REFRESH in page and f"<title>Harness request {record_id[:8]}</title>" in page
            assert '<span class="proposing">proposing</span>' in page and f'<p class="text">{TEXT}</p>' in page
            assert "into the step" in page

            # The version page opens the same way; the wrong token, no token or a token elsewhere does not.
            response = await client.get("/reef/harness/releases/0/page", params=QUERY)
            assert response.status == 200 and "<title>Harness step 0</title>" in await response.text()
            response = await client.get(link, params={**QUERY, "token": "nope"})
            assert response.status == 401 and await response.text() == "invalid service token"
            response = await client.get(link, params={"scenario": SCENARIO})
            assert response.status == 401
            response = await client.get("/reef/harness/releases", params=QUERY)
            assert response.status == 401
            # The header wins when present, and without a scenario from anywhere the page is a 400.
            response = await client.get(link, params=QUERY, headers={"Authorization": "Bearer nope"})
            assert response.status == 401
            response = await client.get(link, params={"token": "secret"})
            assert response.status == 400
            # The headers keep working, and win over a query scenario.
            response = await client.get(link, headers=headers, params={"scenario": "other"})
            assert response.status == 200 and '<span class="proposing">proposing</span>' in await response.text()

            release.set()
            for _ in range(200):
                response = await client.get(link, params=QUERY)
                page = await response.text()
                if REFRESH not in page:
                    break
                await asyncio.sleep(0.05)
            assert response.status == 200 and REFRESH not in page, page
            assert '<span class="selected">selected</span>' in page
            assert "published as release " in page and "restart reef-pi to install it" in page
            assert 'href="/reef/harness/releases/1/page?scenario=agents&amp;token=secret">step 1' in page
            assert '<li><span class="tag">create</span>r1 <span class="tag">rules</span></li>' in page

            # An unknown id, and a record that is no training instruction, are 404s naming the id.
            response = await client.get("/reef/harness/requests/nope/page", params=QUERY)
            assert response.status == 404 and "has no harness request 'nope'" in await response.text()
            inference = AgentRecord.create(
                scenario=SCENARIO,
                request_type=RequestType.INFERENCE,
                payload={"messages": [{"role": "user", "content": "q"}]},
                agent_record_id="i1",
            )
            await asyncio.to_thread(dispatcher.accept_record, inference)
            response = await client.get("/reef/harness/requests/i1/page", params=QUERY)
            assert response.status == 404 and "has no harness request 'i1'" in await response.text()
        finally:
            release.set()
            await client.close()

    try:
        asyncio.run(run())
    finally:
        release.set()
        dispatcher.close()
