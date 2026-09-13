"""One HTML page per filed harness request: where its step stands, then the verdict once the step settles.

``GET /reef/harness/requests/{record_id}/page`` builds it from the request's
agent record, the scenario's catalog rows and the running step's progress.
The catalog row whose ``metrics.training_request.id`` is the record id
settles the request: the page then shows that row's verdict as the version
page words it, the mutations, what the review left uncovered, why the
proposer produced nothing when the step recorded that, and links the
version page. Until then the page names the state the request is in
(``queued`` before a step takes it, ``proposing`` and ``gating`` from the
backend's progress, ``running`` while the trainer holds the request and the
backend reports no phase, ``settling`` while the row that consumed the
record lands) and reloads itself every ``REFRESH_SECONDS``, so a person
opens the link right after asking and watches. Like the version page it
loads no asset and is pure ASCII.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlencode

from reef.service.release_page import STYLE, _esc, mutations_of, verdict_of
from reef.train.cordis_backend.backend import StepProgress

#: Seconds between the page's own reloads while the request is not settled.
REFRESH_SECONDS = 5

#: The states a request passes through before a row settles it; the version page's classes cover the verdicts.
_PROGRESS_STYLE = ".queued,.settling{color:var(--mute)}.proposing,.gating,.running{color:var(--accent)}\n"

#: What each state means, in the words the page prints beside it.
_STATE_WORDS = {
    "queued": "no step has taken the request yet; the trainer runs one step per instruction, oldest first",
    "proposing": "the proposer is writing the change: the served model reads the request and the tree",
    "gating": "the gate is running the candidate through its episodes",
    "running": "the step holds the request and reports no phase; its row follows",
    "settling": "the step that consumed the request is committing its row",
}


def settled_step(rows: Sequence[Mapping[str, Any]], record_id: str) -> int | None:
    """The step whose row answered the request: the one whose ``metrics.training_request.id`` is ``record_id``."""
    for index, row in enumerate(rows):
        metrics = row.get("metrics")
        request = metrics.get("training_request") if isinstance(metrics, Mapping) else None
        if isinstance(request, Mapping) and request.get("id") == record_id:
            return index
    return None


def _elapsed(seconds: float) -> str:
    """Seconds as a person reads them: ``42 s`` under two minutes, else ``3 min 05 s``."""
    whole = max(0, int(seconds))
    if whole < 120:
        return f"{whole} s"
    return f"{whole // 60} min {whole % 60:02d} s"


def _short(release_id: Any) -> str:
    return str(release_id)[:8] if release_id else "-"


def _span(state: str) -> str:
    return f'<span class="{_esc(state.split(" ")[0])}">{_esc(state)}</span>'


def _state(record: Mapping[str, Any], progress: StepProgress | None, consumed: bool) -> str:
    """The unsettled request's state: the backend's phase for it, the trainer's hold on it, else the record's.

    A record that is not compacted waits for its step; a compacted one was
    consumed by a commit whose row is about to show, since the row that
    names the request lands in the same commit as the compaction."""
    if progress is not None and progress.request_id == record["agent_record_id"]:
        return progress.phase
    if consumed:
        return "running"
    return "queued" if record.get("compacted_at") is None else "settling"


def _request(record: Mapping[str, Any]) -> str:
    payload = record.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    parts = [
        f'<p class="text">{_esc(payload.get("text"))}</p>',
        (
            f'<p class="sub">from session <span class="id">{_esc(payload.get("session"))}</span> on release '
            f'<span class="id">{_esc(payload.get("release_id"))}</span></p>'
        ),
    ]
    requires = payload.get("requires")
    items = [item for item in requires if isinstance(item, Mapping)] if isinstance(requires, Sequence) else []
    if items:
        named = ", ".join(f"{item.get('name')} ({item.get('kind')})" for item in items)
        parts.append(f"<p>needs from your machine: {_esc(named)}</p>")
    return "\n".join(parts)


def _progress(record: Mapping[str, Any], state: str, progress: StepProgress | None, now: float) -> str:
    lines = [
        f"<tr><th>state</th><td>{_span(state)}</td></tr>",
        f"<tr><th>meaning</th><td>{_esc(_STATE_WORDS.get(state, state))}</td></tr>",
    ]
    if progress is not None and progress.request_id == record["agent_record_id"]:
        lines.append(f"<tr><th>step time</th><td>{_esc(_elapsed(now - progress.started_at))} into the step</td></tr>")
        if progress.episodes_total is not None:
            lines.append(f"<tr><th>episodes</th><td>{progress.episodes_total} in the gate</td></tr>")
        if progress.step_record:
            lines.append(f'<tr><th>step record</th><td class="id">{_esc(progress.step_record)}</td></tr>')
    else:
        filed = record.get("created_at")
        if isinstance(filed, (int, float)):
            lines.append(f"<tr><th>filed</th><td>{_esc(_elapsed(now - filed))} ago</td></tr>")
    return (
        "<table><tbody>" + "".join(lines) + "</tbody></table>\n"
        f'<p class="sub">this page reloads every {REFRESH_SECONDS} seconds until the step settles</p>'
    )


def _meaning(verdict: str, row: Mapping[str, Any], metrics: Mapping[str, Any], step: int) -> str:
    """What the verdict means for the person who asked, with the next action; the words the session prints."""
    release = _short(row.get("release_id"))
    if verdict == "selected":
        return f"published as release {release}; restart reef-pi to install it (the update notice offers it)"
    if verdict == "pending":
        return (
            f"ready as release {release} but changes an extension, so it waits for your review: "
            f"/reef-versions {step}, then /reef-versions {step} promote"
        )
    if verdict.startswith("promoted"):
        return f"won the gate and was {verdict}; the release that step published serves it"
    if verdict == "rejected":
        selection = metrics.get("selection")
        reason = selection.get("reason") if isinstance(selection, Mapping) else None
        return f"did not pass the gate ({reason or 'the gate refused it'}); nothing changed: rephrase or split the request"
    if verdict == "skipped":
        return f"produced no change ({metrics.get('skipped')}); nothing changed"
    return f"the step ended as {verdict}"


def _step_link(step: int, link_query: Mapping[str, str] | None) -> str:
    href = f"/reef/harness/releases/{step}/page"
    if link_query:
        # The version page opens the way this page was opened: the query parameters travel with the link.
        href += "?" + urlencode(dict(link_query))
    return f'<a href="{_esc(href)}">step {step}: the version page</a>'


def _verdict(step: int, rows: Sequence[Mapping[str, Any]], link_query: Mapping[str, str] | None) -> str:
    row = rows[step]
    metrics = row.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    verdict = verdict_of(row, rows)
    lines = [
        f"<tr><th>verdict</th><td>{_span(verdict)}</td></tr>",
        f"<tr><th>meaning</th><td>{_esc(_meaning(verdict, row, metrics, step))}</td></tr>",
    ]
    if metrics.get("error"):
        # An instruction whose step failed is committed with a skip row that carries the failure.
        lines.append(f"<tr><th>error</th><td>{_esc(metrics['error'])}</td></tr>")
    notes = metrics.get("proposal_notes")
    failure = notes.get("failure") if isinstance(notes, Mapping) else None
    if isinstance(failure, str) and failure.strip():
        lines.append(f"<tr><th>proposer failure</th><td>{_esc(failure)}</td></tr>")
    lines.append(f'<tr><th>release</th><td class="id">{_esc(row.get("release_id"))}</td></tr>')
    lines.append(f"<tr><th>page</th><td>{_step_link(step, link_query)}</td></tr>")
    return "<table><tbody>" + "".join(lines) + "</tbody></table>"


def _what_changed(metrics: Mapping[str, Any]) -> str:
    mutations = mutations_of(metrics)
    if not mutations:
        return '<p class="empty">nothing: the step recorded no mutation</p>'
    items = []
    for mutation in mutations:
        options = mutation.get("options")
        kind = options.get("name") if isinstance(options, Mapping) else None
        items.append(
            f'<li><span class="tag">{_esc(mutation.get("op") or "?")}</span>{_esc(mutation.get("id") or "?")} '
            f'<span class="tag">{_esc(kind or "?")}</span></li>'
        )
    return "<ul>" + "".join(items) + "</ul>"


def _review(metrics: Mapping[str, Any]) -> str:
    """The Review section, only when the step recorded one: the verdict and what the entries left uncovered."""
    notes = metrics.get("proposal_notes")
    review = notes.get("review") if isinstance(notes, Mapping) else None
    if not isinstance(review, Mapping):
        return ""
    uncovered = review.get("uncovered")
    items = [item for item in uncovered if isinstance(item, str)] if isinstance(uncovered, Sequence) else []
    listed = "<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in items) + "</ul>" if items else ""
    return (
        f"<h2>Review</h2>\n<p>the proposer's review of its entries against the request: "
        f'{_span(str(review.get("verdict") or "unknown"))}</p>\n'
        + (f"<h3>uncovered</h3>{listed}\n" if items else '<p class="empty">nothing left uncovered</p>\n')
    )


def build_request_page(
    record: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    progress: StepProgress | None = None,
    consumed: bool = False,
    link_query: Mapping[str, str] | None = None,
    now: float | None = None,
) -> str:
    """The page for the request stored as ``record``, against the catalog ``rows`` oldest first.

    ``record`` is the agent record as ``Dispatcher.read_record`` answers it
    (``agent_record_id``, ``created_at``, ``compacted_at`` and the
    ``POST /reef/train`` payload). ``progress`` is the training backend's
    running step, counted only when it names this request; ``consumed`` says
    whether the trainer's reserved batch carries the request. ``link_query``
    is carried to the version page link. ``now`` is the clock the elapsed
    times are read against. Pure ASCII out: other characters leave as
    numeric references.
    """
    record_id = str(record["agent_record_id"])
    now = time.time() if now is None else now
    step = settled_step(rows, record_id)
    state = verdict_of(rows[step], rows) if step is not None else _state(record, progress, consumed)
    title = f"Harness request {record_id[:8]}"
    sub = f'request <span class="id">{_esc(record_id)}</span> | {_span(state)}'
    filed = record.get("created_at")
    if isinstance(filed, (int, float)):
        sub += f" | filed at {filed:.0f}"
    head = "" if step is not None else f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">\n'
    if step is None:
        body = f"<h2>Progress</h2>\n{_progress(record, state, progress, now)}\n"
    else:
        metrics = rows[step].get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        body = (
            f"<h2>Verdict</h2>\n{_verdict(step, rows, link_query)}\n"
            f"<h2>What changed</h2>\n{_what_changed(metrics)}\n"
            f"{_review(metrics)}"
        )
    page = (
        f"{head}<title>{title}</title>\n<style>{STYLE}{_PROGRESS_STYLE}</style>\n<main>\n"
        f'<h1>{title}</h1>\n<p class="sub">{sub}</p>\n'
        f"<h2>Request</h2>\n{_request(record)}\n"
        f"{body}"
        "</main>\n"
    )
    return page.encode("ascii", "xmlcharrefreplace").decode("ascii")


__all__ = ["REFRESH_SECONDS", "build_request_page", "settled_step"]
