"""Harbor agent that runs one CEO-Bench episode with the agent role served by Reef.

The Harbor environment holds a pinned CEO-Bench checkout
(``harbor/environment/Dockerfile``). ``run()`` starts a reef-client sidecar on
this host that stamps the scenario and token onto every model call and keeps
each exchange with its receipt, then runs the benchmark's own bash-agent
runner inside the container with the agent role pointed at that sidecar over
``/v1/chat/completions``. The two simulator roles keep the benchmark's provider
settings and never pass through Reef.

The reward is online and weekly. Every request the agent sends carries the
dashboard of the simulated week it is in (``=== Week N Dashboard (Day D) ===``
with the week's opening cash, subscribers, seats, and listed prices), so the
sidecar's captures say which week each turn belongs to and what the week
started with. While the episode runs, a reporter thread watches for the next
week's dashboard; when it appears the previous week is over, and every turn
of that week is reported with the week's change in company value as its
score: cash plus the subscription run-rate the dashboard implies, over the
weeks left in the episode, credited over the next ``CEOBENCH_CREDIT_WEEKS``
weeks with a discount and scaled against the weeks before it
(``harness.report``). The last weeks end with the verifier's final cash,
which a watcher thread reads from Harbor's ``result.json`` after the trial.
When the episode ends, the run directory (``world.nmdb``, config, checkpoint,
logs) is copied into the trial's log directory and the receipts, with their
weeks and token counts, go into the agent context in call order.

Connection settings come from the environment set by ``run.sh``:

- ``REEF_SERVICE_URL`` (required): the Reef service as the task container
  reaches it, so a LAN address rather than ``127.0.0.1``
- ``REEF_SCENARIO``, ``REEF_TOKEN``, ``REEF_TIMEOUT_S``

Variables named ``SAAS_BENCH_*``, ``OPENAI_*``, ``ANTHROPIC_*``, and ``AWS_*``
are forwarded into the container for the simulator roles, so their credentials
and any provider override stay outside the repository.

``CEOBENCH_TRAIN_MAX_TOKENS`` (0 or unset: no limit) is the trainer's window:
the engine serves the model's full context, but a turn whose prompt and
completion together exceed this many tokens is recorded and never reported,
because the trainer could not hold it.

``CEOBENCH_VALUE_HORIZON_WEEKS`` (default 26) caps the weeks of run-rate a
week's opening state is valued at, so a subscriber is worth at most about six
months of the listed price and the valuation converges on cash as the episode
ends. ``CEOBENCH_CREDIT_WEEKS`` (default 4) and ``CEOBENCH_CREDIT_DISCOUNT``
(default 0.8) set how many later weeks' value changes a week is credited with
and how fast they discount; a week is reported once that many weeks have
opened after it. ``CEOBENCH_SCORE_CLIP`` (default 0.05) and
``CEOBENCH_SCORE_FLOOR`` (default 0.003) clip a week's credit and floor the
running scale it is divided by before it is posted.

Only a week's decision turns are reported: the turns whose tool call changed
the company (prices, spend, targeting, research, deals, posts, or the week
advanced; ``DECISION_CALLS``). Turns that only queried the books, read docs,
or wrote workspace files are recorded but not trained on, so a bad week does
not teach the policy to stop looking before it acts.

``CEOBENCH_ENGINE_READS`` (default on; ``0`` turns it off) has the week gate
read the engine's own monthly recurring revenue through the task container
at each week start, so the valuation uses the books rather than the
listed-price estimate; the agent's observation is untouched.

``CEOBENCH_REPORTS`` (default on; ``0`` turns it off) is the untrained
baseline switch: the harness still groups the turns by week and records the
weekly values in the trial metadata, but posts no report, so the stack in
``serve.yaml`` serves the base model for the whole episode and trains
nothing. The pacer is off with it.

``CEOBENCH_PACE_BATCH`` (0 or unset: off) paces the game to the trainer. Set
to the recipe's batch size, the sidecar holds a request until every batch
the reported weeks filled has committed a training release, so a week is
played by a policy trained on every week the credit window has closed, and
no turn is generated while a step publishes its adapter.
``CEOBENCH_PACE_TIMEOUT_S`` (default 1800) bounds one such wait; a batch the
recipe declined would otherwise hold the game forever.
"""

import asyncio
import atexit
import io
import json
import os
import re
import shlex
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import NamedTuple
from urllib.parse import urlsplit

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from reef_client import ReefClient
from reef_client.serve import CaptureStore, ServeConfig, build_handler

from .report import (
    DEFAULT_CREDIT_DISCOUNT,
    DEFAULT_CREDIT_WEEKS,
    DEFAULT_SCORE_CLIP,
    DEFAULT_SCORE_FLOOR,
    DEFAULT_VALUE_HORIZON_WEEKS,
    ScoreScale,
    post_week_reports,
    valuation,
    week_credit,
)

#: The pinned checkout and the run root inside the task container.
CEOBENCH_DIR = "/opt/ceobench"
RUNS_DIR = "/workspace/ceobench-runs"
#: Episode defaults; ``kwargs`` on the Harbor agent config override them.
DEFAULT_SEED = 42
DEFAULT_DAYS = 500
#: Environment variables the simulator roles read; forwarded verbatim.
FORWARDED_ENV_PREFIXES = ("SAAS_BENCH_", "OPENAI_", "ANTHROPIC_", "AWS_")
#: How often the reporter looks for a finished week while the episode runs.
WEEK_POLL_S = 5.0
#: How often the pacer re-reads the scenario's releases while it holds a week.
PACE_POLL_S = 10.0
#: How long one read of the engine's books may take (a docker exec and one SQL query).
ENGINE_READ_TIMEOUT_S = 120.0
#: Runs inside the task container: asks the runner's engine for the monthly
#: recurring revenue of the subscribed base (the engine's own definition:
#: each individual subscription at its effective price, each enterprise
#: subscription at its per-seat price times seats) and the week's ledger by
#: category. Prints one JSON line; ``error`` when the engine cannot answer.
ENGINE_READ = """
import glob, json, re, urllib.request
try:
    def alive(candidate):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{candidate}/health", timeout=5) as response:
                return response.status == 200
        except Exception:
            return False
    ports = []
    for path in glob.glob("/workspace/ceobench-runs/run_*/checkpoint.json"):
        try:
            ports.append(int(json.load(open(path)).get("api_server_port") or 0))
        except Exception:
            pass
    try:  # the runner prints the port at start, but its log is block-buffered
        ports += [int(p) for p in re.findall(r"Server started: port=(\\d+)", open("/workspace/ceobench-runs/runner.log", errors="replace").read())]
    except Exception:
        pass
    for line in open("/proc/net/tcp").read().splitlines()[1:]:  # every loopback listener, engine included
        fields = line.split()
        if len(fields) > 3 and fields[3] == "0A" and fields[1].startswith("0100007F:"):
            ports.append(int(fields[1].split(":")[1], 16))
    port = next(candidate for candidate in dict.fromkeys(ports) if candidate and alive(candidate))
    def query(sql):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/query", data=json.dumps({"sql": sql}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read())
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(result.get("error") or "query failed")
        data = result.get("data", result) if isinstance(result, dict) else result
        return data.get("rows") or []
    # Live subscriptions at their effective price times seats: the engine's own
    # MRR definition. seat_count is exposed on subscriptions (an integer; 1 for
    # an individual), so no join with the customers table is needed.
    subscribed = "status = 'subscribed' AND end_day IS NULL"
    try:
        rows = query(
            "SELECT COALESCE(SUM(effective_price * COALESCE(seat_count, 1)), 0) AS mrr"
            " FROM subscriptions WHERE " + subscribed
        )
        basis = "engine"
    except Exception:
        rows = query("SELECT COALESCE(SUM(effective_price), 0) AS mrr FROM subscriptions WHERE " + subscribed)
        basis = "effective_price"
    mrr = float(list(rows[0].values())[0]) if rows else 0.0
    ledger = {}
    try:
        for row in query(
            "SELECT category, COALESCE(SUM(amount), 0) AS total FROM ledger"
            " WHERE day > (SELECT COALESCE(MAX(day), 0) - 7 FROM ledger) GROUP BY category"
        ):
            values = list(row.values())
            ledger[str(values[0])] = float(values[1])
    except Exception:
        pass
    print(json.dumps({"mrr": mrr, "basis": basis, "ledger_week": ledger}))
except Exception as error:
    print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
"""
#: The weekly dashboard header the benchmark's engine returns, with the
#: week's opening cash, individual subscribers, and enterprise seats on the
#: lines after it.
DASHBOARD_RE = re.compile(
    r"=== Week (\d+) Dashboard \(Day (\d+)\) ===\s*\n\s*\n"
    r"Cash: (-?)\$(-?[\d,]+)\n"
    r"Individual Subscribers: (\d+)\n"
    r"Enterprise Subscribed Seats: (\d+)"
)
#: The listed plan prices in the same dashboard's configuration block. The
#: agent's own scripts print ``Prices:`` lines too, so the block anchors it.
PRICES_RE = re.compile(r"--- Current Config ---\s*\nPrices: A=\$(\d+), B=\$(\d+), C=\$(\d+)")


class WeekStart(NamedTuple):
    """A week's opening state as its dashboard shows it, plus the engine's own MRR when read."""

    week: int
    day: int
    cash: float
    subscribers: int
    seats: int
    prices: tuple[float, float, float]  # plans A, B, C as listed, monthly
    mrr: float | None = None  # the engine's monthly recurring revenue, when the harness read it

    @property
    def run_rate(self) -> float:
        """Monthly subscription revenue: the engine's MRR when read, else an estimate from the dashboard.

        The estimate counts each individual subscriber at the lowest nonzero
        listed price and each enterprise seat at plan C's: the dashboard
        shows neither the plan mix nor negotiated seat prices, so it is a
        floor, not the books.
        """
        if self.mrr is not None:
            return self.mrr
        listed = [price for price in self.prices if price > 0]
        return self.subscribers * (min(listed) if listed else 0.0) + self.seats * self.prices[2]


#: SDK calls and CLI commands that change the company: money spent, prices,
#: targeting, research, deals, posts, and the week advanced. Everything else
#: the agent can do (queries, status, reading docs, files in its workspace)
#: only reads, and such turns are recorded but not trained on: a week's
#: outcome is credited to the decisions in it, not to looking at the books.
DECISION_CALLS = (
    "next-week",
    "next_week",
    "set_prices",
    "set_promotion",
    "set_lead_promotion",
    "set_model_tiers",
    "set_usage_quotas",
    "set_capacity_tier",
    "set_daily_spend",
    "set_targeted_ad_spend",
    "set_targeted_dev_spend",
    "set_targeted_ops_spend",
    "set_ads_strength",
    "start_research_project",
    "research_market",
    "research_group",
    "send_enterprise_deal",
    "reject_enterprise_deal",
    "post_social_media",
)
DECISION_RE = re.compile(r"\b(" + "|".join(re.escape(call) for call in DECISION_CALLS) + r")\b")
#: A bash command that runs a Python script file (not inline ``python-c`` code).
SCRIPT_RUN_RE = re.compile(r"\bpython3?\s+(?!-c\b)(\S+\.py)\b")


def tool_calls(turn: dict) -> list[tuple[str, dict]]:
    """``(tool name, arguments)`` of every tool call in a captured turn's response."""
    calls = []
    for choice in (turn.get("response") or {}).get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        for call in (message or {}).get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {"raw": arguments}
            calls.append((str(function.get("name") or ""), arguments if isinstance(arguments, dict) else {}))
    return calls


def turn_decision(turn: dict, scripts: dict[str, str]) -> str | None:
    """The company-changing call a turn made, or ``None`` for a turn that only read.

    ``scripts`` holds the files the agent has written so far (path to
    content), updated here as ``write_file``/``edit_file`` calls pass, so a
    bash command that runs one of them is judged by what it contains; a
    script this episode did not write is taken to act.
    """
    for name, arguments in tool_calls(turn):
        if name in ("write_file", "edit_file"):
            path = str(arguments.get("path") or "")
            content = str(arguments.get("content") or arguments.get("new_string") or "")
            if path:
                scripts[path] = content if name == "write_file" else scripts.get(path, "") + "\n" + content
            continue
        if name != "bash":
            continue
        command = str(arguments.get("command") or "")
        found = DECISION_RE.search(command)
        if found:
            return found.group(1).replace("-", "_")
        for match in SCRIPT_RUN_RE.finditer(command):
            script = match.group(1)
            basename = script.rsplit("/", 1)[-1]
            content = next((body for path, body in scripts.items() if path.rsplit("/", 1)[-1] == basename), None)
            if content is None:
                return "script"
            found = DECISION_RE.search(content)
            if found:
                return found.group(1).replace("-", "_")
    return None


def runner_command(base_url: str, model: str, seed: int, days: int) -> str:
    """The benchmark's bash-agent baseline, with the agent role at ``base_url``."""
    args = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "saas_bench.agents.bash_agent.run_test",
        "--provider",
        "openai",
        "--base-url",
        base_url,
        "--api-key",
        "reef",  # the sidecar replaces it with the Reef token
        "--model",
        model,
        "--reasoning-effort",
        "none",
        "--seed",
        str(seed),
        "--days",
        str(days),
        "--workspace",
        RUNS_DIR,
    ]
    return f"mkdir -p {RUNS_DIR} && cd {CEOBENCH_DIR} && {shlex.join(args)} > {RUNS_DIR}/runner.log 2>&1"


def turn_tokens(turn: dict) -> int:
    """Prompt plus completion tokens of one captured turn (0 when unreported)."""
    usage = (turn.get("response") or {}).get("usage") or {}
    return int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)


def dashboards(content: str) -> list[WeekStart]:
    """Every dashboard in ``content``, in order of appearance.

    The prices are looked for between a dashboard's header and the next
    one's; a dashboard without its configuration block (cut short, or quoted
    without it) keeps its week and cash and lists no prices.
    """
    starts = []
    matches = list(DASHBOARD_RE.finditer(content))
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(content)
        priced = PRICES_RE.search(content, match.end(), end)
        prices = tuple(float(price) for price in priced.groups()) if priced else (0.0, 0.0, 0.0)
        sign = -1.0 if match.group(3) == "-" else 1.0
        starts.append(
            WeekStart(
                week=int(match.group(1)),
                day=int(match.group(2)),
                cash=sign * float(match.group(4).replace(",", "")),
                subscribers=int(match.group(5)),
                seats=int(match.group(6)),
                prices=(prices[0], prices[1], prices[2]),
            )
        )
    return starts


def turn_week(turn: dict) -> WeekStart | None:
    """The opening state of the latest week whose dashboard the turn's request carries.

    The runner rebuilds the conversation from the new dashboard after every
    ``next-week``; taking the latest header also covers a transcript that
    still carries an earlier week's dashboard. ``None`` when the request has
    no dashboard at all.
    """
    latest: WeekStart | None = None
    for message in (turn.get("request") or {}).get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            continue
        for start in dashboards(content):
            if latest is None or start.week > latest.week:
                latest = start
    return latest


def forwarded_environment(environ: dict[str, str]) -> dict[str, str]:
    forwarded = {key: value for key, value in environ.items() if key.startswith(FORWARDED_ENV_PREFIXES)}
    # Reef serves /v1/chat/completions, not the Responses API the runner
    # prefers for OpenAI-compatible endpoints (see reef.patch).
    forwarded["SAAS_BENCH_OPENAI_CHAT_COMPLETIONS"] = "1"
    return forwarded


class WeekLedger:
    """The episode's turns grouped by simulated week, in call order.

    ``observe`` reads the sidecar's captures; a served turn without a
    dashboard of its own belongs to the week the previous turn was in. A week
    is valued from its opening state (:func:`harness.report.valuation`) with
    ``total_weeks - week`` weeks left, at most ``horizon_weeks`` of them, and
    credited with the value changes of the ``credit_weeks`` weeks from it on,
    discounted by ``discount`` per week (:func:`harness.report.week_credit`).
    """

    def __init__(
        self,
        total_weeks: int = 0,
        horizon_weeks: int = DEFAULT_VALUE_HORIZON_WEEKS,
        credit_weeks: int = DEFAULT_CREDIT_WEEKS,
        discount: float = DEFAULT_CREDIT_DISCOUNT,
    ) -> None:
        if credit_weeks < 1:
            raise ValueError("credit_weeks must be at least 1")
        self.total_weeks = total_weeks
        self.horizon_weeks = horizon_weeks
        self.credit_weeks = credit_weeks
        self.discount = discount
        self.weeks: dict[int, dict] = {}  # week -> {"start": WeekStart, "turns": [(receipt, tokens)]}
        self.turns: list[dict] = []  # {"receipt", "tokens", "week"} per served turn
        self.posted: set[int] = set()
        self.scores: dict[int, float] = {}  # week -> the scaled score it was posted with
        #: Weeks whose dashboard was seen on a request not yet served (the
        #: pacer's peek), so the week before can close before that request
        #: is forwarded.
        self.announced: dict[int, WeekStart] = {}

    def announce(self, start: WeekStart) -> None:
        self.announced.setdefault(start.week, start)

    def observe(self, captured: list[dict]) -> None:
        current: int | None = None
        scripts: dict[str, str] = {}
        self.weeks = {week: {"start": start, "turns": []} for week, start in self.announced.items()}
        self.turns = []
        for turn in captured:
            if turn.get("status") != 200 or not turn.get("receipt"):
                continue
            seen = turn_week(turn)
            if seen is not None:
                self.weeks.setdefault(seen.week, {"start": seen, "turns": []})
                current = seen.week
            record = {
                "receipt": turn["receipt"],
                "tokens": turn_tokens(turn),
                "week": current,
                "decision": turn_decision(turn, scripts),
            }
            self.turns.append(record)
            if current is not None:
                self.weeks[current]["turns"].append((record["receipt"], record["tokens"], record["decision"]))

    def value(self, start: WeekStart) -> float:
        """What a week's opening state is worth, given the weeks left after it."""
        return valuation(start.cash, start.run_rate, self.total_weeks - start.week, self.horizon_weeks)

    def _closing(
        self, ordered: list[int], position: int, final_cash: float | None
    ) -> tuple[float, float, float] | None:
        """``(cash_end, value_end, credit)`` of the week at ``position``, or ``None`` while it is open.

        The week closes with the opening state of the next week seen and is
        credited with the value changes of the ``credit_weeks`` weeks from it
        on, so it stays open until that many later weeks have opened. With
        ``final_cash`` the episode is over: the last week seen ends there,
        valued as cash alone, and a window that reaches the end is cut short.
        """
        deltas: list[float] = []
        closing: tuple[float, float] | None = None
        for offset in range(self.credit_weeks):
            index = position + offset
            value = self.value(self.weeks[ordered[index]]["start"])
            if index + 1 < len(ordered):
                start = self.weeks[ordered[index + 1]]["start"]
                cash_end, value_end = start.cash, self.value(start)
            elif final_cash is not None:
                cash_end, value_end = final_cash, final_cash
            else:
                return None
            deltas.append(value_end - value)
            closing = closing or (cash_end, value_end)
            if index + 1 >= len(ordered):
                break  # the episode ended inside the window
        if closing is None:
            return None
        return (*closing, week_credit(deltas, self.discount))

    def finished_weeks(self, final_cash: float | None = None) -> list[tuple[int, float, float, float]]:
        """Unreported weeks whose credit is known, as ``(week, cash_end, value_end, credit)``."""
        ordered = sorted(self.weeks)
        finished = []
        for position, week in enumerate(ordered):
            if week in self.posted:
                continue
            closing = self._closing(ordered, position, final_cash)
            if closing is not None:
                finished.append((week, *closing))
        return finished

    def summary(self, final_cash: float | None = None) -> list[dict]:
        ordered = sorted(self.weeks)
        rows = []
        for position, week in enumerate(ordered):
            entry = self.weeks[week]
            start: WeekStart = entry["start"]
            closing = self._closing(ordered, position, final_cash) or (None, None, None)
            rows.append(
                {
                    "week": week,
                    "day": start.day,
                    "cash_start": start.cash,
                    "cash_end": closing[0],
                    "subscribers": start.subscribers,
                    "seats": start.seats,
                    "run_rate": start.run_rate,
                    "value_start": self.value(start),
                    "value_end": closing[1],
                    "credit": closing[2],
                    "score": self.scores.get(week),
                    "turns": len(entry["turns"]),
                    "reported": week in self.posted,
                }
            )
        return rows


class TrainingPacer:
    """Hold a new week until the trainer has consumed the weeks before it.

    ``expected_releases`` is what the reported turns must have produced: one
    training release per full batch. ``wait`` blocks until the scenario has
    that many releases beyond the count at episode start, or the timeout
    passes, in which case the shortfall is forgiven so later weeks do not
    wait for a batch the recipe declined.
    """

    def __init__(self, batch_size: int, timeout_s: float, count_releases, logger) -> None:
        self.batch_size = int(batch_size)
        self.timeout_s = float(timeout_s)
        self._count_releases = count_releases
        self._logger = logger
        self._base: int | None = None
        self._forgiven = 0
        self._lock = threading.Lock()

    def expected_releases(self, posted_turns: int) -> int:
        return posted_turns // self.batch_size - self._forgiven

    def wait(self, week: int, posted_turns: int) -> float:
        """Block until the trainer caught up; return the seconds spent waiting."""
        started = time.time()
        with self._lock:
            if self._base is None:
                self._base = self._count_releases() or 0
            expected = self.expected_releases(posted_turns)
            while True:
                observed = (self._count_releases() or 0) - self._base
                if observed >= expected:
                    break
                if time.time() - started >= self.timeout_s:
                    self._forgiven += expected - observed
                    self._logger.warning(
                        "week %d: trainer committed %d of %d expected releases after %.0fs; going on",
                        week,
                        observed,
                        expected,
                        self.timeout_s,
                    )
                    break
                time.sleep(PACE_POLL_S)
        return time.time() - started


def paced_handler(base_handler, gate):
    """Wrap the sidecar's handler so a request waits while a filled batch is still training."""

    class PacedHandler(base_handler):
        def _forward(self, forward_path: str, routed_session):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            if forward_path.endswith("/chat/completions") and raw:
                try:
                    body = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    body = None
                seen = turn_week({"request": body}) if isinstance(body, dict) else None
                if seen is not None:
                    gate(seen)
            self.rfile = io.BytesIO(raw)
            return super()._forward(forward_path, routed_session)

    return PacedHandler


class HarborAgent(BaseAgent):
    """One Harbor trial, one CEO-Bench episode through Reef."""

    def __init__(self, *args, seed: int = DEFAULT_SEED, days: int = DEFAULT_DAYS, **kwargs):
        super().__init__(*args, **kwargs)
        self._service_url = os.environ.get("REEF_SERVICE_URL", "").rstrip("/")
        if not self._service_url:
            raise ValueError("the ceobench harness requires REEF_SERVICE_URL")
        self._scenario = os.environ.get("REEF_SCENARIO", "ceobench-sao")
        self._token = os.environ.get("REEF_TOKEN", "reef-local")
        self._seed = int(seed)
        self._days = int(days)
        self._client = ReefClient(
            self._service_url, token=self._token, timeout_s=float(os.environ.get("REEF_TIMEOUT_S", "7200"))
        )
        self._init_week_reporting()
        # Harbor runs the verifier after run() returns and ends the trial by
        # writing result.json; watch for it from construction, so the last
        # week's reward still reaches Reef when the episode itself failed late.
        self._report_watch_from = time.time()
        self._reporter = threading.Thread(target=self._report_trial_result, daemon=True)
        self._reporter.start()
        atexit.register(self._reporter.join, 120.0)  # hundreds of reports; don't drop them

    def _init_week_reporting(self) -> None:
        self._capture = CaptureStore()
        environ = os.environ
        self._ledger = WeekLedger(
            total_weeks=self._days // 7,
            horizon_weeks=int(environ.get("CEOBENCH_VALUE_HORIZON_WEEKS", "") or DEFAULT_VALUE_HORIZON_WEEKS),
            credit_weeks=int(environ.get("CEOBENCH_CREDIT_WEEKS", "") or DEFAULT_CREDIT_WEEKS),
            discount=float(environ.get("CEOBENCH_CREDIT_DISCOUNT", "") or DEFAULT_CREDIT_DISCOUNT),
        )
        self._scale = ScoreScale(
            clip=float(environ.get("CEOBENCH_SCORE_CLIP", "") or DEFAULT_SCORE_CLIP),
            floor=float(environ.get("CEOBENCH_SCORE_FLOOR", "") or DEFAULT_SCORE_FLOOR),
        )
        self._engine_reads = (environ.get("CEOBENCH_ENGINE_READS", "1") or "1") != "0"
        self._reports = (environ.get("CEOBENCH_REPORTS", "1") or "1") != "0"
        self._ledger_lock = threading.Lock()
        self._max_tokens = int(os.environ.get("CEOBENCH_TRAIN_MAX_TOKENS", "0") or 0)
        # Nothing is reported for an untrained baseline, so nothing trains and
        # there is no batch to wait for.
        batch = int(os.environ.get("CEOBENCH_PACE_BATCH", "0") or 0) if self._reports else 0
        self._pacer = (
            TrainingPacer(
                batch,
                float(os.environ.get("CEOBENCH_PACE_TIMEOUT_S", "1800") or 1800),
                self._count_training_releases,
                self.logger,
            )
            if batch > 0
            else None
        )
        self._gated_week: int | None = None

    @staticmethod
    def name() -> str:
        return "reef-ceobench"

    def version(self) -> str | None:
        return None

    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the image carries the pinned checkout."""

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        # The week gate reads the engine's books through the task container
        # from the sidecar's thread, so it needs this loop and environment.
        self._environment = environment
        self._loop = asyncio.get_running_loop()
        server = self._start_sidecar()
        episode_over = threading.Event()
        weekly = threading.Thread(target=self._report_weeks_online, args=(episode_over,), daemon=True)
        weekly.start()
        try:
            sidecar_host = urlsplit(self._service_url).hostname or "172.17.0.1"
            base_url = f"http://{sidecar_host}:{server.server_address[1]}/v1"
            command = runner_command(base_url, self.model_name or "reef", self._seed, self._days)
            self.logger.info("ceobench seed=%s days=%s agent via %s", self._seed, self._days, base_url)
            result = await environment.exec(command, env=forwarded_environment(dict(os.environ)))
        finally:
            episode_over.set()
            weekly.join()
            server.shutdown()

        turns = self._capture.snapshot()
        self._post_finished_weeks()
        with self._ledger_lock:
            ledger_turns = list(self._ledger.turns)
            weeks = self._ledger.summary()
        await environment.download_dir(RUNS_DIR, self.logs_dir / "ceobench")
        context.metadata = {
            **(context.metadata or {}),
            "reef": {
                "agent_record_ids": [turn["receipt"] for turn in ledger_turns],
                "agent_record_tokens": [turn["tokens"] for turn in ledger_turns],
                "agent_record_weeks": [turn["week"] for turn in ledger_turns],
                "agent_record_decisions": [turn["decision"] for turn in ledger_turns],
            },
            "ceobench": {
                "seed": self._seed,
                "days": self._days,
                "horizon_weeks": self._ledger.horizon_weeks,
                "credit_weeks": self._ledger.credit_weeks,
                "discount": self._ledger.discount,
                "score_clip": self._scale.clip,
                "score_floor": self._scale.floor,
                "engine_reads": self._engine_reads,
                "reports": self._reports,
                "turns": len(turns),
                "exit_code": result.return_code,
                "weeks": weeks,
            },
        }
        usage = [((turn.get("response") or {}).get("usage") or {}) for turn in turns]
        context.n_input_tokens = sum(int(item.get("prompt_tokens") or 0) for item in usage)
        context.n_output_tokens = sum(int(item.get("completion_tokens") or 0) for item in usage)
        if result.return_code != 0:
            raise RuntimeError(f"ceobench runner exited {result.return_code}; see {self.logs_dir / 'ceobench'}")

    def _start_sidecar(self) -> ThreadingHTTPServer:
        # Override, not setdefault: the runner's OpenAI client sends its own
        # Authorization header, and the scenario is this harness's to choose.
        config = ServeConfig(
            upstream=self._service_url,
            listen_host="0.0.0.0",
            listen_port=0,  # an ephemeral port, so concurrent trials never collide
            override_headers={"x-reef-scenario": self._scenario, "authorization": f"Bearer {self._token}"},
        )
        self._capture = CaptureStore()
        handler = build_handler(config, self._capture)
        if self._pacer is not None:
            handler = paced_handler(handler, self._gate_week)
        server = ThreadingHTTPServer((config.listen_host, config.listen_port), handler)
        threading.Thread(target=server.serve_forever, name="ceobench-sidecar", daemon=True).start()
        return server

    def _gate_week(self, start: WeekStart) -> None:
        """Before a request is served, close the weeks its week finishes and wait for training.

        A new week's first request closes the weeks the credit window
        finished and reports them; every request then waits until the
        batches the reported turns filled have committed, so no turn is
        generated while a step publishes its adapter (the engine cannot swap
        the adapter under a request in flight).
        """
        if self._pacer is None:
            return
        week = start.week
        if self._gated_week is None or week > self._gated_week:
            self._gated_week = week
            books = self._read_engine() if self._engine_reads else None
            if books is not None and books.get("mrr") is not None:
                start = start._replace(mrr=float(books["mrr"]))
            self.logger.info(
                "week %d starts (day %d): cash %.0f, %d subscribers, %d seats, run-rate %.0f/month (%s)",
                week,
                start.day,
                start.cash,
                start.subscribers,
                start.seats,
                start.run_rate,
                "engine MRR" if start.mrr is not None else "listed-price estimate",
            )
            with self._ledger_lock:
                self._ledger.announce(start)
            self._post_finished_weeks()
        with self._ledger_lock:
            posted = sum(
                self._reportable(entry["turns"])
                for posted_week, entry in self._ledger.weeks.items()
                if posted_week in self._ledger.posted
            )
        waited = self._pacer.wait(week, posted)
        if waited > 1.0:
            self.logger.info("week %d held %.0fs for training", week, waited)

    def _reportable(self, turns) -> int:
        return sum(
            1
            for _receipt, tokens, decision in turns
            if decision is not None and (not self._max_tokens or tokens <= self._max_tokens)
        )

    def _read_engine(self) -> dict | None:
        """The engine's own books at this moment, read through the task container.

        The runner's engine keeps the live world in memory and answers SQL on
        ``/query``; its port is in the runner's log. The read runs as the
        container's root, outside the agent's shell, and changes nothing the
        agent sees. ``None`` when the read fails; the caller then falls back
        to the dashboard estimate.
        """
        environment, loop = getattr(self, "_environment", None), getattr(self, "_loop", None)
        if environment is None or loop is None:
            return None
        command = f"python3 - <<'PY'\n{ENGINE_READ}\nPY"
        try:
            future = asyncio.run_coroutine_threadsafe(environment.exec(command), loop)
            result = future.result(timeout=ENGINE_READ_TIMEOUT_S)
            line = (result.stdout or "").strip().splitlines()[-1]
            books = json.loads(line)
        except Exception as error:
            self.logger.warning("engine read failed; using the dashboard estimate: %s", error)
            return None
        if not isinstance(books, dict):
            return None
        if books.get("error"):
            self.logger.warning("engine read failed; using the dashboard estimate: %s", books["error"])
            return None
        return books

    def _count_training_releases(self) -> int | None:
        request = urllib.request.Request(
            f"{self._service_url}/reef/scenarios/{self._scenario}/releases",
            headers={"Authorization": f"Bearer {self._token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, ValueError):
            return None
        return sum(1 for row in payload.get("releases", []) if row.get("operation") == "training")

    def _report_weeks_online(self, episode_over: threading.Event) -> None:
        """Report every week as soon as the next week's dashboard shows up."""
        while not episode_over.wait(WEEK_POLL_S):
            self._post_finished_weeks()

    def _post_finished_weeks(self, final_cash: float | None = None) -> int:
        """Post the unreported weeks whose closing state is known; return how many."""
        with self._ledger_lock:
            self._ledger.observe(self._capture.snapshot())
            finished = self._ledger.finished_weeks(final_cash)
            for week, cash_end, value_end, credit in finished:
                entry = self._ledger.weeks[week]
                start: WeekStart = entry["start"]
                value_start = self._ledger.value(start)
                score = self._scale.scale(credit)
                posted = (
                    post_week_reports(
                        self._client,
                        self._scenario,
                        week=week,
                        day=start.day,
                        cash_start=start.cash,
                        cash_end=cash_end,
                        value_start=value_start,
                        value_end=value_end,
                        credit=credit,
                        score=score,
                        turns=entry["turns"],
                        max_tokens=self._max_tokens,
                    )
                    if self._reports
                    else []
                )
                self._ledger.posted.add(week)
                self._ledger.scores[week] = score
                self.logger.info(
                    "%s week %d (credit %.4f, score %.2f; value %.0f -> %.0f, cash %.0f -> %.0f)"
                    " against %d of %d decision turns (%d turns)",
                    "reported" if self._reports else "recorded",
                    week,
                    credit,
                    score,
                    value_start,
                    value_end,
                    start.cash,
                    cash_end,
                    len(posted),
                    sum(1 for _receipt, _tokens, decision in entry["turns"] if decision is not None),
                    len(entry["turns"]),
                )
        return len(finished)

    def _report_trial_result(self) -> None:
        """Close the last week with the verifier's final cash once Harbor writes result.json."""
        result_path = self.logs_dir.parent / "result.json"
        while not (result_path.exists() and result_path.stat().st_mtime >= self._report_watch_from):
            time.sleep(1.0)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        if rewards.get("final_cash") is None:
            self.logger.warning(
                "trial %s ended without a verifier final cash; last week not reported", result.get("id")
            )
            return
        self._post_finished_weeks(final_cash=float(rewards["final_cash"]))
