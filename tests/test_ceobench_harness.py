"""Host-side contracts of the CEO-Bench example (recipes/sao/examples/ceobench)."""

from __future__ import annotations

import asyncio
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.test_example_entrypoints import EXAMPLE_DIRS, _load_harness

EXAMPLE_DIR = EXAMPLE_DIRS["ceobench"]
SCORE_PATH = EXAMPLE_DIR / "harbor" / "tests" / "score.py"


def _load_score_module():
    namespace = runpy.run_path(str(SCORE_PATH), run_name="score")
    return SimpleNamespace(**namespace)


class _Environment:
    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.commands: list[tuple[str, dict]] = []
        self.downloads: list[tuple[str, Path]] = []

    async def exec(self, command, env=None):
        self.commands.append((command, dict(env or {})))
        return SimpleNamespace(return_code=self.return_code, stdout="", stderr="")

    async def download_dir(self, source_dir, target_dir):
        self.downloads.append((source_dir, Path(target_dir)))


class _Sidecar:
    server_address = ("0.0.0.0", 29123)

    def __init__(self) -> None:
        self.stopped = False

    def shutdown(self) -> None:
        self.stopped = True


class _Capture:
    def __init__(self, turns: list[dict]) -> None:
        self._turns = turns

    def snapshot(self):
        return list(self._turns)


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def report(self, scenario, payload, *, recipe=None):
        self.calls.append((scenario, payload))
        return {"accepted": True}


def _dashboard(week: int, day: int, cash: int, *, subscribers: int = 0, seats: int = 0, prices=(0, 0, 0)) -> str:
    a, b, c = prices
    return (
        f"=== Week {week} Dashboard (Day {day}) ===\n\nCash: ${cash:,}\n"
        f"Individual Subscribers: {subscribers}\nEnterprise Subscribed Seats: {seats}\nOpen Issues: 0\n\n"
        f"--- Current Config ---\nPrices: A=${a}, B=${b}, C=${c}\nModel Tiers: A=1, B=1, C=1\n"
    )


def _start(agent_module, week: int, day: int, cash: float, *, subscribers=0, seats=0, prices=(0.0, 0.0, 0.0)):
    return agent_module.WeekStart(week, day, cash, subscribers, seats, prices)


def _turn(receipt: str, dashboard: str | None, prompt_tokens: int, completion_tokens: int) -> dict:
    messages = [{"role": "system", "content": "You are the CEO."}]
    if dashboard is not None:
        messages.append({"role": "user", "content": dashboard})
    return {
        "status": 200,
        "receipt": receipt,
        "request": {"messages": messages},
        "response": {"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}},
    }


def _agent(agent_module, monkeypatch, turns: list[dict], *, service_url="http://10.0.0.7:28900"):
    import threading

    agent = object.__new__(agent_module.HarborAgent)
    agent.model_name = "reef"
    agent.logs_dir = Path("/tmp/trial/agent")
    agent.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    agent._service_url = service_url
    agent._scenario = "ceobench-host-test"
    agent._seed = 7
    agent._days = 14
    agent._client = _Client()
    agent._capture = _Capture([])
    # One week of credit and a scale that leaves small credits as they are,
    # so the scores below read as the value changes they come from.
    agent._ledger = agent_module.WeekLedger(total_weeks=2, credit_weeks=1)
    agent._scale = agent_module.ScoreScale(clip=10.0, floor=1.0)
    agent._ledger_lock = threading.Lock()
    agent._max_tokens = 0
    sidecar = _Sidecar()

    def start_sidecar():
        agent._capture = _Capture(turns)
        return sidecar

    monkeypatch.setattr(agent, "_start_sidecar", start_sidecar)
    monkeypatch.setattr(agent_module, "WEEK_POLL_S", 0.01)
    return agent, sidecar


@pytest.mark.unit
def test_runner_command_points_the_benchmark_agent_at_the_sidecar(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")

    command = agent_module.runner_command("http://10.0.0.7:29123/v1", "reef", 42, 500)

    assert command.startswith(f"mkdir -p {agent_module.RUNS_DIR} && cd {agent_module.CEOBENCH_DIR} && uv run")
    assert "saas_bench.agents.bash_agent.run_test" in command
    for flag in (
        "--provider openai",
        "--base-url http://10.0.0.7:29123/v1",
        "--model reef",
        "--seed 42",
        "--days 500",
    ):
        assert flag in command
    assert command.endswith(f"--workspace {agent_module.RUNS_DIR} > {agent_module.RUNS_DIR}/runner.log 2>&1")


@pytest.mark.unit
def test_forwarded_environment_keeps_simulator_settings_out_of_the_repository(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")

    forwarded = agent_module.forwarded_environment(
        {
            "SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER": "openai",
            "OPENAI_BASE_URL": "http://sim:30100/v1",
            "ANTHROPIC_API_KEY": "secret",
            "AWS_REGION": "us-east-2",
            "HOME": "/home/x",
            "REEF_TOKEN": "reef-local",
        }
    )

    assert forwarded == {
        "SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER": "openai",
        "OPENAI_BASE_URL": "http://sim:30100/v1",
        "ANTHROPIC_API_KEY": "secret",
        "AWS_REGION": "us-east-2",
        "SAAS_BENCH_OPENAI_CHAT_COMPLETIONS": "1",
    }


@pytest.mark.unit
def test_turn_week_parses_the_latest_dashboard(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")

    fresh = _turn("r", _dashboard(0, 0, 1_000_000), 1, 1)
    assert agent_module.turn_week(fresh) == _start(agent_module, 0, 0, 1_000_000.0)

    # A transcript that still holds an earlier week's dashboard resolves to the latest one.
    carried = _turn("r", _dashboard(0, 0, 1_000_000), 1, 1)
    carried["request"]["messages"].append(
        {"role": "tool", "content": "ok\n" + _dashboard(1, 7, 982_311, subscribers=3, prices=(10, 39, 99))}
    )
    assert agent_module.turn_week(carried) == _start(
        agent_module, 1, 7, 982_311.0, subscribers=3, prices=(10.0, 39.0, 99.0)
    )

    # The agent's own scripts print price lines; only the dashboard's configuration block counts.
    scripted = _turn("r", _dashboard(0, 0, 1_000_000), 1, 1)
    scripted["request"]["messages"].append(
        {"role": "tool", "content": "=== Setup ===\nPrices: A=$15, B=$49, C=$149\n"}
    )
    assert agent_module.turn_week(scripted) == _start(agent_module, 0, 0, 1_000_000.0)

    # Both spellings of a negative balance; a dashboard cut before its configuration block lists no prices.
    for broke in ("Cash: -$1,234", "Cash: $-1,234"):
        header = f"=== Week 9 Dashboard (Day 63) ===\n\n{broke}\nIndividual Subscribers: 0\nEnterprise Subscribed Seats: 0\n"
        assert agent_module.turn_week(_turn("r", header, 1, 1)) == _start(agent_module, 9, 63, -1234.0)

    assert agent_module.turn_week({"request": {"messages": [{"role": "user", "content": "hello"}]}}) is None


@pytest.mark.unit
def test_week_start_run_rate_uses_the_lowest_listed_price_and_plan_c_for_seats(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")

    assert _start(agent_module, 1, 7, 0.0, subscribers=3, seats=10, prices=(10.0, 39.0, 99.0)).run_rate == 1020.0
    assert _start(agent_module, 1, 7, 0.0, subscribers=2, prices=(0.0, 49.0, 99.0)).run_rate == 98.0
    assert _start(agent_module, 1, 7, 0.0, subscribers=2, seats=4).run_rate == 0.0


@pytest.mark.unit
def test_valuation_counts_the_run_rate_over_the_remaining_horizon(monkeypatch) -> None:
    _, report_module = _load_harness(monkeypatch, "ceobench")

    assert report_module.valuation(100_000.0, 3_000.0, 40, 26) == 118_200.0
    assert report_module.valuation(100_000.0, 3_000.0, 5, 26) == 103_500.0
    assert report_module.valuation(100_000.0, 3_000.0, 0, 26) == 100_000.0
    assert report_module.valuation(100_000.0, 3_000.0, -1, 26) == 100_000.0
    assert report_module.week_credit([-10_000.0, 5_000.0, 2_500.0], 0.5) == (-10_000 + 2_500 + 625) / 1_000_000
    assert report_module.week_credit([], 0.5) == 0.0


@pytest.mark.unit
def test_score_scale_clips_outliers_and_divides_by_the_running_median(monkeypatch) -> None:
    _, report_module = _load_harness(monkeypatch, "ceobench")
    scale = report_module.ScoreScale(clip=0.05, floor=0.003)

    assert scale.scale(-0.34) == -1.0  # a six-figure purchase clips to the limit and sets the first scale
    assert scale.scale(0.02) == 0.02 / ((0.02 + 0.05) / 2)
    assert scale.scale(-0.001) == -0.001 / 0.02  # the median of 0.001, 0.02, 0.05

    quiet = report_module.ScoreScale(clip=0.05, floor=0.003)
    assert quiet.scale(0.0001) == 0.0001 / 0.003  # the floor keeps a quiet start from amplifying noise
    quiet.scale(0.0001)
    assert quiet.scale(0.5) == 3.0  # clipped to 0.05, then capped against the tiny running scale

    with pytest.raises(ValueError, match="positive"):
        report_module.ScoreScale(clip=0.0)


@pytest.mark.unit
def test_week_credit_spans_the_following_weeks_and_is_cut_short_at_the_end(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    ledger = agent_module.WeekLedger(total_weeks=4, credit_weeks=2, discount=0.5)
    turns = [_turn("r-1", _dashboard(0, 0, 1_000_000), 1, 1), _turn("r-2", _dashboard(1, 7, 990_000), 1, 1)]
    ledger.observe(turns)
    # Week 0 waits until two weeks have opened after it.
    assert ledger.finished_weeks() == []

    turns.append(_turn("r-3", _dashboard(2, 14, 985_000), 1, 1))
    ledger.observe(turns)
    # Week 0 closes on week 1's opening state and is credited with its own
    # change and half of week 1's.
    assert ledger.finished_weeks() == [(0, 990_000.0, 990_000.0, (-10_000 - 2_500) / 1_000_000)]
    # The episode ends: week 1's window is cut short at the final cash, and
    # week 2 closes on the final cash alone.
    assert ledger.finished_weeks(final_cash=980_000.0) == [
        (0, 990_000.0, 990_000.0, (-10_000 - 2_500) / 1_000_000),
        (1, 985_000.0, 985_000.0, (-5_000 - 2_500) / 1_000_000),
        (2, 980_000.0, 980_000.0, -5_000 / 1_000_000),
    ]
    with pytest.raises(ValueError, match="credit_weeks"):
        agent_module.WeekLedger(total_weeks=4, credit_weeks=0)


@pytest.mark.unit
def test_week_ledger_groups_turns_and_closes_weeks_with_the_next_dashboard(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    ledger = agent_module.WeekLedger(total_weeks=2, credit_weeks=1)
    ledger.observe(
        [
            _turn("r-1", _dashboard(0, 0, 1_000_000), 10, 3),
            {"status": 500, "receipt": None, "response": {"error": {"message": "engine restarting"}}},
            _turn("r-2", _dashboard(0, 0, 1_000_000), 20, 5),
            _turn("r-3", _dashboard(1, 7, 982_311, subscribers=30, prices=(10, 39, 99)), 30, 7),
            _turn("r-4", None, 40, 9),  # no dashboard of its own: still week 1
        ]
    )

    assert [(turn["receipt"], turn["tokens"], turn["week"]) for turn in ledger.turns] == [
        ("r-1", 13, 0),
        ("r-2", 25, 0),
        ("r-3", 37, 1),
        ("r-4", 49, 1),
    ]
    assert ledger.weeks[0]["turns"] == [("r-1", 13), ("r-2", 25)]
    week_1 = _start(agent_module, 1, 7, 982_311.0, subscribers=30, prices=(10.0, 39.0, 99.0))
    assert ledger.weeks[1] == {"start": week_1, "turns": [("r-3", 37), ("r-4", 49)]}
    # Week 1 opens worth its cash plus the one week left of 30 subscribers at the $10 plan.
    assert ledger.value(week_1) == 982_381.0
    # Week 0 closed when week 1's dashboard appeared; week 1 waits for the final cash, valued as cash.
    credit_0, credit_1 = (982_381 - 1_000_000) / 1_000_000, (793_047 - 982_381) / 1_000_000
    assert ledger.finished_weeks() == [(0, 982_311.0, 982_381.0, credit_0)]
    assert ledger.finished_weeks(final_cash=793_047.0) == [
        (0, 982_311.0, 982_381.0, credit_0),
        (1, 793_047.0, 793_047.0, credit_1),
    ]
    ledger.posted.add(0)
    ledger.scores[0] = -1.0
    assert ledger.finished_weeks() == []
    assert ledger.summary(final_cash=793_047.0) == [
        {
            "week": 0,
            "day": 0,
            "cash_start": 1_000_000.0,
            "cash_end": 982_311.0,
            "subscribers": 0,
            "seats": 0,
            "run_rate": 0.0,
            "value_start": 1_000_000.0,
            "value_end": 982_381.0,
            "credit": credit_0,
            "score": -1.0,
            "turns": 2,
            "reported": True,
        },
        {
            "week": 1,
            "day": 7,
            "cash_start": 982_311.0,
            "cash_end": 793_047.0,
            "subscribers": 30,
            "seats": 0,
            "run_rate": 300.0,
            "value_start": 982_381.0,
            "value_end": 793_047.0,
            "credit": credit_1,
            "score": None,
            "turns": 2,
            "reported": False,
        },
    ]


@pytest.mark.unit
def test_harness_reports_a_week_as_soon_as_the_next_one_starts(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    monkeypatch.setenv("SAAS_BENCH_ENTERPRISE_LLM_PROVIDER", "openai")
    turns = [
        _turn("r-1", _dashboard(0, 0, 1_000_000), 10, 3),
        {"status": 500, "receipt": None, "response": {"error": {"message": "engine restarting"}}},
        _turn("r-2", _dashboard(0, 0, 1_000_000), 20, 5),
        _turn("r-3", _dashboard(1, 7, 982_311, subscribers=3, prices=(10, 39, 99)), 30, 7),
    ]
    agent, sidecar = _agent(agent_module, monkeypatch, turns)
    environment = _Environment()
    context = SimpleNamespace(metadata={"prior": True}, n_input_tokens=0, n_output_tokens=0)

    asyncio.run(agent.run("play", environment, context))

    (command, env), *rest = environment.commands
    assert not rest
    assert "--base-url http://10.0.0.7:29123/v1" in command and "--seed 7 --days 14" in command
    assert env["SAAS_BENCH_OPENAI_CHAT_COMPLETIONS"] == "1"
    assert env["SAAS_BENCH_ENTERPRISE_LLM_PROVIDER"] == "openai"
    assert sidecar.stopped
    assert environment.downloads == [(agent_module.RUNS_DIR, Path("/tmp/trial/agent/ceobench"))]

    # Week 0 was reported (its two turns) once week 1's dashboard appeared; week 1 waits.
    # Week 1 opens worth its cash plus one week of three $10 subscribers: the score follows the value.
    payloads = [payload for _, payload in agent._client.calls]
    assert [payload["references"] for payload in payloads] == [["r-1"], ["r-2"]]
    credit = (982_318 - 1_000_000) / 1_000_000
    assert {payload["score"] for payload in payloads} == {credit}
    assert payloads[1]["metadata"]["ceobench"] == {
        "week": 0,
        "day": 0,
        "cash_start": 1_000_000.0,
        "cash_end": 982_311.0,
        "value_start": 1_000_000.0,
        "value_end": 982_318.0,
        "credit": credit,
        "turn": 1,
        "turns": 2,
    }
    assert len({payload["agent_record_id"] for payload in payloads}) == 2

    assert context.metadata["reef"] == {
        "agent_record_ids": ["r-1", "r-2", "r-3"],
        "agent_record_tokens": [13, 25, 37],
        "agent_record_weeks": [0, 0, 1],
    }
    assert context.metadata["ceobench"] == {
        "seed": 7,
        "days": 14,
        "horizon_weeks": 26,
        "credit_weeks": 1,
        "discount": 0.8,
        "score_clip": 10.0,
        "score_floor": 1.0,
        "turns": 4,
        "exit_code": 0,
        "weeks": [
            {
                "week": 0,
                "day": 0,
                "cash_start": 1_000_000.0,
                "cash_end": 982_311.0,
                "subscribers": 0,
                "seats": 0,
                "run_rate": 0.0,
                "value_start": 1_000_000.0,
                "value_end": 982_318.0,
                "credit": credit,
                "score": credit,
                "turns": 2,
                "reported": True,
            },
            {
                "week": 1,
                "day": 7,
                "cash_start": 982_311.0,
                "cash_end": None,
                "subscribers": 3,
                "seats": 0,
                "run_rate": 30.0,
                "value_start": 982_318.0,
                "value_end": None,
                "credit": None,
                "score": None,
                "turns": 1,
                "reported": False,
            },
        ],
    }
    assert context.metadata["prior"] is True
    assert (context.n_input_tokens, context.n_output_tokens) == (60, 15)


@pytest.mark.unit
def test_last_week_closes_with_the_verifier_final_cash(monkeypatch, tmp_path) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    turns = [
        _turn("r-1", _dashboard(0, 0, 1_000_000), 10, 3),
        _turn("r-2", _dashboard(1, 7, 982_311, subscribers=3, prices=(10, 39, 99)), 30, 7),
    ]
    agent, _ = _agent(agent_module, monkeypatch, turns)
    agent.logs_dir = tmp_path / "agent"
    agent._report_watch_from = 0.0
    asyncio.run(agent.run("play", _Environment(), SimpleNamespace(metadata=None, n_input_tokens=0, n_output_tokens=0)))
    assert [payload["references"] for _, payload in agent._client.calls] == [["r-1"]]

    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-9",
                "verifier_result": {"rewards": {"reward": 0.793, "final_cash": 793_047.0, "survival_days": 14}},
            }
        ),
        encoding="utf-8",
    )
    agent._report_trial_result()

    scenario, payload = agent._client.calls[-1]
    assert scenario == "ceobench-host-test"
    assert payload["references"] == ["r-2"]
    # The last week is valued as cash alone at both ends of the comparison's close.
    assert payload["score"] == payload["metadata"]["ceobench"]["credit"] == (793_047 - 982_318) / 1_000_000
    assert payload["metadata"]["ceobench"]["week"] == 1 and payload["metadata"]["ceobench"]["cash_end"] == 793_047.0
    assert payload["metadata"]["ceobench"]["value_end"] == 793_047.0
    assert "week 1" in payload["feedback"]
    # A second look changes nothing: the week is already posted.
    agent._report_trial_result()
    assert len(agent._client.calls) == 2


@pytest.mark.unit
def test_training_pacer_waits_for_the_batches_the_reported_turns_filled(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    monkeypatch.setattr(agent_module, "PACE_POLL_S", 0.0)
    releases = iter([3, 3, 4, 5])  # 3 at episode start, then the trainer catches up
    logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    pacer = agent_module.TrainingPacer(16, timeout_s=60, count_releases=lambda: next(releases), logger=logger)

    # 35 reported turns fill two batches of 16; the third release beyond the
    # baseline is not required until a third batch fills.
    assert pacer.expected_releases(35) == 2
    pacer.wait(2, 35)
    assert pacer._base == 3


@pytest.mark.unit
def test_training_pacer_forgives_a_batch_the_trainer_never_commits(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    monkeypatch.setattr(agent_module, "PACE_POLL_S", 0.0)
    warnings = []
    logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: warnings.append(a))
    pacer = agent_module.TrainingPacer(16, timeout_s=0.0, count_releases=lambda: 0, logger=logger)

    pacer.wait(3, 32)  # two batches expected, none committed: times out at once

    assert len(warnings) == 1
    # The shortfall is forgiven: the next week only waits for new batches.
    assert pacer.expected_releases(32) == 0
    assert pacer.expected_releases(48) == 1


@pytest.mark.unit
def test_gate_closes_the_previous_week_before_serving_the_next(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    turns = [
        _turn("r-1", _dashboard(0, 0, 1_000_000), 10, 3),
        _turn("r-2", _dashboard(0, 0, 1_000_000), 20, 5),
    ]
    agent, _sidecar = _agent(agent_module, monkeypatch, turns)
    agent._capture = _Capture(turns)  # the sidecar is already serving
    waits = []

    class Pacer:
        def wait(self, week, posted_turns):
            waits.append((week, posted_turns))
            return 0.0

    agent._pacer = Pacer()
    agent._gated_week = None

    # The first request of week 1 arrives: week 0 closes with week 1's opening state and is reported.
    agent._gate_week(_start(agent_module, 1, 7, 982_311.0))

    assert [payload["references"] for _, payload in agent._client.calls] == [["r-1"], ["r-2"]]
    reported = agent._client.calls[0][1]["metadata"]["ceobench"]
    assert (reported["cash_end"], reported["value_end"]) == (982_311.0, 982_311.0)
    assert waits == [(1, 2)]
    # Later requests of the same week pass without another wait.
    agent._gate_week(_start(agent_module, 1, 7, 982_311.0))
    assert waits == [(1, 2)]


@pytest.mark.unit
def test_paced_handler_peeks_at_the_week_and_replays_the_body(monkeypatch) -> None:
    import io

    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    seen = []
    forwarded = []

    class Base:
        def _forward(self, forward_path, routed_session):
            forwarded.append((forward_path, routed_session, self.rfile.read()))

    handler_cls = agent_module.paced_handler(Base, seen.append)
    body = json.dumps({"messages": [{"role": "user", "content": _dashboard(4, 28, 940_968)}]}).encode()
    handler = object.__new__(handler_cls)
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)

    handler._forward("/v1/chat/completions", None)

    assert seen == [_start(agent_module, 4, 28, 940_968.0)]
    assert forwarded == [("/v1/chat/completions", None, body)]


@pytest.mark.unit
def test_harness_fails_the_trial_when_the_runner_exits_nonzero(monkeypatch) -> None:
    agent_module, _ = _load_harness(monkeypatch, "ceobench")
    agent, sidecar = _agent(agent_module, monkeypatch, [{"status": 200, "receipt": "r-1", "response": {}}])
    environment = _Environment(return_code=3)
    context = SimpleNamespace(metadata=None, n_input_tokens=0, n_output_tokens=0)

    with pytest.raises(RuntimeError, match="exited 3"):
        asyncio.run(agent.run("play", environment, context))

    # The receipts and the run directory are kept even for a failed episode.
    assert sidecar.stopped
    assert context.metadata["reef"] == {
        "agent_record_ids": ["r-1"],
        "agent_record_tokens": [0],
        "agent_record_weeks": [None],
    }
    assert context.metadata["ceobench"]["weeks"] == []
    assert environment.downloads


@pytest.mark.unit
def test_turns_longer_than_the_training_window_are_not_reported(monkeypatch) -> None:
    _, report_module = _load_harness(monkeypatch, "ceobench")
    client = _Client()

    posted = report_module.post_week_reports(
        client,
        "ceobench-host-test",
        week=3,
        day=21,
        cash_start=900_000.0,
        cash_end=905_000.0,
        value_start=900_000.0,
        value_end=910_000.0,
        credit=0.0125,
        score=1.25,
        turns=[("r-1", 9000), ("r-2", 50000), ("r-3", 48000)],
        max_tokens=49152,
    )

    # The 50k-token turn was served and recorded but cannot be trained on.
    assert posted == [{"accepted": True}] * 2
    payloads = [payload for _, payload in client.calls]
    assert [payload["references"] for payload in payloads] == [["r-1"], ["r-3"]]
    assert [payload["metadata"]["ceobench"]["turn"] for payload in payloads] == [0, 2]
    # The scaled credit is the score; the credit and the values travel along.
    assert {payload["score"] for payload in payloads} == {1.25}
    assert {payload["metadata"]["ceobench"]["credit"] for payload in payloads} == {0.0125}
    feedback = payloads[0]["feedback"]
    assert "week 3" in feedback and "credit 0.0125, score 1.25" in feedback
    assert "value 900000 -> 910000" in feedback and "cash 900000 -> 905000" in feedback
    # No limit reports every turn.
    client.calls.clear()
    report_module.post_week_reports(
        client,
        "ceobench-host-test",
        week=3,
        day=21,
        cash_start=900_000.0,
        cash_end=905_000.0,
        value_start=900_000.0,
        value_end=910_000.0,
        credit=0.0125,
        score=1.25,
        turns=[("r-1", 1), ("r-2", 2), ("r-3", 3)],
    )
    assert len(client.calls) == 3


@pytest.mark.unit
def test_scorer_locates_the_single_run_and_prefers_the_checkpointed_database(tmp_path) -> None:
    score = _load_score_module()
    runs = tmp_path / "runs"
    run_dir = runs / "run_abc123"
    live = run_dir / "agent_workspace" / "sessions" / "s1"
    live.mkdir(parents=True)
    (live / "world.nmdb").write_bytes(b"live")

    assert score.find_run_dir(runs) == run_dir
    assert score.find_world_db(run_dir) == live / "world.nmdb"
    (run_dir / "world.nmdb").write_bytes(b"checkpointed")
    assert score.find_world_db(run_dir) == run_dir / "world.nmdb"

    (runs / "run_def456").mkdir()
    with pytest.raises(RuntimeError, match="expected one run"):
        score.find_run_dir(runs)
    with pytest.raises(FileNotFoundError):
        score.find_run_dir(tmp_path / "empty")


@pytest.mark.unit
def test_entrypoint_runs_one_episode_and_drains_training(monkeypatch, capsys) -> None:
    calls = []

    class Lab:
        def __init__(self, path):
            self.path = Path(path)

        async def run(self, task, agent, tags=None):
            calls.append((self.path, Path(task), agent, tags))
            return SimpleNamespace(rewards={"reward": 0.9, "final_cash": 900000.0}, tags={}, uri="file:///trial")

    reef_eval = ModuleType("reef_eval")
    reef_eval.Lab = Lab
    monkeypatch.setitem(sys.modules, "reef_eval", reef_eval)
    for key, value in {
        "REEF_SERVICE_URL": "http://127.0.0.1:1/",  # nothing listens: the drain skips
        "REEF_SCENARIO": "ceobench-host-test",
        "REEF_TOKEN": "reef-local",
        "CEOBENCH_SEED": "43",
        "CEOBENCH_DAYS": "14",
    }.items():
        monkeypatch.setenv(key, value)

    runpy.run_path(str(EXAMPLE_DIR / "run.py"))

    # One episode: the policy adapts inside the episode it is scored on.
    (lab, task, agent, tags), *rest = calls
    assert not rest
    assert str(task.relative_to(EXAMPLE_DIR)) == "harbor"
    assert agent == {"name": "harness:HarborAgent", "model_name": "reef", "kwargs": {"seed": 43, "days": 14}}
    assert tags == {"seed": 43, "days": 14}
    assert lab == EXAMPLE_DIR / "work" / "lab"
    out = capsys.readouterr().out
    assert "reward" in out and "not reachable" in out


@pytest.mark.unit
def test_entrypoint_fails_when_harbor_reports_an_error(monkeypatch) -> None:
    class Lab:
        def __init__(self, _path):
            pass

        async def run(self, _task, _agent, tags=None):
            return SimpleNamespace(rewards={}, tags={"error": "environment failed"}, uri="file:///failed-trial")

    reef_eval = ModuleType("reef_eval")
    reef_eval.Lab = Lab
    monkeypatch.setitem(sys.modules, "reef_eval", reef_eval)
    monkeypatch.setenv("REEF_SERVICE_URL", "http://127.0.0.1:1/")

    with pytest.raises(RuntimeError, match="Harbor trial failed: environment failed"):
        runpy.run_path(str(EXAMPLE_DIR / "run.py"))


@pytest.mark.unit
def test_task_pins_the_upstream_commit_and_ships_no_credentials() -> None:
    dockerfile = (EXAMPLE_DIR / "harbor" / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG CEOBENCH_COMMIT=d2b7b32e5301a571b77f5f68bd1032adbcd5b464" in dockerfile
    patch = (EXAMPLE_DIR / "harbor" / "environment" / "reef.patch").read_text(encoding="utf-8")
    assert "SAAS_BENCH_OPENAI_CHAT_COMPLETIONS" in patch and "SAAS_BENCH_" in patch
    for path in EXAMPLE_DIR.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".toml", ".md", ".patch"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            assert "sk-ant-" not in text and "AKIA" not in text, path
    serve = (EXAMPLE_DIR / "serve.yaml").read_text(encoding="utf-8")
    assert "batch-size: 1" in serve and "recipes.sao.recipe:SAORecipe" in serve
    assert json.loads(json.dumps({"ok": True}))  # keeps json imported for the reward fixture above
