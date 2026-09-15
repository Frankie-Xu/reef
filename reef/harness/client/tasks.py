"""Play Harbor task directories with a Harbor agent through Reef, and report each scored episode.

The agent's model calls go through Reef's capture proxy, so every inference is a record on the service
and its receipt comes back here. When the verifier scores the episode, one report goes to
``/reef/report``: the reward as the score, the receipts as the references, and the task's name, path
and digest under ``metadata.task``. A recipe's reported processor turns those records into training
samples the way it does for any report. The agent is the caller's choice: any name Harbor knows
(``terminus-2`` unless told otherwise) or an import path, with the served model and the proxy filled
into the ``{model}``, ``{base_url}`` and ``{api_key}`` placeholders of its configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from reef_client.client import ReefClient, ReefClientError

from reef.core.tasks import HarborTaskError, manifest_task_paths, read_harbor_task
from reef.harness.client.wrapper import CaptureProxy

#: Terminus 2 with its model and endpoint left to the binding; the same placeholders the adapter descriptors use.
DEFAULT_AGENT: Mapping[str, object] = {
    "name": "terminus-2",
    "model_name": "{model}",
    "kwargs": {"api_base": "{base_url}/v1", "llm_kwargs": {"api_key": "{api_key}"}},
}


class TaskPlayError(RuntimeError):
    """A task could not be played, or its episode could not be reported."""


@dataclass(frozen=True)
class EpisodeRow:
    """What one played episode came back with: the verifier's rewards, an error text, and where the trial is."""

    rewards: Mapping[str, float]
    error: str = ""
    trial_uri: str | None = None


class TaskLab(ABC):
    """What plays one Harbor task under one agent: reef-eval's Lab, behind an interface a test can stand in for."""

    @abstractmethod
    async def run(
        self,
        task_path: Path,
        agent: Mapping[str, object],
        *,
        key: str,
        tags: Mapping[str, str],
        overrides: Mapping[str, object],
    ) -> EpisodeRow: ...


class ReefEvalLab(TaskLab):
    """reef-eval's Lab under ``work_dir``: Harbor trials under ``trials/``, results in ``results.sqlite``."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = Path(work_dir)

    async def run(
        self,
        task_path: Path,
        agent: Mapping[str, object],
        *,
        key: str,
        tags: Mapping[str, str],
        overrides: Mapping[str, object],
    ) -> EpisodeRow:
        try:
            from reef_eval import Lab
        except ImportError as exc:
            raise TaskPlayError(
                "playing tasks needs reef-eval: install reef-infra[terminus] on Python 3.12 or later"
            ) from exc
        # A Lab binds its asyncio primitives to the loop of its first use, so every episode gets its own Lab.
        row = await Lab(self.work_dir).run(str(task_path), dict(agent), tags=dict(tags), key=key, **dict(overrides))
        row_tags = dict(row.tags)
        rewards = {str(name): float(value) for name, value in dict(row.rewards).items()}
        return EpisodeRow(rewards=rewards, error=str(row_tags.get("error") or ""), trial_uri=row.uri)


def bound_agent(agent: Mapping[str, object], *, model: str, base_url: str, api_key: str) -> dict[str, object]:
    """The agent configuration with ``{model}``, ``{base_url}`` and ``{api_key}`` filled in wherever they sit."""
    values = {"{model}": model, "{base_url}": base_url, "{api_key}": api_key}

    def fill(value: object) -> object:
        if isinstance(value, str):
            for placeholder, replacement in values.items():
                value = value.replace(placeholder, replacement)
            return value
        if isinstance(value, Mapping):
            return {str(name): fill(item) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [fill(item) for item in value]
        return value

    return {str(name): fill(item) for name, item in agent.items()}


def episode_reward(rewards: Mapping[str, float]) -> float | None:
    """The verifier's reward: the ``reward`` entry, else the first one; None when the verifier wrote nothing finite."""
    if not rewards:
        return None
    value = rewards["reward"] if "reward" in rewards else next(iter(rewards.values()))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def task_identity(task_path: Path) -> dict[str, str]:
    """What names the task in a report: its directory name and path, plus the reef digest when it carries one."""
    identity = {"name": task_path.name, "path": str(task_path)}
    # A task that reef.core.tasks did not write has no digest; the name and the path still identify it.
    with contextlib.suppress(HarborTaskError):
        identity["digest"] = read_harbor_task(task_path).digest
    return identity


@dataclass(frozen=True)
class TaskPlay:
    """One played task: what the verifier said and what reached Reef."""

    task_path: Path
    name: str
    episode_id: str
    reward: float | None
    rewards: Mapping[str, float]
    error: str
    receipts: tuple[str, ...]
    report_agent_record_ids: tuple[str, ...]
    trial_uri: str | None

    @property
    def is_reported(self) -> bool:
        return bool(self.report_agent_record_ids)


class TaskPlayer:
    """Plays task directories with one agent against one Reef scenario, and reports every scored episode."""

    def __init__(
        self,
        *,
        reef_url: str,
        scenario: str,
        model: str,
        work_dir: Path,
        token: str | None = None,
        agent: Mapping[str, object] | None = None,
        environment: str = "docker",
        labels: Mapping[str, str] | None = None,
        extra_instruction_paths: Sequence[Path] = (),
        per_receipt: bool = False,
        lab: TaskLab | None = None,
    ) -> None:
        for label, value in (
            ("reef_url", reef_url),
            ("scenario", scenario),
            ("model", model),
            ("environment", environment),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TaskPlayError(f"{label} must be a non-empty string")
        self.reef_url = reef_url.rstrip("/")
        self.scenario = scenario
        self.model = model
        self.token = token
        self.work_dir = Path(work_dir)
        self.agent: dict[str, object] = dict(agent) if agent is not None else dict(DEFAULT_AGENT)
        if not (self.agent.get("name") or self.agent.get("import_path")):
            raise TaskPlayError("agent must carry a Harbor agent name or an import_path")
        self.environment = environment
        self.labels = {str(name): str(value) for name, value in dict(labels or {}).items()}
        for path in extra_instruction_paths:
            if not Path(path).is_file():
                raise TaskPlayError(f"extra instruction file {path} does not exist")
        self.extra_instruction_paths = tuple(Path(path) for path in extra_instruction_paths)
        self.per_receipt = per_receipt
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.lab: TaskLab = lab if lab is not None else ReefEvalLab(self.work_dir / "lab")
        self.client = ReefClient(self.reef_url, token=token)

    @property
    def agent_name(self) -> str:
        return str(self.agent.get("name") or self.agent.get("import_path"))

    def play(self, task_path: Path) -> TaskPlay:
        """Play one task directory and report its episode when the verifier scored it."""
        return asyncio.run(self.play_async(task_path))

    def play_all(self, task_paths: Iterable[Path]) -> tuple[TaskPlay, ...]:
        """Play task directories one after another; every episode reported before the next starts."""
        return tuple(self.play(path) for path in task_paths)

    async def play_async(self, task_path: Path) -> TaskPlay:
        task_path = Path(task_path)
        if not (task_path / "task.toml").is_file():
            raise TaskPlayError(f"{task_path} is not a Harbor task directory: no task.toml")
        episode_id = uuid.uuid4().hex
        tags = {"task": task_path.name, "episode": episode_id, **self.labels}
        proxy = CaptureProxy(self.reef_url, self.scenario, self.token, tags=tags)
        proxy.start()
        try:
            agent = bound_agent(
                self.agent, model=self.model, base_url=f"http://127.0.0.1:{proxy.port}", api_key=self.token or "reef"
            )
            overrides: dict[str, object] = {"environment": {"type": self.environment}}
            if self.extra_instruction_paths:
                overrides["extra_instruction_paths"] = [str(path) for path in self.extra_instruction_paths]
            row = await self.lab.run(task_path, agent, key=episode_id, tags=tags, overrides=overrides)
        finally:
            turns = proxy.drain()
            proxy.stop()
        receipts = tuple(str(turn["receipt"]) for turn in turns if turn.get("receipt") and turn.get("status") == 200)
        reward = episode_reward(row.rewards)
        report_ids: tuple[str, ...] = ()
        if reward is not None and receipts:
            report_ids = self.report(task_path, episode_id, reward, row, receipts)
        return TaskPlay(
            task_path=task_path,
            name=task_path.name,
            episode_id=episode_id,
            reward=reward,
            rewards=dict(row.rewards),
            error=row.error,
            receipts=receipts,
            report_agent_record_ids=report_ids,
            trial_uri=row.trial_uri,
        )

    def report(
        self, task_path: Path, episode_id: str, reward: float, row: EpisodeRow, receipts: Sequence[str]
    ) -> tuple[str, ...]:
        """Post the episode's reward against its receipts; one report, or one per receipt; the record ids."""
        payload: dict[str, object] = {
            "score": reward,
            "feedback": f"verifier reward {reward} on {task_path.name}",
            "metadata": {
                "task": task_identity(task_path),
                "episode": {
                    "id": episode_id,
                    "agent": self.agent_name,
                    "labels": dict(self.labels),
                    "rewards": dict(row.rewards),
                    "trial_uri": row.trial_uri,
                },
            },
        }
        groups = [[receipt] for receipt in receipts] if self.per_receipt else [list(receipts)]
        record_ids = []
        for references in groups:
            try:
                answer = self.client.report(self.scenario, payload, references=references)
            except ReefClientError as exc:
                raise TaskPlayError(
                    f"the report for {task_path.name} was refused ({exc.status}): {exc.body[:300]}"
                ) from exc
            record_ids.append(str(answer.get("agent_record_id", "")))
        return tuple(record_ids)


def parsed_labels(pairs: Sequence[str]) -> dict[str, str]:
    labels = {}
    for pair in pairs:
        name, separator, value = pair.partition("=")
        if not separator or not name:
            raise TaskPlayError(f"a label is NAME=VALUE, not {pair!r}")
        labels[name] = value
    return labels


def main(argv: Sequence[str] | None = None, *, lab: TaskLab | None = None) -> int:
    """Play task directories, or one side of a split manifest, and print one JSON line per task."""
    parser = argparse.ArgumentParser(
        prog="python -m reef.harness.client.tasks",
        description="Play Harbor task directories through Reef and report each scored episode.",
    )
    parser.add_argument("tasks", nargs="*", type=Path, help="task directories; or --manifest with --tasks-root")
    parser.add_argument("--manifest", type=Path, help="a split manifest written by reef.core.tasks")
    parser.add_argument("--tasks-root", type=Path, help="the directory the manifest's task names live under")
    parser.add_argument("--side", choices=("train", "eval"), default="train", help="which side of the manifest")
    parser.add_argument("--reef-url", required=True, help="the Reef service, e.g. http://127.0.0.1:8900")
    parser.add_argument("--scenario", required=True, help="the scenario the records and reports belong to")
    parser.add_argument("--model", required=True, help="the served model name the agent asks for")
    parser.add_argument("--token", default=os.environ.get("REEF_TOKEN") or None, help="defaults to REEF_TOKEN")
    parser.add_argument("--work-dir", type=Path, default=Path("work/play"), help="trials and results land here")
    parser.add_argument(
        "--agent-json",
        default=None,
        help="Harbor AgentConfig fields as JSON with {model}, {base_url} and {api_key} placeholders; terminus-2 by default",
    )
    parser.add_argument("--environment", default="docker", help="the Harbor environment type")
    parser.add_argument(
        "--label", action="append", default=[], metavar="NAME=VALUE", help="tags every call and report"
    )
    parser.add_argument(
        "--instructions", action="append", type=Path, default=[], help="a file appended to every task's instruction"
    )
    parser.add_argument(
        "--per-receipt", action="store_true", help="one report per model call instead of one per episode"
    )
    arguments = parser.parse_args(argv)
    if arguments.manifest is not None:
        if arguments.tasks or arguments.tasks_root is None:
            parser.error("--manifest goes with --tasks-root and no task directories")
        task_paths: tuple[Path, ...] = manifest_task_paths(arguments.manifest, arguments.tasks_root, arguments.side)
    elif arguments.tasks:
        task_paths = tuple(arguments.tasks)
    else:
        parser.error("name task directories or a --manifest")
    agent = json.loads(arguments.agent_json) if arguments.agent_json else None
    if agent is not None and not isinstance(agent, dict):
        parser.error("--agent-json must hold an object")
    player = TaskPlayer(
        reef_url=arguments.reef_url,
        scenario=arguments.scenario,
        model=arguments.model,
        work_dir=arguments.work_dir,
        token=arguments.token,
        agent=agent,
        environment=arguments.environment,
        labels=parsed_labels(arguments.label),
        extra_instruction_paths=arguments.instructions,
        per_receipt=arguments.per_receipt,
        lab=lab,
    )
    is_complete = True
    for play in player.play_all(task_paths):
        line = {
            "task": play.name,
            "reward": play.reward,
            "receipts": len(play.receipts),
            "reports": list(play.report_agent_record_ids),
            "error": play.error,
        }
        print(json.dumps(line))
        is_complete = is_complete and play.is_reported
    return 0 if is_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
