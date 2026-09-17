"""Figure 3 of arXiv:2601.19897 through reef-eval: one model learns Tool Use, then Science Q&A.

An arm (``sdft`` or ``sft``) is a stream of two Harbor tasks, ``harbor/tooluse``
then ``harbor/science``, run under ``harness:HarborAgent``. Each stage is a
fresh Reef stack started from the previous stage's exported weights, the way
the reference implementation chains its single-task runs; the stack's
recipe is the arm's (``serve.yaml``: the sdft recipe; ``serve-sft.yaml``: the
sft recipe), its learning-rate schedule spans exactly the stage's steps.
Each arm's stack takes four GPUs (the actor with the rollout engines
colocated) and its own host port, so ``--arm both`` runs the two arms side
by side on an eight-GPU host, each in its own process.

For each stage:

    start   — ``docker compose up`` with the arm's config, the stage's
              starting weights and its step count
    run     — ``lab.run`` builds the task's images, starts the judge, and
              runs the stage in the task container; the judge scores the
              served model on both skills every few steps
    record  — the verifier's final reward (this task's accuracy after the
              last step, the other skill's beside it) and every judge score
              land in the Lab store tagged ``stream``, ``arm``, ``position``
    stop    — ``docker compose down``; the stage's HF export seeds the next

``plot.py`` draws the figure from the store. Rows already recorded are
skipped, so a crashed stream resumes; a new ``--stream`` name starts over.

    uv run --no-project --python 3.12 --with "reef-eval[harbor]" --with reef-client \\
        --with-editable . run.py --arm both
    SKILLS_STEPS=2 ... run.py --arm sdft --stream smoke      # two steps per stage
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from reef_eval import Lab

HERE = Path(__file__).resolve().parent
HARBOR = HERE / "harbor"
STAGES = ("tooluse", "science")
#: The reference's training splits: the stage's step count follows from them.
TRAINING_PROMPTS = {"tooluse": 4046, "science": 2674}
ARM_CONFIGS = {"sdft": "serve.yaml", "sft": "serve-sft.yaml"}
#: Each arm's GPUs (SKILLS_GPUS_<ARM> overrides, four comma-separated ids) and the host port its Reef publishes.
ARM_GPUS = {"sdft": "0,1,2,3", "sft": "4,5,6,7"}
ARM_PORTS = {"sdft": 28902, "sft": 28903}
#: The serve configs' lr-warmup-iters; Megatron requires the decay span to be longer.
LR_WARMUP_STEPS = 10
AGENT = {"name": "harness:HarborAgent", "model_name": "reef"}
#: The stack's container name prefix, as docker-compose.yaml fixes it; the arm's name follows.
CONTAINER = "reef-sdft-skills"
#: Container paths: the base model under the models mount, a stage's export under the run mount.
BASE_MODEL = "/root/models/Qwen2.5-7B-Instruct"
EXPERIMENT_MOUNT = "/var/lib/experiment"

RUN_DIR = Path(os.environ.get("RUN_DIR", HERE / "work")).resolve()


def stage_steps(task: str) -> int:
    """Optimizer steps the stage runs: the runner's schedule over the split, capped like the runner."""
    epochs = int(os.environ.get("SKILLS_EPOCHS", "2"))
    prompts_per_step = int(os.environ.get("SKILLS_PROMPTS_PER_STEP", "32"))
    cap = int(os.environ.get("SKILLS_STEPS", "0"))
    steps = epochs * TRAINING_PROMPTS[task] // prompts_per_step
    return min(steps, cap) if cap else steps


def latest_export(stage_dir: Path) -> Path:
    """The stage's last HF export (the bridge names them by training step)."""
    exports = [path for path in (stage_dir / "checkpoints" / "hf").glob("*") if path.name.isdigit()]
    if not exports:
        raise FileNotFoundError(f"no HF export under {stage_dir}/checkpoints/hf; the stage did not train")
    return max(exports, key=lambda path: int(path.name))


def arm_environment(arm: str) -> dict[str, str]:
    """What the arm's stack and its tasks read from the environment: its GPUs and its host port."""
    gpus = os.environ.get(f"SKILLS_GPUS_{arm.upper()}", ARM_GPUS[arm]).split(",")
    if len(gpus) != 4:
        raise ValueError(f"SKILLS_GPUS_{arm.upper()} must name four GPUs, got {gpus}")
    return {
        "ARM": arm,
        "REEF_HOST_PORT": str(ARM_PORTS[arm]),
        **{f"REEF_GPU_{index}": gpu.strip() for index, gpu in enumerate(gpus)},
    }


def compose(arm: str, *arguments: str, environment: dict[str, str]) -> None:
    subprocess.run(
        ["docker", "compose", "-p", f"skills-{arm}", "-f", str(HERE / "docker-compose.yaml"), *arguments],
        check=True,
        env={**os.environ, **arm_environment(arm), **environment},
    )


def start_stack(arm: str, model_path: str, steps: int, stage_dir: Path) -> None:
    """Start the arm's Reef stack on ``model_path`` with a schedule over ``steps``; state goes to ``stage_dir``.

    A smoke run's few steps still get a schedule longer than the warmup, as
    Megatron insists; its learning rate then never leaves the warmup ramp.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "SERVE_CONFIG": ARM_CONFIGS[arm],
        "SDFT_MODEL_PATH": model_path,
        "SDFT_LR_DECAY_ITERS": str(max(steps, LR_WARMUP_STEPS + 1)),
        # The stage checkpoints once, at its last step; every step's weights still reach the engines.
        "SDFT_CHECKPOINT_INTERVAL": str(steps),
        "RUN_DIR": str(stage_dir),
        "EXPERIMENT_DIR": str(RUN_DIR),
    }
    print(f"==> the {arm} stack on {model_path}, {steps} steps, state in {stage_dir}", flush=True)
    compose(arm, "up", "-d", "--wait", environment=environment)


def stop_stack(arm: str, stage_dir: Path) -> None:
    compose(arm, "down", "--timeout", "120", environment={"RUN_DIR": str(stage_dir), "EXPERIMENT_DIR": str(RUN_DIR)})
    # A stack mid-step can outlive compose's grace; the next stage needs its GPUs.
    subprocess.run(["docker", "rm", "-f", f"{CONTAINER}-{arm}"], check=False, capture_output=True)


async def run_arm(arm: str, stream: str, seed: int) -> None:
    lab = Lab(RUN_DIR / "lab")
    # The task containers read the arm's host port from this process's environment.
    os.environ.update(arm_environment(arm))
    model_path = BASE_MODEL
    for position, task in enumerate(STAGES):
        stage_dir = RUN_DIR / stream / arm / task
        tags = {"stream": stream, "arm": arm, "position": position, "seed": seed}
        # The scenario names the stage; the task's compose file hands it to both containers.
        os.environ["REEF_SCENARIO"] = f"{stream}-{arm}-{task}"
        os.environ["SKILLS_SEED"] = str(seed)
        key = f"{stream}@{position:03d}:{arm}:{task}:seed{seed}"
        row = lab.store.get(key)
        if row is None:
            start_stack(arm, model_path, stage_steps(task), stage_dir)
            try:
                row = await lab.run(str(HARBOR / task), AGENT, tags=tags, key=key)
            finally:
                stop_stack(arm, stage_dir)
        else:
            print(f"[{stream}/{arm} {position}] {task}: recorded already, reusing its export", flush=True)
        if row.tags.get("error") or "reward" not in row.rewards:
            raise RuntimeError(
                f"stage {position} ({task}) of {stream}/{arm} failed: {row.tags.get('error') or 'no reward'}; "
                f"its row is recorded under {lab.root}, so rerun with a new --stream name"
            )
        print(f"[{stream}/{arm} {position}] {task}: {json.dumps(row.rewards)}", flush=True)
        export = latest_export(stage_dir)
        model_path = f"{EXPERIMENT_MOUNT}/{export.relative_to(RUN_DIR)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=[*sorted(ARM_CONFIGS), "both"], required=True)
    parser.add_argument("--stream", default="figure3", help="the stream's name in the Lab store (a new name reruns)")
    parser.add_argument("--seed", type=int, default=42, help="the training order's seed")
    arguments = parser.parse_args()
    for variable in ("REEF_TOKEN", "REEF_IMAGE", "MODEL_DIR", "REEF_ROOT"):
        if not os.environ.get(variable):
            sys.exit(f"run.py: {variable} is not set; run.sh sets it")
    if arguments.arm != "both":
        asyncio.run(run_arm(arguments.arm, arguments.stream, arguments.seed))
        return
    # One process per arm: each sets its own port and GPUs in its environment
    # for the task containers it starts, and the Lab store takes both.
    commands = [
        [sys.executable, __file__, "--arm", arm, "--stream", arguments.stream, "--seed", str(arguments.seed)]
        for arm in sorted(ARM_CONFIGS)
    ]
    with ThreadPoolExecutor(len(commands)) as pool:
        results = list(pool.map(lambda command: subprocess.run(command, check=False).returncode, commands))
    failed = [arm for arm, code in zip(sorted(ARM_CONFIGS), results, strict=True) if code]
    if failed:
        sys.exit(f"run.py: the {', '.join(failed)} arm(s) failed; see their output above")


if __name__ == "__main__":
    main()
