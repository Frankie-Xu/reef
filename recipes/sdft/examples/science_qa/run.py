"""The Science Q&A training stream, written out.

For each step of ``PROMPTS_PER_STEP`` training prompts, in the order
``science.wave_schedule`` fixes from the seed:

    ask     — one chat completion per prompt through Reef at temperature 1.0:
              the student's on-policy sample, recorded with its tokens and
              log-probs
    report  — the dataset's demonstration (GPT-4o's response) as the report's
              ``context`` against that sample's receipt
    learn   — the sdft recipe batches the step's reports, Slime runs one
              optimizer step of the self-distillation loss, and the updated
              weights are published; the loop blocks on that release, so the
              next step's samples come from the updated policy

This is the reference's ``main.py`` protocol (2 epochs, 32 prompts per
optimizer step, one sample per prompt, a 1024-token window) with Reef in the
place of TRL's trainer. Every ``EVAL_EVERY`` steps, and before the first, the
test split is scored (``evaluate.py``); the accuracies and the per-step
statistics land under ``work/`` for the README's curve.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import science
from evaluate import evaluate

EPOCHS = int(os.environ.get("SDFT_EPOCHS", "2"))
PROMPTS_PER_STEP = int(os.environ.get("SDFT_PROMPTS_PER_STEP", "32"))  # MUST equal serve.yaml's batch size
SEED = int(os.environ.get("SDFT_SEED", "42"))
MAX_TOKENS = int(os.environ.get("SDFT_MAX_TOKENS", "1024"))
EVAL_EVERY = int(os.environ.get("SDFT_EVAL_EVERY", "20"))
#: A ceiling on the steps to run (0 runs the whole schedule); a smoke run sets a few.
STEPS = int(os.environ.get("SDFT_STEPS", "0"))
#: Ceiling on waiting for one step's training release, weight publish included.
TRAIN_TIMEOUT_S = float(os.environ.get("SDFT_TRAIN_TIMEOUT_S", "3600"))
WORK = Path(os.environ.get("SDFT_WORK_DIR", "work"))


def score_test_split(client, test_examples, label: str) -> float:
    result = evaluate(client, test_examples, label=label)
    (WORK / "eval").mkdir(parents=True, exist_ok=True)
    (WORK / "eval" / f"{label}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[eval {label}] accuracy {result['accuracy']:.4f} ({result['num_correct']}/{result['num_total']}) "
        f"in {result['elapsed_s']:.0f}s",
        flush=True,
    )
    return result["accuracy"]


def main() -> None:
    train = science.load_split("train")
    test = science.load_split("eval")
    schedule = science.wave_schedule(len(train), EPOCHS, PROMPTS_PER_STEP, SEED)
    if STEPS:
        schedule = schedule[:STEPS]
    client = science.make_client()
    WORK.mkdir(parents=True, exist_ok=True)
    steps_log = WORK / "steps.jsonl"
    print(
        f"{len(train)} training prompts, {len(test)} test prompts, {len(schedule)} steps of {PROMPTS_PER_STEP} "
        f"({EPOCHS} epochs, seed {SEED}); service {science.SERVICE_URL}, scenario {science.SCENARIO}",
        flush=True,
    )

    releases_before = science.wait_for_training(0, TRAIN_TIMEOUT_S)
    score_test_split(client, test, "step-0")
    for step, prompt_indices in enumerate(schedule, start=1):
        started = time.time()
        examples = [train[index] for index in prompt_indices]
        outputs = science.ask_all(
            client,
            [row["messages"] for row in examples],
            temperature=1.0,
            max_tokens=MAX_TOKENS,
            concurrency=PROMPTS_PER_STEP,
        )
        sampled = time.time()
        for row, (_, receipt, _) in zip(examples, outputs, strict=True):
            client.report(science.SCENARIO, {"references": [receipt], "metadata": {"context": row["output_text"]}})
        releases = science.wait_for_training(releases_before + step, TRAIN_TIMEOUT_S)
        finished = time.time()
        record = {
            "step": step,
            "training_releases": releases,
            "mean_completion_tokens": sum(tokens for _, _, tokens in outputs) / len(outputs),
            "truncated": sum(1 for _, _, tokens in outputs if tokens >= MAX_TOKENS),
            "sample_s": sampled - started,
            "train_s": finished - sampled,
        }
        with steps_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"[step {step}/{len(schedule)}] {record['mean_completion_tokens']:.0f} tokens/sample, "
            f"{record['truncated']} truncated, sample {record['sample_s']:.0f}s, train {record['train_s']:.0f}s",
            flush=True,
        )
        if EVAL_EVERY and step % EVAL_EVERY == 0 and step != len(schedule):
            score_test_split(client, test, f"step-{step}")
    score_test_split(client, test, f"step-{len(schedule)}")


if __name__ == "__main__":
    main()
