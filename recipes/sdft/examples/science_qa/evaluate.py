"""Score the served model on the reference's Science Q&A test split.

Every test prompt is answered greedily (temperature 0, a 2048-token window,
the reference's ``eval_science.py`` settings) through Reef, and the letter
inside the last ``<answer>`` tag is checked against the gold letter: exact
match, the paper's metric. ``run.py`` calls :func:`evaluate` before training
and every few steps; run this file alone to score whatever the stack serves
now.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import science
from reef_client import ReefClient

MAX_TOKENS = 2048
CONCURRENCY = 64


def evaluate(client: ReefClient, examples: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    """Accuracy over ``examples`` and the per-example predictions."""
    started = time.time()
    outputs = science.ask_all(
        client, [row["prompt"] for row in examples], temperature=0.0, max_tokens=MAX_TOKENS, concurrency=CONCURRENCY
    )
    samples = []
    for row, (text, receipt, completion_tokens) in zip(examples, outputs, strict=True):
        predicted = science.extract_answer(text)
        samples.append(
            {
                "answer": row["answer"],
                "predicted": predicted,
                "correct": predicted == row["answer"],
                "completion_tokens": completion_tokens,
                "receipt": receipt,
            }
        )
    correct = sum(1 for sample in samples if sample["correct"])
    return {
        "label": label,
        "accuracy": correct / len(samples),
        "num_correct": correct,
        "num_total": len(samples),
        "mean_completion_tokens": sum(sample["completion_tokens"] for sample in samples) / len(samples),
        "elapsed_s": time.time() - started,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="eval", help="name of the result file under --output-dir")
    parser.add_argument("--output-dir", default="work/eval", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N test prompts (0 = all)")
    arguments = parser.parse_args()

    examples = science.load_split("eval")
    if arguments.limit:
        examples = examples[: arguments.limit]
    result = evaluate(science.make_client(), examples, label=arguments.label)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    path = arguments.output_dir / f"{arguments.label}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[{arguments.label}] accuracy {result['accuracy']:.4f} ({result['num_correct']}/{result['num_total']}) "
        f"in {result['elapsed_s']:.0f}s -> {path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
