"""Shared pieces of the Science Q&A stream: the reference dataset, the Reef calls, the scorer.

Imported by ``run.py`` (the training stream) and ``evaluate.py`` (the test
split) from this directory. Everything here mirrors the reference
implementation (idanshen/Self-Distillation at ``d77573212fa0``): ``main.py``
loads the splits and builds the prompts, ``eval_science.py`` extracts the
``<answer>`` tag and checks it against the gold letter.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reef_client import ReefClient

REFERENCE_REPOSITORY = "https://github.com/idanshen/Self-Distillation"
REFERENCE_COMMIT = "d77573212fa0"
MODEL = "reef"  # the model name the requests carry; Reef's SGLang serves it

#: The deployment run.sh starts; the same values serve.yaml declares.
SERVICE_URL = os.environ.get("REEF_SERVICE_URL", "http://127.0.0.1:28901")
TOKEN = os.environ.get("REEF_TOKEN", "reef-local")
SCENARIO = os.environ.get("REEF_SCENARIO", "sdft-science-qa")
#: ``data/science_data`` of the reference checkout run.sh makes.
DATA_DIR = Path(os.environ.get("SCIENCE_DATA_DIR", "work/self-distillation/data/science_data"))

#: The service is gone or rejecting requests; waiting cannot help.
SERVICE_GONE = -1


def load_split(split: str) -> list[dict[str, Any]]:
    """The reference's ``train_data`` or ``eval_data`` split as plain dicts.

    A training example carries ``messages`` (system and user) and
    ``output_text`` (GPT-4o's response, the demonstration); a test example
    carries ``prompt`` (the same two messages) and ``answer`` (a letter).
    """
    from datasets import load_from_disk

    path = DATA_DIR / f"{split}_data"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run.sh checks out {REFERENCE_REPOSITORY} at {REFERENCE_COMMIT}")
    return [dict(row) for row in load_from_disk(str(path))]


def wave_schedule(count: int, epochs: int, prompts_per_step: int, seed: int) -> list[list[int]]:
    """The training order: each epoch a fresh shuffle, cut into steps of ``prompts_per_step`` prompts.

    Steps run across the epoch boundary so every step has the same size (the
    recipe batches exactly that many reports); the prompts left at the very
    end that do not fill a step are dropped.
    """
    generator = random.Random(seed)
    order: list[int] = []
    for _ in range(epochs):
        epoch = list(range(count))
        generator.shuffle(epoch)
        order.extend(epoch)
    return [
        order[start : start + prompts_per_step]
        for start in range(0, len(order) - prompts_per_step + 1, prompts_per_step)
    ]


def extract_answer(text: str) -> str:
    """The reference scorer's rule: the text inside the last ``<answer>`` tag, stripped."""
    answer = text.split("<answer>")[-1]
    return answer.split("</answer>")[0].strip()


def make_client(timeout_s: float = 1800.0) -> ReefClient:
    return ReefClient(SERVICE_URL, token=TOKEN, timeout_s=timeout_s)


def ask(
    client: ReefClient, messages: Sequence[dict[str, str]], *, temperature: float, max_tokens: int
) -> tuple[str, str, int]:
    """One chat completion through Reef: the text, its receipt, and its completion token count.

    Sampling beyond temperature and the window comes from serve.yaml's
    ``sampling_defaults`` (top_p 1, no top_k, no repetition penalty, the
    reference's vLLM settings).
    """
    response, agent_record_id = client.inference_with_record(
        SCENARIO,
        "/v1/chat/completions",
        {"model": MODEL, "messages": list(messages), "max_tokens": max_tokens, "temperature": temperature},
    )
    return response["choices"][0]["message"]["content"], agent_record_id, int(response["usage"]["completion_tokens"])


def ask_all(
    client: ReefClient,
    prompts: Sequence[Sequence[dict[str, str]]],
    *,
    temperature: float,
    max_tokens: int,
    concurrency: int,
) -> list[tuple[str, str, int]]:
    """``ask`` for every prompt at once, results in prompt order."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(
            pool.map(lambda messages: ask(client, messages, temperature=temperature, max_tokens=max_tokens), prompts)
        )


def training_release_count() -> int | None:
    """Training releases committed so far; ``None`` while the service is busy.

    The scenario's registry lock serializes release reads with training, so
    this request legitimately stalls for the length of an in-flight train
    step (which includes a weight publish). A timeout is "try again", not an
    error. A refused connection or an HTTP rejection is terminal.
    """
    request = urllib.request.Request(
        f"{SERVICE_URL}/reef/scenarios/{SCENARIO}/releases", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return 0  # the scenario does not exist yet: the first request creates it
        return SERVICE_GONE  # answered and rejected: not our deployment
    except urllib.error.URLError as error:
        if isinstance(getattr(error, "reason", None), ConnectionRefusedError):
            return SERVICE_GONE
        return None  # stalled behind a train step; try again
    except TimeoutError:
        return None
    return sum(1 for row in payload["releases"] if row.get("operation") == "training")


def wait_for_training(expected: int, timeout_s: float) -> int:
    """Block until the scenario has committed ``expected`` training releases; return the count seen."""
    deadline = time.time() + timeout_s
    while True:
        count = training_release_count()
        if count == SERVICE_GONE:
            raise RuntimeError(f"the Reef service at {SERVICE_URL} is gone or rejects scenario {SCENARIO}")
        if count is not None and count >= expected:
            return count
        if time.time() > deadline:
            raise TimeoutError(f"training release {expected} did not commit within {timeout_s:.0f}s (seen: {count})")
        time.sleep(5.0)
