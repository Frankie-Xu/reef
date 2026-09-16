# SDFT on Science Q&A

This example reproduces the single-task result of
[Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)
on its Science Q&A split (Table 5): Qwen2.5-7B-Instruct on the Chemistry L-3
subset of SciKnowEval, learning from GPT-4o demonstrations. The method itself is
the `sdft` recipe package (`recipes/sdft/`). This directory holds the loop around
it: the reference implementation's dataset, a driver that streams the training
prompts through Reef and reports each demonstration against the student's
receipt, and the reference's exact-match scorer run against the served model.

The [`sdft` recipe page](../../../../docs/user-guide/recipes/sdft.rst) documents
the recipe's configuration and driver flags, and
[Evolve your model](../../../../docs/user-guide/evolve-your-model.rst) walks
through the training stack this example starts. This README records the
protocol, its distance from the paper's, and the numbers.

```text
science.py            the reference dataset, the Reef calls, the scorer's answer rule, the training order
run.py                the stream: 32 prompts, 32 samples, 32 reports, one step, wait for the release, repeat
evaluate.py           the test split through the served model, greedy, exact match on the <answer> tag
plot.py               the learning curve of a recorded run, in the repository's figure style
serve.yaml            Reef + Ray + Slime/Megatron + SGLang stack config: full fine-tuning, the paper's optimizer
serve-sft.yaml        the same stack with the SFT control's recipe
baseline_sft.py       the SFT control: the demonstration as the assistant turn, Slime's stock sft_loss
docker-compose.yaml   the stack in the reef image, five GPUs
run.sh                checks out the reference at its pin, starts the stack, runs run.py
results/              the recorded runs
```

## The protocol

The reference implementation ([idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation)
at `d77573212fa0`) ships the split under `data/science_data`: 2674 training
prompts, each a system message fixing the `<reasoning>`/`<answer>` format and
a user message with a four-option chemistry question, plus `output_text`, the
GPT-4o response kept as the demonstration; and 507 test prompts with their
gold letter. Its `main.py` trains with TRL for 2 epochs at learning rate 5e-5,
32 prompts per optimizer step, one on-policy sample per prompt in a 1024-token
window, per-token forward KL, truncated importance sampling capped at 2, the
first three response tokens skipped; `eval_science.py` decodes the test split
greedily with a 2048-token window and scores the text inside the last
`<answer>` tag by exact match.

`run.py` keeps that protocol with Reef in the trainer's place. It fixes the
training order from the seed (each epoch a fresh shuffle, cut into steps of 32
across the epoch boundary, the last 4 prompts of 5348 dropped: 167 steps), and
for each step sends the 32 prompts through Reef at temperature 1.0, reports
each demonstration as the report's `context` against the sample's receipt, and
waits for the step's training release before sampling the next step, so every
sample is on policy. `serve.yaml` sets the paper's optimizer on the Slime
driver: full fine-tuning of the 7B model on four GPUs (tensor parallel 4),
AdamW at 5e-5 with a cosine schedule over the 167 steps and a 10% warmup, no
weight decay, gradient clipping at 1; the `sdft` family's flags carry the
forward KL, the cap of 2 and the three skipped tokens. The test split is scored
before training and every 20 steps.

## The SFT control

Table 5 compares SDFT with supervised fine-tuning on the same demonstrations.
`baseline_sft.py` is that arm on the same stack: a recipe that accepts the
same reports, a processor that renders each demonstration as the assistant
turn of the recorded request with the served model's chat template and
trains those tokens (the content, the end-of-turn token and the template's
turn separator) with Slime's stock `sft_loss`, and an objective that names
the family by dotted reference so the Slime driver and its workers import it
from this directory. The student's samples are recorded and ignored, so
`run.py` drives both arms unchanged: the same prompts in the same order, the
same 32-prompt steps, the same optimizer in `serve-sft.yaml`.

What differs from the reference: sampling goes through SGLang instead of vLLM
(the same settings: temperature 1, top-p 1, no top-k, no repetition penalty),
the trainer is Megatron instead of TRL on one GPU, and the reference's
`ref_model_mixup_alpha` (a slow-moving copy of the weights that TRL keeps for
its KL-to-reference term, off at `beta = 0`) has no counterpart.

## Setup (once)

The training stack needs the GPU environment described in
[Evolve your model](../../../../docs/user-guide/evolve-your-model.rst), as the
`reef` image. On the host:

```bash
pip install uv
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/Qwen2.5-7B-Instruct
```

## Run

```bash
cd recipes/sdft/examples/science_qa
./run.sh                                    # the SDFT arm: the stack, then the 167 steps
SDFT_STEPS=2 ./run.sh                       # a smoke run: the baseline score and two steps
uv run --no-project --python 3.12 --with reef-client --with datasets evaluate.py --label now
uv run --no-project --python 3.12 --with matplotlib plot.py --run-dir work --out results/<run>
docker compose down                         # stop the stack
SERVE_CONFIG=serve-sft.yaml REEF_SCENARIO=sft-science-qa RUN_DIR=$PWD/work-sft ./run.sh   # the SFT control
```

`run.sh` reads `REEF_IMAGE` (default `reef`), `MODEL_DIR` (default
`~/models`), `RUN_DIR` (default `./work`: the reference checkout, the stack's
state and checkpoints, `steps.jsonl` and `eval/*.json`), `SERVE_CONFIG`
(default `serve.yaml`), and `REEF_GPU_0..4`. A trained stack is bound to its
scenario and run directory, so each arm gets its own `RUN_DIR` and
`REEF_SCENARIO`, and `docker compose down` separates them.
`run.py` reads `SDFT_EPOCHS`, `SDFT_PROMPTS_PER_STEP` (must equal the recipe's
batch size in `serve.yaml`), `SDFT_SEED`, `SDFT_MAX_TOKENS`, `SDFT_EVAL_EVERY`
and `SDFT_STEPS`.

## Results

In progress.
