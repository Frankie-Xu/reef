# SAO on CEO-Bench

This example runs [CEO-Bench](https://ceobench.com)
([paper](https://arxiv.org/abs/2606.18543),
[code](https://github.com/zlab-princeton/ceobench-src)) through Reef and trains
the [`sao` recipe](../../../../docs/user-guide/recipes/sao.rst) on it. CEO-Bench
simulates an AI startup for 500 days from $1M in cash: 34 tools, a 19-table
database, simulated social media, and a market with hidden preferences,
competitor pressure, and delayed consequences. The primary metric is final
cash; survival days and bankruptcy are secondary. The benchmark's own
bash agent plays the game: `harness/` is that agent, its loop played from
the host with its prompt, tools, and tool executor taken from the pinned
checkout, and its model calls served by Reef, so every call is recorded and
attributable. The two simulator roles (social posts, enterprise customers)
stay outside Reef.

```text
harbor/                one CEO-Bench episode as a Harbor task: the world and the verifier
  task.toml              48h agent window, resource limits
  instruction.md         what the task is (the model never sees it: CEO-Bench owns its prompt)
  environment/
    Dockerfile           python:3.13 + uv + the pinned CEO-Bench checkout, patched and rebuilt
    reef.patch           the changes the engine needs (below)
    engine.py            starts the episode's session and engine; stops it for the verifier
  tests/
    test.sh              runs the verifier inside the task container
    score.py             decrypts the run's world.nmdb: reward, final cash, survival days, bankrupt
harness/               the benchmark's bash agent (imports reef_client, not reef)
  __init__.py            lazily exports HarborAgent
  agent.py               the agent: the benchmark's loop, conversation, feedback texts, retries
  tools.py               its six tools, run in the agent's workspace inside the task container
  harbor_agent.py        HarborAgent: one trial, the agent served by Reef and its weeks credited
  report.py              the reward: weeks, valuation and credit, scaled scores, posting, the pacer
serve.yaml             Reef + Ray + Slime/Megatron + SGLang, Qwen3.6-27B through LoRA, critic colocated
docker-compose.yaml    the stack in the reef image, host networking, six GPUs
run.py                 one episode, trained while it is played
run.sh                 brings the stack up, then runs run.py through reef-eval
pyproject.toml         makes the harness importable
results/               the untrained baseline and the trained episode: weeks.csv, holds.csv, manifest.json
```

## The harness

`harness/` is CEO-Bench's bash agent, the way the benchmark's own
`bash_agent/` package is `agent.py` and `tools.py`. What the model is asked,
what it may call, and what its tools return are the benchmark's; what the
harness adds is where the model calls go, when a week is credited, and how
long the game waits for the trainer.

- **`agent.py`** is the benchmark's `BashAgent` at the pinned commit, its
  OpenAI chat-completions path, kept step for step: the conversation starts
  empty and is rebuilt from the system prompt (with the workspace's
  `MEMORY.md` appended) at every week advance; one tool call per turn, the
  rest of a response's calls answered `[Skipped - only one tool per turn
  ...]`; the benchmark's feedback texts for a response without a tool call or
  with arguments that are not JSON; its retry rules for API errors. The
  prompt and the tool definitions are not copied into this repository: when
  an episode starts the harness reads them from the pinned checkout in the
  task image, built by the benchmark's own classes, so they are the
  benchmark's byte for byte.
- **`tools.py`** is the benchmark's tool executor: `bash`, `read_file`,
  `write_file`, `edit_file`, `search_files`, and `glob_files`, confined to
  the agent's workspace, with the same shell environment, output assembly
  (`[stderr]`, `[exit code: N]`, the 30,000-character cut), timeout rules,
  and file semantics. The harness uploads it into the task container and
  runs each call through Harbor's `exec` as the unprivileged `agent` user;
  the benchmark sandboxes the same shell with `bwrap` where it can, and here
  the container and the user boundary are the sandbox.
- **`harbor_agent.py`** plays one Harbor trial. It has the task start the
  episode's engine session (`harbor/environment/engine.py`), reads the
  engine's status, dashboard, and books over its HTTP API, and serves every
  model call through a reef-client proxy on the host's loopback:
  `reef_client.serve` replaces the `Authorization` header with the Reef
  token, stamps `x-reef-scenario`, forwards the request body unchanged, and
  keeps each exchange with its `x-reef-agent-record-id` receipt. The agent's
  OpenAI client sees a plain base URL.
- **`report.py`** is the reward. Every captured turn is filed under the
  simulated week it was played in (the harness knows the engine's day at
  every call), a finished week is credited as described under "Reward
  shaping", and its decision turns are posted to Reef while the episode
  runs. The last weeks close with the final cash the engine reports when
  the episode ends.

When the episode ends the run directory (`world.nmdb`, `config.json`,
`logs/`, `agent_workspace/`) is downloaded next to the trial's agent logs,
`turns.jsonl` there lists every tool call with its output, and the receipts
go into the agent context in call order with their week, token count, and
decision. Harbor then runs `tests/test.sh` in the same container, which
stops the engine if the harness could not and scores the run: `score.py`
opens the run's `world.nmdb` with the checkout's own `load_session_db` and
writes `reward.json` with the final cash as the running sum of the `ledger`
table, survival days as the last day any daily table reached, `bankrupt` as
final cash below zero, and `reward` as final cash over the starting balance
(1.0 is break-even).

### The patch to CEO-Bench

`reef.patch` is applied to the pinned checkout at image build time and
touches the engine only:

- `server_entry.py`: `SAAS_BENCH_<FIELD>` variables set the simulator
  roles' provider and model; provider `none` runs both roles on the engine's
  template posts; `SAAS_BENCH_SIMULATOR_TIMEOUT_S` bounds one simulator
  request (300 s in `run.sh`).
- `customer_llm.py`, `simulation.py`: an OpenAI path for the two
  social-media calls that only had Anthropic and Bedrock ones, and tolerance
  for the token counts SGLang's Responses endpoint leaves out.

Everything else is the benchmark as published: default `config.py`
difficulty, the bash agent's prompt, tools, and `temperature=1.0`. The
agent's tools run as the unprivileged `agent` user the image creates
(`CEOBENCH_TOOL_USER`), and `CEOBENCH_BASH_TIMEOUT_S` (3600 s in `run.sh`)
widens the benchmark's 1200 s limit on one bash command for late-game
`next-week` calls.

### Simulator roles

The two simulator roles (the customers who post on social media, and the
enterprise buyers) are their own LLM calls inside the engine, outside Reef.
The episodes recorded below run both on `Qwen3-4B-Instruct-2507`, served by
SGLang on one spare GPU of the same node as an OpenAI-compatible endpoint,
so no paid API is involved:

```bash
docker run -d --name ceobench-sim --network host --gpus '"device=7"' -v ~/models:/root/models reef \
  python -m sglang.launch_server --model-path /root/models/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen3-4B-Instruct-2507 --host 0.0.0.0 --port 30100 --tp 1 \
  --mem-fraction-static 0.28 --context-length 32768
export SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER=openai SAAS_BENCH_SOCIAL_POST_LLM_MODEL=Qwen3-4B-Instruct-2507
export SAAS_BENCH_ENTERPRISE_LLM_PROVIDER=openai SAAS_BENCH_ENTERPRISE_LLM_MODEL=Qwen3-4B-Instruct-2507
export OPENAI_BASE_URL=http://<host>:30100/v1 OPENAI_API_KEY=local
```

The benchmark's own setting is Haiku 4.5 for social posts and Sonnet 4.5
for enterprise customers through the Anthropic API (`ANTHROPIC_API_KEY`;
Bedrock with `AWS_*` credentials and `SAAS_BENCH_*_LLM_PROVIDER=bedrock`),
which is what `run.sh` uses when the variables above are unset and what a
result meant to compare with the paper or the leaderboard needs. Setting
both roles to `none` runs the market on the engine's template posts, with
the same satisfaction and virality mechanics but no generated text. No
credential lives in the repository.

## Reward shaping

CEO-Bench advances in weeks: the agent works in one conversation until it
calls `next-week` and rebuilds it from the next dashboard. The harness knows
the engine's day at every model call, so every turn is filed under its
week, and once enough later weeks have opened each decision turn of week N
is reported with the week's credit as its score:

    run_rate_N = the engine's MRR at the week's start (the dashboard's listed-price
                 estimate when the books cannot be read)
    V_N        = cash_N + run_rate_N x 7/30 x min(weeks left after week N, H)
    credit_N   = sum over j < K of gamma^j x (V_{N+j+1} - V_{N+j}) / $1,000,000
    score_N    = clip(credit_N, +-C) / max(median |credit| so far, F), capped at +-3

- `V_N` is the company's value at the week's start: cash plus its monthly
  recurring revenue over the weeks left, at most `H` = 26
  (`CEOBENCH_VALUE_HORIZON_WEEKS`).
- `credit_N` sums the value changes of the next `K` = 4 weeks
  (`CEOBENCH_CREDIT_WEEKS`), discounted by `gamma` = 0.8
  (`CEOBENCH_CREDIT_DISCOUNT`), so the week that pays for acquisition is
  credited with the subscribers that arrive after it. The last weeks close
  with the engine's final cash.
- `score_N` clips the credit at `C` = 0.05 (`CEOBENCH_SCORE_CLIP`) and
  divides by the running median magnitude, floored at `F` = 0.003
  (`CEOBENCH_SCORE_FLOOR`), so one six-figure R&D purchase cannot set the
  scale for the episode.
- Only decision turns are reported: tool calls that change the company
  (`DECISION_CALLS` in `harness/report.py`). Turns that only read are
  recorded but not trained on, and so are turns over
  `CEOBENCH_TRAIN_MAX_TOKENS` (24k, the trainer's window).

The game is paced to the trainer (`CEOBENCH_PACE_BATCH`, the recipe's batch
size, 8): before each model call the harness reports the weeks the credit
window has closed, then waits until every filled batch has committed a
release, so a week is played by a policy trained on every week reported so
far. A wait over `CEOBENCH_PACE_TIMEOUT_S` (20 minutes) is forgiven.
`CEOBENCH_REPORTS=0` turns reporting and pacing off (the untrained
baseline). The Harbor reward stays the benchmark's final cash over the
starting balance; it is evaluation only.

## Run

Prerequisites: Docker with the NVIDIA runtime, `uv`, the `reef` image
(`docker build -f docker/Dockerfile.reef -t reef .` from the repository root),
the policy model, and credentials for the simulator roles.

```bash
cd recipes/sao/examples/ceobench
hf download Qwen/Qwen3.6-27B --local-dir ~/models/Qwen3.6-27B
export ANTHROPIC_API_KEY=...
CEOBENCH_SEED=42 CEOBENCH_DAYS=500 ./run.sh
```

`run.sh` reads `REEF_IMAGE` (default `reef`), `MODEL_DIR` (`~/models`),
`RUN_DIR` (`./work`), and `REEF_GPU_0..5` (the six devices the stack uses:
four actor GPUs, TP4 with the critic colocated, and a two-GPU rollout
engine). The policy is `Qwen3.6-27B` trained through Megatron Bridge LoRA:
the base stays frozen in the actor and in the SAO critic, the adapters and
the critic's value head train, and the rollout engine serves the published
adapter. Turns longer than `CEOBENCH_TRAIN_MAX_TOKENS` (24k tokens, the
largest a step fits beside the two resident bases; the header of
`serve.yaml` gives the budget) are served and recorded but not trained on.
`run.sh` mints a token into `$RUN_DIR/token`, brings the stack up with `docker compose up --wait`,
and runs `run.py` in an ephemeral `uv` environment with `reef-eval[harbor]`
and this harness. The episode row lands in `work/lab`, the trial's run
directory under the trial's `agent/ceobench/`.

This is test-time training: the policy adapts inside the episode it is
scored on, and the number to compare is that episode's final cash against the
same seed played by the untrained model. The value model need not start
cold: `serve.yaml` points `--critic-init` at `$RUN_DIR/critic-init`, and a
copy of an earlier run's latest critic checkpoint there
(`checkpoints/megatron-critic/iter_N` with its
`latest_checkpointed_iteration.txt`) is loaded, weights and optimizer, at
the first start; the critic then trains for two commits on its own before
the actor's first update (`num-critic-only-steps`). An empty directory
leaves it cold. Replicates are independent runs from
the base model, one stack each (`docker compose down` between them, or a
fresh `RUN_DIR`): a Reef process trains one scenario for its lifetime, so a
second seed on the same stack would start from the first seed's adapter.
After the episode `run.py` waits for the scenario's training releases to
stop growing (fifteen quiet minutes, longer than one step), so the adapter
on disk is the one the episode ended with:

```bash
curl -sS -H "Authorization: Bearer $(cat work/token)" \
  http://$(hostname -I | awk '{print $1}'):28900/reef/scenarios/ceobench-sao/releases
```

## Results

Seed 42, 500 days (the benchmark rounds it down to 71 whole weeks, 497
days), the simulator roles as above, one episode each. Neither number is
comparable with the leaderboard: both simulator roles are a local
`Qwen3-4B-Instruct-2507` (see "Simulator roles"), and each row is one
episode at temperature 1.0. Both episodes were played
by the previous form of this harness, which ran the benchmark's own runner
inside the task container with its agent role redirected to Reef; the
harness above plays the same agent from the host and is to be re-run.

| Policy | Reward | Outcome | Final cash | `reward` |
| --- | --- | --- | ---: | ---: |
| `Qwen3.6-27B`, untrained (`CEOBENCH_REPORTS=0`) | none | bankrupt on day 255 (week 37) | -$66 | -0.00007 |
| `Qwen3.6-27B`, trained in the episode (`serve.yaml`) | the weekly credit above | completed, day 497 | $257,682 | 0.258 |

| Week | Day | Untrained cash | Subscribers | Trained cash | Subscribers | Engine MRR / month |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | $977,614 | 19 | $821,930 | 23 | $207 |
| 5 | 35 | $934,244 | 194 | $789,541 | 198 | $2,692 |
| 7 | 49 | $918,908 | 374 | $752,089 | 314 | $4,896 |
| 10 | 70 | $905,911 | 658 | $725,410 | 586 | $10,089 |
| 12 | 84 | $897,818 | 670 | $698,655 | 647 | $11,626 |
| 15 | 105 | $882,751 | 559 | $649,008 | 639 | $11,858 |
| 17 | 119 | $877,846 | 466 | $295,903 | 327 | $5,422 |
| 20 | 140 | $364,051 | 310 | $289,548 | 183 | $776 |
| 22 | 154 | $352,430 | 227 | $286,912 | 38 | $0 |
| 25 | 175 | $8,343 | 208 | $285,052 | 0 | $0 |
| 30 | 210 | $3,899 | 10 | $282,077 | 0 | $0 |
| 37 | 259 | -$66 (bankrupt on day 255) | 1 | $277,912 | 0 | $0 |
| 50 | 350 | bankrupt | | $270,177 | 0 | $0 |
| 70 | 490 | bankrupt | | $258,277 | 0 | $0 |

### Untrained baseline

The same episode with reporting off: `CEOBENCH_REPORTS=0` records the
weeks without posting, so nothing trains and the engine serves the base
model. It gets its own scenario and run directory:

```bash
docker compose down
RUN_DIR=$PWD/work-baseline REEF_SCENARIO=ceobench-baseline CEOBENCH_REPORTS=0 \
  CEOBENCH_SEED=42 CEOBENCH_DAYS=500 ./run.sh
```

`results/2026-09-13-baseline-qwen3.6-27b-seed42/` holds the manifest, the
run configuration, and `weeks.csv` (served by a standalone SGLang engine,
TP4, the model's 262k window).

| | |
| --- | --- |
| Outcome | bankrupt on day 255 (week 37); final cash -$66, `reward` -0.00007 |
| Turns | 695 in 35 minutes, 11.5M input / 228k output tokens |

The agent priced low ($15/$49/$99, down to $4/$24/$49 by day 161), grew to
671 subscribers by week 11 while losing $4,000 to $7,000 a week, bought
five R&D tiers ($844,000 in all) as subscribers churned, and went bankrupt
on day 255.

### Trained episode

`results/2026-09-14-trained-qwen3.6-27b-seed42/` holds the same files plus
`holds.csv` (the pacer's holds) and, in `weeks.csv`, the engine's MRR, the
values, the credit and score, and the decision counts. `serve.yaml` as
shipped: batch 8, the critic warm-started from an earlier episode, the
pacer on.

| | |
| --- | --- |
| Outcome | completed, day 497 (week 71); final cash $257,682, `reward` 0.258 |
| Turns | 493 in 3h52m; 144 decision turns, 125 of them reported (19 were over the 24k window) |
| Training | 15 releases: 2 critic-only steps, then 13 actor updates, the first served from week 9; the pacer held the game for 51 minutes in all, 370 s at most |
| Rewards | one positive week (6, +0.006); every other week negative, weeks 12 to 16 at the clip |

Week 0 bought R&D (-$178,070). Weeks 1 to 9 grew the base to 499
subscribers at $6,000 to $10,000 a week ($80 to $170 per subscriber; MRR
$8,367 a month) and were credited -0.002 to -0.023 each: under a 26-week
horizon a $14 subscriber is worth about $85, less than it cost. From week
10 the base plateaued (a capacity outage, 92 open issues, conversion down
from 22% to 5%); the policy spent more (ops $1,200 a day, ads $800, a 25%
promotion) and bought a $333,000 R&D tier in week 15. Promotions above the
plan price then emptied the base by week 22, and the last 49 weeks were
cost-cutting at $595 a week.

### What the comparison shows

The trained episode ends with cash and the untrained one does not, which
is what the benchmark scores, but both take the same shape: growth at a
loss, one large purchase, an empty company. The scheme did what it was
built to do (decision turns only, the engine's MRR, scores inside +-3, the
actor updated from week 9), yet the credit was negative in 70 of 71 weeks:
at $80 to $170 per subscriber against $85 of horizon value, no week of
growth scores positive under a 26-week horizon, so the policy learns to act
less. The horizon (`CEOBENCH_VALUE_HORIZON_WEEKS`) is the lever to change
first; the 24k training window is a second limit (weeks 11 and 19
contributed no decision turns).
