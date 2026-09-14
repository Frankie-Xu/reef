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

`reef.patch` is applied to the pinned checkout at image build time and the
public bundle is rebuilt so the engine carries it. It touches the engine
only; the agent, its tools, and its runner live in `harness/`. Four
changes:

- `server_entry.py`: `SAAS_BENCH_<FIELD>` environment variables override the
  simulator roles' provider and model (`SOCIAL_POST_LLM_PROVIDER`,
  `SOCIAL_POST_LLM_MODEL`, `ENTERPRISE_LLM_PROVIDER`, `ENTERPRISE_LLM_MODEL`)
  without editing `config.py` and rebuilding the bundle.
- `server_entry.py`: provider `none` (set for both roles) runs the engine with
  no customer LLM at all. The engine already supports that: customer, macro,
  and competitor posts come from its built-in templates, the agent's own posts
  are not judged, and enterprise negotiation is structured rather than
  generated at this commit. It is the switch for runs that should not depend
  on a paid model.
- `customer_llm.py`: token counts missing from a Responses reply count as
  zero instead of failing the cost log. SGLang's Responses endpoint fills
  `prompt_tokens` but not `input_tokens`.
- `customer_llm.py`, `simulation.py`: the two social-media functions that
  only had Bedrock and Anthropic paths (judging the agent's own post from
  each customer group's view, and a customer's reply to it) get the same
  OpenAI Responses fallback as the other simulator calls. Without it the
  engine's `next-week` fails the first time the agent posts.
- `server_entry.py`: `SAAS_BENCH_SIMULATOR_TIMEOUT_S` bounds one simulator
  request (default the SDK's; `run.sh` sets 300), so a stuck call fails
  instead of stalling the week.

Everything else is the benchmark as published: default `config.py`
difficulty (competitor feedback range 0.2 to 0.5), the bash agent's prompt
and tools, and its `temperature=1.0` sampling. The agent's tools run as the
unprivileged `agent` user the image creates (`CEOBENCH_TOOL_USER` in
`run.sh`), which keeps the engine's source and host-side bundle out of its
reach, and `CEOBENCH_BASH_TIMEOUT_S` widens the benchmark's limit on one
bash command from 1200 s to 3600 s, since a `next-week` late in a long game
can run past it.

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

The reward is online and weekly. CEO-Bench advances in weeks: the agent works
in one conversation until it calls `next-week`, the engine steps seven days
and returns the next dashboard, and the agent rebuilds its conversation from
it. The harness plays the agent turn by turn and reads the engine's day
after every tool call, so each captured turn is filed under its week, and
each week's opening state comes from the engine's dashboard
(`=== Week N Dashboard (Day D) ===`, then the opening cash, individual
subscribers, and enterprise seats, and further down the listed plan prices).
When week N+1 opens, week N is over and, once the credit window below has
closed, each of its decision turns is reported with the week's change in
company value:

    run_rate_N = the engine's MRR at the week's start (live subscriptions at their effective
                 price times seats), read through the task container; when that read fails,
                 subscribers_N x lowest nonzero listed price_N + seats_N x plan C price_N
    V_N        = cash_N + run_rate_N x 7/30 x min(weeks left after week N, H)
    credit_N   = sum over j < K of gamma^j x (V_{N+j+1} - V_{N+j}) / $1,000,000
    score_N    = clip(credit_N, +-C) / max(median |credit| so far, F), capped at +-3

as a single-reference report, so the `sao` recipe trains on the week's turns
while the agent is already playing later weeks and the engine serves the
updated adapter from then on. `H` is `CEOBENCH_VALUE_HORIZON_WEEKS` (26 in
`run.sh`, about six months of a subscription, which stands in for retention);
the weeks left are `days // 7 - N`, so the valuation converges on cash as the
episode ends. The run-rate is the engine's own MRR: at each week start the
harness runs one SQL query against the engine (`/query`, the same
read-only endpoint the agent's `query` tool uses) through the task
container, as root and outside the agent's shell, so the agent's
observation and tools are untouched (`CEOBENCH_ENGINE_READS=0` turns the
read off). The dashboard estimate remains the fallback; it sees neither the
plan mix nor promotions nor negotiated seat prices, so it is a floor.

Only a week's decision turns are reported: the ones whose tool call changed
the company (prices, promotions, tiers, quotas, capacity, spend, targeting,
research, enterprise deals, social posts, or `next-week`;
`DECISION_CALLS` in `harness/agent.py`, matched in the bash command or in a
script the agent wrote and ran). Turns that only queried the books, read
docs, or wrote workspace files are recorded but not trained on: scoring
every turn of a week alike would punish looking at the data as much as the
purchase it preceded, so the week's outcome lands on the decisions in it.
With a few decision turns a week the recipe's batch is 8.

The credit spans `K = CEOBENCH_CREDIT_WEEKS` weeks (4) discounted by
`gamma = CEOBENCH_CREDIT_DISCOUNT` (0.8) per week, so the week that pays for
acquisition is credited with the subscribers that arrive over the weeks after
it; a week is reported once `K` weeks have opened after it, and the pacer
holds week N until the batches of weeks up to N-K-1 have committed. The
last weeks close with the final cash the engine reports when the episode
ends (the number the verifier also reads from `world.nmdb`), valued as
cash alone, and a window that reaches the end is cut short there. Before it is posted the credit is clipped
at `C = CEOBENCH_SCORE_CLIP` (0.05, a $50,000 swing) and divided by the
running median magnitude of the credits so far, floored at
`F = CEOBENCH_SCORE_FLOOR` (0.003): one six-figure R&D purchase then cannot
set the scale for the whole episode, and an ordinary week's difference from
the one before it keeps a gradient. The Harbor reward itself stays the
benchmark's terminal metric (final cash over the starting balance); it is
evaluation only.

The game is paced to the trainer (`CEOBENCH_PACE_BATCH`, set by `run.sh` to
the recipe's batch size). Before each model call the harness closes the
weeks the credit window has finished and reports them, then waits until
every batch the reported turns filled has committed a training release. A week is therefore
played by a policy trained on every week reported so far, whatever the
ratio of step time to play time, and no turn is generated while a step
publishes its adapter, since the engine cannot swap the adapter under a
request in flight. A wait
longer than `CEOBENCH_PACE_TIMEOUT_S` (20 minutes, about four steps) is
forgiven so a batch the recipe declined cannot hold the game forever;
`run.sh` sets the model call's own limit past it (`CEOBENCH_LLM_TIMEOUT_S`,
the HTTP client's timeout; the benchmark's loop retries a timed-out call).
The untrained baseline runs with the pacer off.

Turns of one week share the week's score; the critic's skip-observation GAE
does the credit assignment inside each turn. A weekly credit is dense enough
for SAO's one-rollout-per-step cadence and lines up with the benchmark's own
decision period. It is still myopic about anything the run-rate does not
see inside the window: R&D raises quality and pays through retention and
upgrades weeks later. A judged turn-level signal of the kind single-stream
PPO wants (`recipes/openclawrl/`) is the extension left open.

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

The untrained number is the same episode on the same stack with reporting
off: `CEOBENCH_REPORTS=0` has the harness record the weeks without posting
a report, so nothing trains and the engine serves the base model
throughout (the pacer is off with it). A stack trains one scenario for its
lifetime, so the baseline gets its own scenario and run directory:

```bash
docker compose down
RUN_DIR=$PWD/work-baseline REEF_SCENARIO=ceobench-baseline CEOBENCH_REPORTS=0 \
  CEOBENCH_SEED=42 CEOBENCH_DAYS=500 ./run.sh
```

`results/2026-09-13-baseline-qwen3.6-27b-seed42/` holds the manifest, the
run configuration, and `weeks.csv` (cash, individual subscribers, and
enterprise seats at every weekly dashboard). This episode was served by a
standalone SGLang engine (TP4, the model's 262k window, the record-only
recipe; `run-config.json` has the configuration) rather than the training
stack's rollout engine.

| | |
| --- | --- |
| Outcome | bankrupt on day 255 (week 37); final cash -$66, `reward` -0.00007 |
| Turns | 695 in 35 minutes, 11.5M input / 228k output tokens; every turn recorded |
| Engine | no errors; the agent's shell ran as the unprivileged user throughout |

The agent priced low from the start (A/B/C at $15/$49/$99, then $15/$29/$69,
$9/$19/$39, and $4/$24/$49 by day 161), grew to 671 subscribers by week 11
while losing $4,000 to $7,000 a week, and never closed an enterprise deal.
When subscribers started churning it bought R&D: tier 1 on day 119
($170,000), tier 2 on day 133 ($340,000), and tiers 3, 2, and 1 again on
day 168 ($334,000), all with negative cash flow and payoffs 35 to 380 days
out. Cash was under $10,000 by week 25 and the company went bankrupt on day
255. For scale, the leaderboard's Haiku 4.5 finishes at $59,600 and its
rule-based baseline at $15.8M; this run is not comparable with either
because the simulator roles are a local `Qwen3-4B-Instruct-2507`.

### Trained episode

`results/2026-09-14-trained-qwen3.6-27b-seed42/` holds the same files plus
`holds.csv` (the pacer's holds) and, in `weeks.csv`, `run_rate` (the
engine's MRR), `value_start` and `value_end`, `credit` (the discounted
four-week valuation change), `score` (as posted, clipped and scaled),
`decisions` (the week's decision turns), and `decisions_reported` (those
inside the 24k training window). The stack is `serve.yaml` as shipped:
batch 8, the critic started from an earlier episode's critic with two
critic-only steps, and the pacer holding every request while a filled
batch trains.

| | |
| --- | --- |
| Outcome | completed, day 497 (week 71); not bankrupt; final cash $257,682, `reward` 0.258 |
| Turns | 493 in 3h52m; 144 decision turns, 125 of them (87%) inside the 24k window and reported; 349 read-only turns recorded, not trained on |
| Training | 15 releases: 2 critic-only steps, then 13 actor updates, the first served from week 9 (day 63). The pacer held 15 week starts for 51 minutes in all, 370 s at most (week 4) |
| Rewards | one positive week (6, +0.006); every other week negative, weeks 12 to 16 at the clip (-0.05 to -0.41 before it) |

The decision turns of a week are what the scheme credits, and the record
shows what those decisions were. Week 0 bought R&D (-$178,070), as the
untrained agent also does later in its episode. Weeks 1 to 9 then grew the
base from 0 to 499 subscribers at $6,000 to $10,000 of cash a week, about
$80 to $170 per subscriber; the engine's MRR reached $8,367 a month. Those
weeks were credited between -0.002 and -0.023, with week 6 the only
positive one: under a 26-week horizon a $14 subscriber is worth about $85,
less than it cost, so the valuation read the growth as a small loss even
with the exact MRR in it. The actor's first update was served at week 9.
From week 10 the base plateaued between 580 and 710: usage passed the
tier-1 capacity cap and an outage cost the week (-0.031), open issues
climbed from 37 to 92, and conversion fell from 22% to 17% and then to 5%
as competitor launches hit. The policy answered with more spending, ops to
$1,200 a day, ads to $800, higher model tiers, a 25% lead promotion, and in
week 15 a $333,000 R&D Tier 2 purchase in the same week as 76
cancellations (-$347,005). Weeks 16 to 22 unwound the base at 60 to 170
subscribers a week while the promotions the policy set to recover
conversion (a $50 lead promotion against an $18 plan, corrected the next
turn) took the effective price to zero; the MRR was $0 by week 22. From
week 23 the policy held the empty company at $595 a week to the end.

### What the comparison shows

The trained episode ends with cash and the untrained one does not, which is
what the benchmark scores. How each got there matters as much as the
number. The untrained agent priced low, grew to 670 subscribers by week 12
at a loss every week, then bought five R&D tiers on the way down and went
bankrupt on day 255. The trained policy grew faster in revenue (an MRR of
$11,858 a month by week 15 against a base priced at $4 to $9), but its one
large purchase and its promotions emptied the base by week 22, and the
remaining 49 weeks were the cost-cutting the reward makes safe: a policy
whose every action is credited negative learns to act less. The scheme did
what it was designed to do, crediting decisions only (the read-only turns
went from most of each batch to none of it), valuing the base at the
engine's MRR, keeping every score inside +-3 with the clip absorbing the
two purchases, and updating the actor from week 9, but the credit itself
was negative in 70 of 71 weeks: with acquisition at $80 to $170 a
subscriber against $85 of horizon value, no week of growth at this
simulator's prices scores positive. The horizon
(`CEOBENCH_VALUE_HORIZON_WEEKS`) decides whether growth can ever be
credited; at 26 weeks the untrained agent's own growth strategy loses, and
a horizon matching the weeks left (up to 71) or a per-lever attribution of
what each spend line brought are the two candidates before another full
run. The 24k window is a second limit: the two long turns of week 11 in
which the policy diagnosed its settings and one week (19) whose context a
12k-token query result pushed past the window contributed no decision turns
at all (`decisions_reported` in `weeks.csv`).


