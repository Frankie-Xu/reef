# SAO on CEO-Bench

This example runs [CEO-Bench](https://ceobench.com)
([paper](https://arxiv.org/abs/2606.18543),
[code](https://github.com/zlab-princeton/ceobench-src)) through Reef and trains
the [`sao` recipe](../../../../docs/user-guide/recipes/sao.rst) on it. CEO-Bench
simulates an AI startup for 500 days from $1M in cash: 34 tools, a 19-table
database, simulated social media, and a market with hidden preferences,
competitor pressure, and delayed consequences. The primary metric is final
cash; survival days and bankruptcy are secondary. The benchmark's own
bash-agent baseline plays the game; here its agent role is served by Reef, so
every model call is recorded and attributable, while the two simulator roles
(social posts, enterprise customers) stay outside Reef, as
[Guidance-TTT](../../../tttd/examples/guidance_ttt/README.md) keeps its frozen
executor. This is step 1 of
[issue #428](https://github.com/Human-Agent-Society/reef/issues/428) with one
reward-shaping choice made for it (step 2 stays open; see below).

```text
harbor/                one CEO-Bench episode as a Harbor task
  task.toml              48h agent window, resource limits
  instruction.md         what the task is (the model never sees it: CEO-Bench owns its prompt)
  environment/
    Dockerfile           python:3.13 + uv + the pinned CEO-Bench checkout, patched and rebuilt
    reef.patch           the changes the checkout needs (below)
  tests/
    test.sh              runs the verifier inside the task container
    score.py             decrypts the run's world.nmdb: reward, final cash, survival days, bankrupt
harness/               agent harness (imports reef_client, not reef)
  __init__.py            lazily exports HarborAgent
  agent.py               HarborAgent: sidecar on the host, the benchmark runner in the container
  report.py              values a week and posts its credit against its decision turns' receipts
serve.yaml             Reef + Ray + Slime/Megatron + SGLang, Qwen3.6-27B through LoRA, critic colocated
docker-compose.yaml    the stack in the reef image, host networking, six GPUs
run.py                 one episode, trained while it is played
run.sh                 brings the stack up, then runs run.py through reef-eval
pyproject.toml         makes the harness importable
results/               the untrained baseline and the trained episode: weeks.csv, holds.csv, manifest.json
```

## The harness

Harbor gives the trial a container built from `harbor/environment/Dockerfile`:
CEO-Bench at commit `d2b7b32e` with its own `uv` environment (Python 3.13,
SQLCipher reader included) and a rebuilt public bundle. `HarborAgent.run`
then does three things.

1. **A reef-client sidecar on the host.** `reef_client.serve` listens on an
   ephemeral port, replaces the `Authorization` header with the Reef token,
   stamps `x-reef-scenario`, forwards everything else to Reef unchanged, and
   keeps each `/v1/chat/completions` exchange with its
   `x-reef-agent-record-id` receipt. The benchmark's OpenAI client sees a plain
   base URL.
2. **The benchmark runner in the container.** One `exec` runs
   `saas_bench.agents.bash_agent.run_test --provider openai --base-url
   http://<host>:<port>/v1 --seed S --days D`, the paper's baseline with the
   agent role redirected. The container reaches the host by the LAN address in
   `REEF_SERVICE_URL`, which is why Reef listens on `0.0.0.0` and `run.sh`
   derives the URL from `hostname -I`. `SAAS_BENCH_*`, `OPENAI_*`,
   `ANTHROPIC_*`, and `AWS_*` variables are forwarded into that `exec` for the
   simulator roles; nothing else from the host environment is.
3. **The run directory and the receipts.** When the runner exits, the run
   directory (`world.nmdb`, `config.json`, `checkpoint.json`, `logs/`,
   `agent_workspace/`) is downloaded next to the trial's agent logs, the
   receipts go into the agent context in call order with their week and
   token count, and the sidecar stops.

Harbor then runs `tests/test.sh` in the same container. `score.py` opens the
run's `world.nmdb` with the checkout's own `load_session_db` and writes
`reward.json` with the final cash as the running sum of the `ledger` table,
survival days as the last day any daily table reached, `bankrupt` as final
cash below zero, and `reward` as final cash over the starting balance
(1.0 is break-even). A watcher thread in the harness reads Harbor's
`result.json` for the final cash that closes the episode's last week
(`harness/report.py`).

### The patch to CEO-Bench

`reef.patch` is applied to the pinned checkout at image build time and the
public bundle is rebuilt so the engine carries it. Seven changes:

- `agents/bash_agent/agent.py`: `SAAS_BENCH_OPENAI_CHAT_COMPLETIONS=1` pins the
  agent to `/v1/chat/completions`. The runner otherwise prefers the OpenAI
  Responses API for any OpenAI-compatible endpoint it does not recognize, and
  Reef serves chat completions and Anthropic messages.
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
- `agents/bash_agent/agent.py`: `SAAS_BENCH_MAX_COMPLETION_TOKENS` caps the
  agent's completion request (default 16384, the benchmark's value; `run.sh`
  leaves it unset). It exists for engines whose window cannot hold the
  default plus the prompt.
- `customer_llm.py`, `simulation.py`: the two social-media functions that
  only had Bedrock and Anthropic paths (judging the agent's own post from
  each customer group's view, and a customer's reply to it) get the same
  OpenAI Responses fallback as the other simulator calls. Without it the
  engine's `next-week` fails the first time the agent posts.
- `agents/bash_agent/tools.py`, `run_test.py`: with `SAAS_BENCH_TOOL_USER`
  set and no `bwrap`, the agent's shell runs as that user through `setpriv`
  and the runner hands it the workspace. The image creates the user
  (`agent`), keeps the engine's source and host-side bundle root-only, and
  `run.sh` sets the variable.
- `agents/bash_agent/tools.py`, `agents/bash_agent/agent.py`,
  `server_entry.py`: `SAAS_BENCH_BASH_TIMEOUT` (the runner's limit on one
  bash command, `next-week` included; default 1200 s), `SAAS_BENCH_LLM_TIMEOUT`
  (its hard wall clock per LLM call; default 600 s) and
  `SAAS_BENCH_SIMULATOR_TIMEOUT_S` (one simulator request; default the
  SDK's) override the runner's fixed limits. A paced game holds an LLM call
  through a training step, and a `next-week` late in a long game can run
  past 1200 s.
  `SAAS_BENCH_LLM_TIMEOUT` also sets the OpenAI client's HTTP timeout
  (`httpx.Timeout(600)` in the runner), which otherwise retried a call the
  serving side was still holding and left the original pending. `run.sh`
  sets 3600, 1800 and 300.

Everything else is the benchmark as published: default `config.py`
difficulty (competitor feedback range 0.2 to 0.5), the bash agent's prompt
and tools, and its `temperature=1.0` sampling.

### Simulator roles

By default the simulator roles keep the benchmark's settings, Haiku 4.5 for
social posts and Sonnet 4.5 for enterprise customers through the Anthropic
API, read from `ANTHROPIC_API_KEY` in the environment `run.sh` runs in.
Bedrock works the same way with `AWS_*` credentials and
`SAAS_BENCH_*_LLM_PROVIDER=bedrock`. No credential lives in the repository.

To run without any customer LLM, set both roles to `none`; the market then
speaks in the engine's template posts, which carry the same satisfaction and
virality mechanics but no generated text:

```bash
export SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER=none SAAS_BENCH_ENTERPRISE_LLM_PROVIDER=none
```

A local OpenAI-compatible server with a Responses endpoint can stand in for
both roles instead:

```bash
export SAAS_BENCH_SOCIAL_POST_LLM_PROVIDER=openai SAAS_BENCH_SOCIAL_POST_LLM_MODEL=<served name>
export SAAS_BENCH_ENTERPRISE_LLM_PROVIDER=openai SAAS_BENCH_ENTERPRISE_LLM_MODEL=<served name>
export OPENAI_BASE_URL=http://<host>:<port>/v1 OPENAI_API_KEY=local
```

Both recorded episodes below used one; a result meant to compare with the
paper must use the benchmark's defaults.

## Reward shaping

The reward is online and weekly. CEO-Bench advances in weeks: the agent works
in one conversation until it calls `next-week`, the engine steps seven days
and returns the next dashboard, and the runner rebuilds the conversation from
it. Every request therefore carries the dashboard of the week it belongs to
(`=== Week N Dashboard (Day D) ===`, then the opening cash, individual
subscribers, and enterprise seats, and further down the listed plan prices),
and the sidecar's captures let the harness group turns by week without
touching the benchmark. A reporter thread polls those captures while the
episode runs; when week N+1's dashboard appears, week N is over and each of
its decision turns is reported with the week's change in company value:

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
week gate runs one SQL query against the runner's engine (`/query`, the
same read-only endpoint the agent's `query` tool uses) through the task
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
last weeks close with the verifier's final cash, valued as cash alone, posted
by a watcher thread once Harbor writes `result.json`, and a window that
reaches the end is cut short there. Before it is posted the credit is clipped
at `C = CEOBENCH_SCORE_CLIP` (0.05, a $50,000 swing) and divided by the
running median magnitude of the credits so far, floored at
`F = CEOBENCH_SCORE_FLOOR` (0.003): one six-figure R&D purchase then cannot
set the scale for the whole episode, and an ordinary week's difference from
the one before it keeps a gradient. The Harbor reward itself stays the
benchmark's terminal metric (final cash over the starting balance); it is
evaluation only.

The game is paced to the trainer (`CEOBENCH_PACE_BATCH`, set by `run.sh` to
the recipe's batch size). The sidecar peeks at each request's dashboard;
the first request of a new week closes the weeks the credit window has
finished and reports them, and every request waits until every batch the
reported turns filled has committed a training release. A week is therefore
played by a policy trained on every week reported so far, whatever the
ratio of step time to play time, and no turn is generated while a step
publishes its adapter, since the engine cannot swap the adapter under a
request in flight. A wait
longer than `CEOBENCH_PACE_TIMEOUT_S` (20 minutes, about four steps) is
forgiven so a batch the recipe declined cannot hold the game forever;
`run.sh` widens the runner's own per-call limits past it
(`SAAS_BENCH_LLM_TIMEOUT`, the wall clock and the HTTP client's). The
untrained baseline runs with the pacer off.

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
comparable with the leaderboard: the simulator roles are local stand-ins,
and each row is one episode at temperature 1.0.

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
because the simulator roles are local stand-ins.

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

### Not yet run

- The untrained baseline with the benchmark's Anthropic simulator roles and
  more seeds (issue #428, acceptance criterion 1); replicates of the trained
  episode on other seeds.
- A trained episode with a valuation horizon that covers the weeks left
  (`CEOBENCH_VALUE_HORIZON_WEEKS=71`), or a per-lever attribution of each
  spend line's return, so that a week of growth at the simulator's
  acquisition cost can score positive. Beyond those, a judged turn-level
  signal.

## Open items

- **License.** `zlab-princeton/ceobench-src` carries no license file. The
  image clones a pinned commit at build time and the repository ships none of
  the benchmark's code or fixtures; confirm terms with the authors before a
  result page cites it.
- **Sandboxing.** The benchmark sandboxes the agent's shell with `bwrap` when
  present and falls back to plain execution otherwise. Here the Harbor
  container is the outer sandbox and the agent's shell runs as an
  unprivileged user inside it (`SAAS_BENCH_TOOL_USER`), so it cannot signal
  the root-owned engine or read the engine's source and host-side bundle.
  The copy of
  the `novamind-operation` zipapp in the agent's workspace still embeds the
  database key, as it does upstream; the benchmark's docs recommend hiding
  it behind a wrapper.
