# The reference path on Apple Silicon — Hermes + Harbor, Qwen3.5-9B, gated

The GSM8K homework stream run the way the benchmark defines it: the real Hermes
agent in a Harbor task container, a judge service beside it, and reef-eval
sequencing 72 sessions — against a host-native Reef+MLX service where the policy
both serves and trains in one process. Earlier MLX records here used a
Mac-native driver standing in for Hermes; this one does not.

Every candidate passes the [candidate gate](../../../../candidate_evaluator.py)
before it can reach serving.

**What this run establishes** is a runtime result, not a learning result: the
harness's per-turn timeout, not the policy, produced the wall of empty sessions
that the ungated 600 s run showed. **What it does not establish** is that the
stream learns the task — it does not.

## The finding: a turn budget calibrated for GPUs silently fails a slower runtime

The reference harness kills a `hermes chat` turn at `TURN_TIMEOUT_S`. A killed
turn is not reported as a timeout: the benchmark records the dead turn as a
`no-reply` failure, which then feeds the training signal. On this runtime a turn
takes 397 s at the median, against a 600 s default — about 1.5× of headroom.

The two runs below differ in that one constant and nothing else:

| per-turn ceiling | sessions | no-reply | replied | accepts |
| --- | --- | --- | --- | --- |
| 600 s (the default) | 72 | **28** | 44 | 0 |
| 1200 s | 72 | **3** | 69 | 2 |

One-sided Fisher, 28/72 against 3/72: **p = 1.6e-07**.

The independent check is the gate's own probe. At each training step it samples
the same weights on eight pinned problems, and `answered_rate` was **1.0 at
every step of both runs** — the policy never lost the ability to answer. The
empty sessions were generation outrunning the clock.

The fix is `OPENCLAWRL_TURN_TIMEOUT_S`, which defaults to 600 so the reference
path is unchanged; raise it on a slower runtime. The measurement lives in
[the runtime notes](../mlx-runtime-notes.md).

## What the gate did

`gate.csv` holds every step. Over 67 optimizer steps the gate **selected 20 and
held out 47**: the probe's clean rate climbed from 0.125 to a peak, fell back,
and every regressing step past that peak stayed out of serving. When the policy
recovered to a new high the gate readmitted it, so this is best-checkpoint
selection made online rather than a one-way ratchet.

That is the gate behaving as designed. It does not show that gating improves
the stream: no ungated Hermes run of the same configuration was made, so the
comparison that would settle it has not been run.

## What did not happen

| | |
| --- | --- |
| `sessions_to_adaptation` | not reached |
| accepts | 2 of 72 (s061, s066) |
| rejected for markdown | 67 |
| probe `gold_rate` | 0.125–0.375 throughout, no trend |

Two accepts against the previous run's zero is not a result — it is not
distinguishable from zero at this sample size. The stream did not adapt.

The shape of the failure is consistent across both runs and worth recording:
the probe's clean rate does move (the policy learns to drop markdown) while
`gold_rate` never does. The probe scores style only — it excludes the
gold-answer check by construction — whereas a session is accepted only when the
reply is clean **and** shows its work **and** carries the answer. Optimising the
first does not carry the other two, and mid-run the policy answered in as few as
86 characters: clean, and empty of work.

One accepted reply (s061) opens with a tool's own diff output, `┊ review diff
a//…/homework/61.txt → b//…`, and still passed with no violations. That is a gap
in the criterion, not a win.

## How it was run

A host-native `reef serve` on the machine's LAN address, the 72 harbor tasks
through `reef-eval stream` with `harness:HermesStreamAgent`, the PRM's votes
from the service's own OpenRouter credential, and the student persona through a
local key-injecting proxy — `student_server.py` sends no `Authorization` header,
so it cannot reach a hosted endpoint directly.

Three operational notes, each of which cost a discarded run:

- **The artifact repository must live outside the checkout.** `serve.yaml`'s
  `artifact_repository`, `artifact_work_dir` and `artifact_cache_dir` are
  relative, so two runs launched from the same directory share one repository,
  and the second is refused with `training is already bound to scenario`.
- **The stream's identity covers more than `--name`.** Adding `--budget` between
  a probe run and the full run forks the stream, and the new one's artifact
  pushes all fail against the scenario the service already bound.
- **The two timeouts are coupled.** Raising the per-turn ceiling without raising
  Harbor's session budget just moves which one fires.

## Files

- `curve.csv`, `learning_curve.png` — the 72 sessions, from [`learning_curve.py`](../learning_curve.py).
- `gate.csv` — per-step gate outcome, probe clean rate, running best, `answered_rate`, `gold_rate`.
- `serve.yaml` — the deployment. The run used the configuration schema of its date; the file carries the same values in the schema the repository loads today.
- [runtime notes](../mlx-runtime-notes.md) — topology, training metrics, the adapter artifact format, the capacity envelope, and the serving-latency measurement above.
