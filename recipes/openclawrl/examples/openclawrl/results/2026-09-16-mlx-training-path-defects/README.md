# Three defects in the MLX training path, found by running it

This is a defect record, not a learning result. Running OpenClaw-RL's GSM8K
stream through the merged MLX backend turned up three things the deployment
asked for and the run did not do. Two earlier MLX runs were kept under
`results/` and have been withdrawn: they carry all three, so their numbers were
never properties of the method.

The reference result for this recipe is still the seven-GPU run in
[the example README](../../README.md#gsm8k-homework-stream), which is unaffected —
it does not take the MLX path.

## 1. `kl_coef` never reached the objective

`MLXRuntime._run_training` dispatches on loss family. The `openclawrl` branch
read `self._openclawrl["kl_coef"]`, which defaulted to zero; the deployment's
`kl_coef` was spent only on the *other* branch, as TTT-Discover-style advantage
shaping. `recipes/openclawrl/recipe.py` always names the `openclawrl` family, so
the config key was dead on the only path this recipe takes.

The tell was in the runs' own records: step metrics reported `kl_coef 0.0` for a
deployment configured with `0.05`. Both withdrawn runs trained with no KL term
at all, and one of them named that term among its load-bearing settings.

Fixed by defaulting the objective's coefficient to the deployment's, so it
reaches whichever mechanism the active family carries — the openclawrl loss has
its own KL term, as the slime reference's `kl_loss_coef` does. The Tinker
backend already worked this way: one key, routed to the resolved loss adapter,
and a family with no KL raises rather than ignoring it.

## 2. The style probe stated the rule it was scoring

The candidate gate's probe asked for a worked answer and then added *"do not use
bold, headings, bullet points, or numbered lists"* — a rule the student persona
never states up front. The probe was therefore measuring compliance with its own
instruction, not what the policy had internalised: a model told not to use
bullets can score well on it while opening every session with them.

Uncoached, the baseline falls from 0.125 to **0.0**, which is the number that
agrees with the 67-of-72 markdown rejection rate the withdrawn gated run
recorded at the session level.

The probe also scored truncation rather than style. At `max_tokens: 96` this
model's "here's a thinking process" preamble consumed the budget before the
arithmetic: `no-shown-work` fired on **50 of 50** offline probe replies at 96
against **13 of 50** at 320. At 320 the replies reach a median 1056 characters,
against the 770 a real session's first reply runs to.

Measured offline against base weights and four saved candidates — 100
generations, every reply scored — the corrected probe shows a hard floor rather
than a resolution problem: all three style markers appear in **100 of 100**
replies, base and candidates alike. The evaluator now also reports
`mean_violations`, the distance to the criterion, which can move while
`clean_rate` is pinned at zero.

## 3. `session-ttl-s: 45` discarded about half the training pairs

The method trains on a reply bound to the student reaction that judges it.
`SessionIndex` retires a session after its idle TTL and `ingest` expires before
observing, so an expired session can never bind the two — and Hermes runs each
turn as a fresh process starting from `[system, user]`, so the
`x-reef-tag-session` tag is the only thing carrying that link. Once the session
is gone nothing recovers it.

Every MLX config said 45 against a documented default of 900. That holds on the
reference's timing, where a reply comes back in seconds. Here a single
generation takes over two minutes and a training step blocks the engine longer
still.

Two independent measurements agree at about half:

| | |
| --- | --- |
| consecutive requests further apart than 45s | 15 of 31 (48%) |
| sessions expiring unbound vs binding | 70 : 69 (50%) |

The second figure swings widely while a run is in flight — sessions flush in
bursts, and it read 60:15 at one point and 12:9 at another. The gap
distribution is the stabler measure, and the settled ratio agrees with it.

Raised to 1800, which covers a generation, the student's round trip and a
training step in between. The cost is that a finished session waits that long
before its last turn retires, delaying a step; the short value silently dropped
the data instead.

## What the run with fixes 1 and 2 shows

`gate.csv` and `sessions.csv` hold the first 21 sessions of a stream run with
the KL term active and the probe corrected, but **still on the 45s TTL** — so it
is not the experiment that answers whether the method learns here. It is
recorded because it is the evidence the first two fixes hold, and because its
flat probe is what the third defect predicts.

| | |
| --- | --- |
| sessions | 21 |
| optimizer steps | 25 |
| `kl_coef` in step metrics | `0.05` on every step |
| probe `clean_rate` | 0.000 throughout |
| probe `mean_violations` | 3.38 → 3.38 (range 3.12–3.38) |
| probe `answered_rate` | 1.00 throughout |
| gate | 25 selected, 0 held out |
| accepts | 3 (s006, s016, s020) |
| sessions with no turn at all | 1 |

Twenty-five optimizer steps move the weights substantially and move the probe
not at all. Adapter displacement from base grows monotonically — `||lora_b||`
from 0.84 to 8.43 over 24 steps, read straight from the safetensors — so the
updates are real and are not being pinned back by the newly active KL term.
That rules out the reading that `kl_coef: 0.05` is too strong.

**The three accepts are not evidence of adaptation.** At 3/21 against the
withdrawn runs' pooled 2/144 a one-sided Fisher test gives p = 0.015, and
against the closer of the two alone p = 0.074. Both are unreliable here for a
reason visible in `sessions.csv`: **all three accepts are 2-turn sessions**,
the shortest in the run, while every non-accepting session ran 3 to 7 turns.
That pattern fits "a subset of problems is easy and the base model answers them
cleanly first try" at least as well as it fits a changed policy — and the probe,
which is a fixed problem set and immune to problem-difficulty confounds, has not
moved at all. The control that would separate them is the base model on those
same three problems, which has not been run.

The gate selecting 25 of 25 is also not a result. With `clean_rate` pinned at
the floor the running best is 0.0, so every candidate reads "within margin of
best" and is selected. The gate is correctly implemented and has no signal to
act on until the rate lifts off the floor.

## Files

- `gate.csv` — per step: loss, `kl_coef`, adapter delta, probe metrics, gate outcome.
- `sessions.csv` — per session: reward, turns, violation count.
- [runtime notes](../mlx-runtime-notes.md) — topology, training metrics, capacity.
