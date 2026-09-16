# sdft

Reproduction of [Self-Distillation Fine-Tuning](https://arxiv.org/abs/2601.19897) as a Reef weight-training recipe. SDFT learns from demonstrations on policy: the served model reads a demonstration in its prompt and is the teacher, the same model without it is the student, and the loss is the per-token KL between their next-token distributions along the student's own sample. The package holds the method; the report contract it defines, a rollout's receipt plus the teacher's `context`, is shared with SDPO.

- Paper: [arXiv:2601.19897](https://arxiv.org/abs/2601.19897)
- Reference implementation: [idanshen/Self-Distillation](https://github.com/idanshen/Self-Distillation) at `d77573212fa0`; the recipe's processor, wire row and loss family map onto its `main.py` (the demonstration prompt) and `distil_trainer.py` (the loss). Forward KL is the default, as the authors' 2026-04-07 note says the paper's results used it; reverse KL is a switch.
- Pins: `slime` pinned to `THUDM/slime@41014d1f29e201137fdffce737bb8bac65bc5219` (via `pyproject.toml` `dependency-groups.runtime`)
- Claim scope: none yet. The paper's Science Q&A result (Table 5) and the CEO-Bench comparison are the roadmap's next items ([#502](https://github.com/Human-Agent-Society/reef/issues/502)).

## Layout

```text
sdft/
  recipe.py          SDFTRecipe: training spec, loss family "sdft", the report contract
  report.py          TeacherContextReport: a rollout's receipt and the text its teacher sees
  teacher_prompt.py  TeacherPromptBuilder: the teacher's request from the recorded one and the context, in the chat template
  processor.py       reported feedback, singleton: one rollout with its demonstration is one unit
  objective.py       selects the sdft loss; the recipe binds the per-sample step schedule
  slime/             the training-plane objective: the self-teacher forward pass and the token KL
```

## Where the rest is documented

[The sdft recipe page](../../docs/user-guide/recipes/sdft.rst) covers the report contract, configuration and the driver flags, and [Loss families](../../docs/developer-guide/loss-families.rst) describes how the family plugs into the Slime backend.
