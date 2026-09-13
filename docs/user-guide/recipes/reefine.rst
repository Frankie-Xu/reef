Reefine
========

Reefine is the built-in recipe for refining a pi coding harness from plain
language instructions. Its implementation is
``reef.recipe.reefine:ReefineRecipe``, and its proposer and evaluator ship in
the Reef wheel, so the service needs no tutorial checkout or training GPUs.

Start the bundled profile with an OpenAI-compatible endpoint:

.. code:: bash

   reef serve --recipe reefine \
     --inference.upstream-url http://127.0.0.1:11434 \
     --inference.upstream-model gemma4:26b \
     --inference.upstream-api-key dummy

The profile listens on ``127.0.0.1:8901``, uses token ``reef-local``, and keeps
state under ``.reef/reefine/``. For custom deployments, copy
``reef/service/profiles/reefine.yaml`` and pass it with ``-c``. The
`Reefine tutorial <https://github.com/Human-Agent-Society/reef/tree/main/tutorials/reefine>`__
includes installation, bug-fix and research demos, and recorded measurements.

From the ask to the install
---------------------------

1. Ask. In a ``reef-pi`` session, ``/reef-harness <what it should do>`` has
   the model think the change through before anything is filed: when it
   triggers, what state the harness must know and how it learns it, what you
   must set up. When an open point would change what gets built, it asks you
   up to three questions, each with concrete options, then files your
   original words with the answers as clarifications; ``--direct`` as the
   first word files at once. From the shell, ``reef-pi harness "<text>"``
   posts the same training instruction to ``POST /reef/train``; add
   ``--wait`` to stay until the step settles. Either ask prints the
   request's page link (``GET /reef/harness/requests/<id>/page`` with the
   scenario and the token as query parameters): open it in a browser and it
   reloads every five seconds, naming the step's state, until the verdict is
   on it.
2. Step. In ``training-mode: manual`` the deployment runs one evolve step
   for each accepted instruction. The served model designs the change first
   (it restates the request, names what triggers the behavior and what state
   the harness must know and where each comes from, and lists what only you
   can provide), writes the entries, and reviews them against the request in
   a second call. The gate then runs the candidate on the health task.
3. Verdict. The session that asked reports the verdict when the step
   settles, and ``reef-pi harness ... --wait`` prints the same line: published
   as a release, ready but waiting for your review because it changes an
   extension, rejected by the gate, or skipped with the reason. Each names
   the next action, and the points the review left uncovered follow it.
   ``/reef-versions <step>`` and ``reef-pi page <step>`` show the step's page.
4. Promote and set up. A release that touches a ``code_extension`` waits as
   pending until ``/reef-versions <step> promote`` (or
   ``POST /reef/scenarios/{scenario}/promote``). When the release needs
   something from your machine, ``reef-pi setup`` lists the ``requires``
   items and runs a check only after you confirm it.
5. Install. Restart ``reef-pi``: the update notice offers the new head, and
   the install refuses a release whose requirements are not checked off.

Behavior and configuration
--------------------------

* ``training-mode: manual`` runs one step for each accepted instruction on
  ``POST /reef/train``. Use ``hybrid`` to also learn from failing reports.
* The served model proposes skills, rules, agent commands, or pi extensions.
  Requests and update notices are enabled in the seed by default.
* ``evolution.review_kinds: [code_extension]`` holds code changes pending
  human promotion. Client requirements must pass setup before installation.
* ``evolution.selection: floor`` is the default: the gate runs the candidate
  alone and publishes it when every task scores at least
  ``evolution.floor_score`` (``1.0``). The current release is not run, and an
  episode that could not run misses the floor.

The health floor
----------------

The profile's one gate task is a health check:

.. code:: yaml

   tasks:
     - '[health] Run the shell command `echo reef-ok` with your shell tool and reply with its exact output
       as a plain word alone on the last line.'

The bundled evaluator grades the reply's last line, ``reef-ok`` exactly. The
floor answers one question: does the tree still work after the change? The
model binding answers, the shell tool runs, every extension loads. It says
nothing about whether the change does what was asked; the step's design and
review notes and the person judge that. Set both ``evolution.tasks`` and
``evolution.evaluate`` for a workload of your own, and
``evolution.selection: score_comparison`` to require the candidate to beat
the current release on them instead.

What the step records
---------------------

Every step's catalog row carries the request under
``metrics.training_request`` and the proposer's notes under
``metrics.proposal_notes``; the step page
(``GET /reef/harness/releases/<step>/page``, ``reef-pi page <step>``) renders
them:

* ``design``: the proposer's plan for the request, a few sentences, as the
  page's Design section.
* ``review``: the second call's verdict, ``complete`` or ``partial``, with
  the points of the request the entries cover and the ones they leave
  uncovered, as the Review section; the verdict line in the session and
  from ``--wait`` names the uncovered points. Absent when the review call
  failed, which never blocks the step.
* ``refused_requires``: the ``requires`` items the proposer wrote that could
  not be honored, each with the reason, under "refused by the step" in the
  Setup section. An ``env`` item whose check is a shell test is brought to
  the one variable it names first, so ``test -n "$TOKEN"`` becomes the
  variable ``TOKEN`` rather than a refusal.
* ``undeclared_env``: the variables a written extension reads through
  ``process.env`` that no ``requires`` item names. Nothing adds them; the
  page shows them so you can set them or ask for the item.
* ``failure``: why a request step produced no change: the model call
  failed (how long it took, the reply budget and the endpoint's error; a
  reply without text adds that a thinking model may have spent the budget
  on its reasoning and names ``REEF_PROPOSER_MAX_TOKENS``) or the reply
  held no usable entry. The verdict line in the session and from ``--wait``
  quotes it, and the page shows it as ``proposer failure`` in the Verdict
  section.

All ``CordisRecipe`` evolution settings remain available, including custom
proposers, seeds, execution settings, and publication policies.

``REEF_PROPOSER_TIMEOUT_S`` and ``REEF_PROPOSER_MAX_TOKENS`` override the model
call budgets. Defaults are 120 seconds and 16384 reply tokens for an
instruction, 60 seconds and 8192 tokens for its review, and 60 seconds and
4096 tokens for failure-driven proposals; the reply budgets are sized for a
thinking model, which spends part of the budget on its reasoning before the
JSON. The tutorial's ``run.sh`` raises the timeout to 900 seconds for its
local model and pins the 16384 token budget.

Migration
---------

The former ``tutorials/harness-requests/`` directory is now
``tutorials/reefine/``. Existing runs can retain their state by moving their
``work/`` directory and keeping the old scenario name in the driver. The
``evolve-your-harness`` tutorial's proposer and evaluator entrypoints delegate
to Reefine, so its existing configurations continue to work.
