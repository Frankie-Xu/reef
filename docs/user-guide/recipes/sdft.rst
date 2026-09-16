SDFT: learn from demonstrations without forgetting
==================================================

Self-Distillation Fine-Tuning (`arXiv:2601.19897
<https://arxiv.org/abs/2601.19897>`__) learns from demonstrations on policy.
The served model, shown a demonstration in its prompt, is the teacher; the
same model without it is the student; the student samples the response and
the loss pulls its next-token distributions toward the teacher's at every
position of that sample. Compared with supervised fine-tuning on the same
demonstrations, the paper reports higher new-task accuracy and far less
forgetting.

+-------------+------------------------------------------------------------+
| Evolves     | model weights                                              |
+-------------+------------------------------------------------------------+
| Signal      | one report with the demonstration as ``context`` per       |
|             | rollout                                                    |
+-------------+------------------------------------------------------------+
| Loss family | ``sdft``                                                   |
+-------------+------------------------------------------------------------+
| Package     | ``recipes/sdft/``                                          |
+-------------+------------------------------------------------------------+
| Processor   | reported feedback, singleton                               |
+-------------+------------------------------------------------------------+
| Needs       | GPUs, and a backend that captures tokens and log-probs     |
+-------------+------------------------------------------------------------+
| Example     | none yet                                                   |
+-------------+------------------------------------------------------------+

What it does
------------

The harness sends a request through Reef, obtains a demonstration of the
response from somewhere else (a reference solution, a stronger model), and
reports the demonstration against the request's receipt. Nothing is executed
for the demonstrator; the student's response is the one the environment saw.
With the default ``batch_size`` of 1, each report is one training step.

.. flow::
   :loop: the next request is served by the updated weights

   Rollout :: the student answers a request
   Demonstration :: a reference response for the same request
   Report :: the demonstration as ``context`` against the rollout's receipt
   Step* :: distil the demonstration-conditioned teacher on the student's own tokens
   Version :: publish the updated weights to the engine

How Reef implements it
----------------------

The processor turns every ``TeacherContextReport`` into one
``TrajectoryItem`` carrying the student's recorded tokens plus
``teacher_tokens``: the teacher's request rendered with the served model's
chat template (``tokenizer_path``), followed by the student's response ids
verbatim. A ``TeacherPromptBuilder`` composes that request from the
student's request and the report's context. The default adds the
demonstration block (``context_template``, whose default is the reference
implementation's wording) to the request's final user message, or as a new
user message when the request ends in a tool result. A harness whose
demonstrations need another layout, such as a native tool-call turn or a
block in the system prompt, ships its own builder and names it with
``teacher_prompt_builder``; the builder reads its settings from the recipe
config through ``from_config`` and returns the messages and tools the
teacher sees.

The ``sdft`` loss family runs on Slime as a ``custom_loss``. Before each
step, a pre-train hook runs one forward pass of the current actor over every
sample's teacher sequence and keeps the teacher's next-token distribution at
each response position. The loss then puts the student's distribution at the
same positions, from the training forward over the plain request, against
it. There is no second copy of the weights: with LoRA the teacher is the same
frozen base plus the same adapter, and only the prompt differs.

Per token, the loss is the KL over the full vocabulary. Forward KL
(teacher toward student, GKD-style) is the default, since the authors report
that the paper's results used it; reverse KL is a switch. Each sample
contributes the mean over its trained response tokens, weighted by the
reference's truncated importance-sampling ratio between the policy and the
rollout engine's log-probs, so SDFT requires an inference backend that
attaches engine-native tensors.

The report contract
-------------------

A report references one inference record and carries the demonstration as
``metadata.context``. A ``score`` is optional metadata; the recipe never
trains on it. The same contract serves SDPO, where the context is the
environment feedback instead of a demonstration.

.. code:: json

   {
     "references": ["<receipt of the student's request>"],
     "metadata": {"context": "<the demonstration>"}
   }

Configuration
-------------

.. config::

   batch_size | 1 | rollouts per optimizer step. Must equal the driver's ``--global-batch-size`` because each sample is its own data-parallel unit.
   tokenizer_path | required | the served model's tokenizer directory; it renders the teacher prompt with the chat template the engine applied.
   max_teacher_tokens | 0 | a report whose teacher sequence is longer is skipped and counted (``teacher_overflow_reports``); 0 disables the check. Set it to the trainer's window.
   context_template | the reference's block | the text the default builder adds to the final user message, with ``{context}`` as the demonstration's placeholder.
   teacher_prompt_builder | empty | ``package.module:Builder`` naming a ``TeacherPromptBuilder`` that composes the teacher's request instead of the default.
   max_staleness | 0 | accepted lag between the producing and serving version.

The Slime driver takes ``--loss-type custom_loss``,
``--use-rollout-logprobs`` and ``--disable-compute-advantages-and-returns``,
plus the family's own flags:

.. config::

   --sdft-kl-direction | forward | ``forward`` is KL(teacher || student), ``reverse`` is KL(student || teacher).
   --sdft-importance-sampling-cap | 2.0 | cap of the truncated importance-sampling weight; 0 disables the correction.
   --sdft-skip-response-tokens | 0 | response tokens at the start of every sample left out of the loss; the paper's runs used 3.

The teacher pass keeps one ``[response tokens, vocabulary / tensor parallel]``
float16 block per sample on the host between the pass and the step, about
2 GB for a 16k-token response of a 248k-vocabulary model at tensor parallel
4, so long-context deployments size ``max_teacher_tokens`` with that in mind.

Related guides
--------------

- `Inference and feedback quickstart <../../getting-started/quickstart.rst>`__:
  learn the request, receipt, and report workflow.
- `Train model weights from agent feedback <../evolve-your-model.rst>`__:
  set up the GPU stack and inspect published updates.
- `Loss families <../../developer-guide/loss-families.rst>`__: how a family
  such as ``sdft`` plugs into the Slime backend.
