"""SPADE (arXiv:2608.19197): self play in adaptive synthetic executable environments, one method, one package.

Reef knows one task format, Harbor; everything SPADE needs beyond it lives here. The Designer writes a
Harbor task, an instruction, a container, a verifier and a reference solution, and every task is one any
Harbor agent can play.

- ``designer``: the Environment Designer's adversarial prompt, its reply parsed, and the experience the prompt
  carries.
- ``harbor``: the written task under the team's structural gate, its hash, a split per generation, and Harbor's
  oracle check.
- ``generation``: one generation end to end: the Designer proposes through Reef, the oracle check refuses, the
  task is written, the solver plays both arms through the task player, regret splits, the manifest and the
  report are written, and each proposal is reported against the Designer's receipt.

The training side follows.
"""

from recipes.beta.spade.designer import (
    DesignerReplyError,
    DesignerRequest,
    HarborReply,
    PlayRecord,
    designer_messages,
    designer_prompt,
    parse_harbor_reply,
)
from recipes.beta.spade.generation import (
    Checks,
    Designer,
    Generation,
    GenerationError,
    GenerationRequest,
    GenerationResult,
    RealChecks,
    ReefDesigner,
    ReefSolver,
    Solver,
    load_experience,
)
from recipes.beta.spade.harbor import (
    GeneratedHarborTask,
    OracleResult,
    content_hash,
    harbor_task,
    oracle_check,
    reply_errors,
    split_generation,
)

__all__ = [
    "Checks",
    "Designer",
    "DesignerReplyError",
    "DesignerRequest",
    "GeneratedHarborTask",
    "Generation",
    "GenerationError",
    "GenerationRequest",
    "GenerationResult",
    "HarborReply",
    "OracleResult",
    "PlayRecord",
    "RealChecks",
    "ReefDesigner",
    "ReefSolver",
    "Solver",
    "content_hash",
    "designer_messages",
    "designer_prompt",
    "harbor_task",
    "load_experience",
    "oracle_check",
    "parse_harbor_reply",
    "reply_errors",
    "split_generation",
]
