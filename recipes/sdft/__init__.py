"""Self-Distillation Fine-Tuning (arXiv:2601.19897): one method, one package.

- ``recipe`` — the SDFT recipe class and its ``WeightTrainingSpec``. Its report
  contract is the shared :class:`reef.core.reports.TeacherContextReport`: a
  rollout's receipt and its teacher's ``context``.
- ``teacher_prompt`` — the teacher prompt built from the recorded request and that context.
- ``processor`` — one rollout with its demonstration, one batch unit.
- ``objective`` — the backend-agnostic training objective.
- ``slime`` — the Slime loss family (self-teacher forward pass, per-token KL)
  and its torch objective. Imported by the training driver and workers only;
  this package's public surface never loads it.
"""

from recipes.sdft.objective import SdftObjective
from recipes.sdft.processor import SDFTProcessor
from recipes.sdft.recipe import SDFTRecipe
from reef.train.algos.registry import register_loss_family_ref

register_loss_family_ref("sdft", "recipes.sdft.slime:SdftAlgorithm")

__all__ = ["SDFTProcessor", "SDFTRecipe", "SdftObjective"]
