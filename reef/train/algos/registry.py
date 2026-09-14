"""Registries a method may fill at import time.

Step preparers register themselves via the ``@register_step_preparer`` class
decorator in their method package. This registry never imports a method
package: a method depends on the machinery, not the other way round. A
preparer may instead register its
:class:`StepPreparer` with :func:`register_preparer` from a module imported at
boot, or is named as a dotted ``"package.module:callable"`` reference wherever
a preparer is named (a recipe's ``training_spec().step_preparer``).
Resolving a dotted reference imports the module and wraps the callable in a
:class:`_CallableStepPreparer` if it is not already a :class:`StepPreparer`.

Loss families register here too, but only as a dotted ``"package.module:SPEC"``
reference (:func:`register_loss_family_ref`) keyed by the training backend
that implements it: the service process never loads a backend's side, and
each backend registry (``slime_backend/loss_families.py``,
``tinker_backend/losses.py``) imports its reference the first time the name
is resolved.
"""

from __future__ import annotations

import importlib

from reef.train.algos.base import StepPreparer, _CallableStepPreparer, _decorated_preparers, register_step_preparer

__all__ = [
    "StepPreparer",
    "loss_family_refs",
    "register_loss_family_ref",
    "register_preparer",
    "register_step_preparer",
    "resolve_preparer",
    "unregister_preparer",
]


class StepPreparerRegistry:
    """Resolves preparers registered by decorators or runtime callers.

    Decorated preparers are read live because a method package may be imported
    after this registry is constructed. Runtime registrations are tracked
    separately; unregistering removes whichever registration owns the name.
    """

    def __init__(self) -> None:
        self._external: dict[str, StepPreparer] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(set(_decorated_preparers) | set(self._external)))

    def register(self, preparer: StepPreparer) -> StepPreparer:
        """Register an external preparer under its ``name``."""
        if not isinstance(preparer, StepPreparer):
            raise TypeError(f"a step preparer registers a StepPreparer, got {type(preparer).__name__}")
        name = preparer.name
        existing = self._external.get(name)
        if existing is preparer:
            return preparer
        decorated = _decorated_preparers.get(name)
        if decorated is preparer:
            return preparer
        if decorated is not None or existing is not None:
            available = ", ".join(self.names)
            raise ValueError(f"step preparer {name!r} is already registered; registered preparers: {available}")
        self._external[name] = preparer
        return preparer

    def unregister(self, name: str) -> None:
        """Remove a preparer by name."""
        runtime = self._external.pop(name, None)
        decorated = _decorated_preparers.pop(name, None)
        if runtime is None and decorated is None:
            registered = ", ".join(self.names) or "none"
            raise KeyError(f"step preparer {name!r} is not registered; registered preparers: {registered}")

    def resolve(self, preparer_id: str) -> StepPreparer:
        """Resolve a registered preparer name or a dotted ``"module:callable"`` reference."""
        preparer = _decorated_preparers.get(preparer_id) or self._external.get(preparer_id)
        if preparer is not None:
            return preparer
        if ":" in preparer_id:
            return self._resolve_dotted(preparer_id)
        available = ", ".join(self.names)
        raise ValueError(f"unknown step preparer {preparer_id!r}; available preparers: {available}")

    def _resolve_dotted(self, reference: str) -> StepPreparer:
        """Import a dotted preparer reference, wrapping plain callables."""
        module_name, separator, attribute = reference.partition(":")
        if not separator or not module_name or not attribute:
            available = ", ".join(self.names)
            raise ValueError(f"unknown step preparer {reference!r}; available preparers: {available}")
        candidate = getattr(importlib.import_module(module_name), attribute)
        if isinstance(candidate, StepPreparer):
            return candidate
        if callable(candidate):
            return _CallableStepPreparer(candidate)
        raise TypeError(f"step preparer {reference!r} is not callable")


PREPARERS = StepPreparerRegistry()


def register_preparer(preparer: StepPreparer) -> StepPreparer:
    """Register an external preparer (module-level convenience)."""
    return PREPARERS.register(preparer)


def unregister_preparer(name: str) -> None:
    """Remove an external preparer registered by :func:`register_preparer`."""
    PREPARERS.unregister(name)


def resolve_preparer(preparer_id: str) -> StepPreparer:
    """Resolve a registered preparer name or a dotted ``"module:callable"`` reference."""
    return PREPARERS.resolve(preparer_id)


#: ``{backend: {loss_family: reference}}``; Slime is the default backend.
_loss_family_refs: dict[str, dict[str, str]] = {}
_DEFAULT_BACKEND = "slime"


def register_loss_family_ref(loss_family: str, reference: str, *, backend: str = _DEFAULT_BACKEND) -> None:
    """Record that ``backend`` implements ``loss_family`` at the dotted ``reference``.

    A method registers one reference per backend it supports; the same family
    name selects each backend's implementation.
    """
    if ":" not in reference:
        raise ValueError(f"loss family reference {reference!r} must be 'package.module:SPEC'")
    if not backend:
        raise ValueError("loss family references need a non-empty backend name")
    table = _loss_family_refs.setdefault(backend, {})
    existing = table.get(loss_family)
    if existing is not None and existing != reference:
        raise ValueError(f"loss family {loss_family!r} is already registered to {existing!r} for {backend}")
    table[loss_family] = reference


def loss_family_refs(backend: str = _DEFAULT_BACKEND) -> dict[str, str]:
    """The registered ``{loss_family: reference}`` table of one backend (a copy)."""
    return dict(_loss_family_refs.get(backend, {}))


def _unregister_loss_family_ref(loss_family: str, *, backend: str = _DEFAULT_BACKEND) -> bool:
    """Remove a registered family reference, returning whether it existed."""
    return _loss_family_refs.get(backend, {}).pop(loss_family, None) is not None
