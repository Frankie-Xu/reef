"""Teacher prompts: the recorded request plus the report's context, in the served model's chat template.

The teacher and the student share weights, so the teacher prompt is the only
difference between their forward passes. A :class:`TeacherPromptBuilder`
composes it from the student's own request and the report's ``context``;
the default appends the context to the request's final user message, the
way the reference implementation does. A deployment whose harness needs
another composition (a demonstration rendered as a native tool-call turn, a
block inside the system prompt) names its own builder with the recipe's
``teacher_prompt_builder`` reference. The result is tokenized with the chat
template the serving engine applied, so the teacher sequence (prompt ids
plus the student's response ids verbatim) scores the student's own tokens.
"""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: The reference implementation's demonstration block (idanshen/Self-Distillation, ``main.py``).
DEFAULT_CONTEXT_TEMPLATE = (
    "This is an example for a response to the question:\n{context}\n\n"
    "Now answer with a response of your own, including the thinking process."
)
CONTEXT_PLACEHOLDER = "{context}"


class TeacherPromptTokenizer(ABC):
    """Render a chat request into the prompt token ids the served model sees."""

    @abstractmethod
    def prompt_token_ids(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None) -> list[int]:
        """The prompt ids of ``messages`` with the generation prompt appended."""


class ChatTemplateTokenizer(TeacherPromptTokenizer):
    """The served model's Hugging Face tokenizer applying its own chat template."""

    def __init__(self, tokenizer_path: str) -> None:
        # transformers belongs to the training environment; importing it here
        # keeps ``import recipes.sdft`` light for the service.
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def prompt_token_ids(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None) -> list[int]:
        rendered = self._tokenizer.apply_chat_template(
            list(messages), tools=list(tools) if tools else None, tokenize=False, add_generation_prompt=True
        )
        return [int(token) for token in self._tokenizer(rendered, add_special_tokens=False)["input_ids"]]


def recorded_request(payload: Mapping[str, Any]) -> tuple[list[Any], list[Any] | None]:
    """The messages and tools of a recorded inference request.

    The rollout backend retains its provider-neutral copy under
    ``response.training``; a record without it carries the request body.
    """
    response = payload.get("response")
    training = response.get("training") if isinstance(response, Mapping) else None
    if isinstance(training, Mapping) and isinstance(training.get("request_messages"), list):
        messages = list(training["request_messages"])
        tools = training.get("request_tools", payload.get("tools"))
    else:
        messages = list(payload.get("messages") or [])
        tools = payload.get("tools")
    return messages, list(tools) if isinstance(tools, list) and tools else None


def flatten_content(content: Any) -> str:
    """The plain text of an OpenAI-style message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, Mapping) and item.get("type") == "text"]
        return " ".join(parts) if parts else ""
    return str(content) if content is not None else ""


def normalize_messages_for_template(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Messages as a chat template expects them: text content, chat roles, tool arguments as objects."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        entry = dict(message)
        if entry.get("role") == "developer":
            entry["role"] = "system"
        content = entry.get("content")
        if content is not None and not isinstance(content, str):
            entry["content"] = flatten_content(content)
        if entry.get("tool_calls"):
            entry["tool_calls"] = [_normalize_tool_call(call) for call in entry["tool_calls"]]
        normalized.append(entry)
    return normalized


def _normalize_tool_call(call: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(call)
    function = normalized.get("function")
    if isinstance(function, Mapping):
        function = dict(function)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function["arguments"] = json.loads(arguments)
            except json.JSONDecodeError:
                function["arguments"] = {}
        normalized["function"] = function
    return normalized


@dataclass(frozen=True)
class TeacherRequest:
    """The chat request the teacher reads: messages and the tools offered with them."""

    messages: list[dict[str, Any]]
    tools: list[Any] | None = None


class TeacherPromptBuilder(ABC):
    """Compose the teacher's request from the student's request and the report's context.

    Subclasses own the composition policy and read their settings from the
    recipe's processor config in :meth:`from_config`. The default
    :class:`AppendContextBuilder` keeps the reference implementation's
    layout; a deployment names another with ``teacher_prompt_builder``.
    """

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TeacherPromptBuilder:
        """Build from the recipe's processor config; the default takes no settings."""
        return cls()

    @abstractmethod
    def build(
        self, request_messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None, context: str
    ) -> TeacherRequest:
        """The teacher's request for one recorded student request and its context."""


class AppendContextBuilder(TeacherPromptBuilder):
    """The reference layout: the context block added to the request's final user message.

    ``main.py`` of the reference implementation rewrites the question's user
    message with the demonstration appended. An agent request that ends in a
    tool result gets the block as a new user message instead, so the context
    still sits right before the response it conditions. The block is
    ``template`` with ``{context}`` replaced.
    """

    def __init__(self, template: str = DEFAULT_CONTEXT_TEMPLATE) -> None:
        if CONTEXT_PLACEHOLDER not in template:
            raise ValueError(f"context_template must contain {CONTEXT_PLACEHOLDER}")
        self._template = template

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> AppendContextBuilder:
        return cls(str(config.get("context_template", DEFAULT_CONTEXT_TEMPLATE)))

    def build(
        self, request_messages: Sequence[Mapping[str, Any]], tools: Sequence[Any] | None, context: str
    ) -> TeacherRequest:
        block = self._template.replace(CONTEXT_PLACEHOLDER, context)
        messages = normalize_messages_for_template(request_messages)
        if messages and messages[-1].get("role") == "user":
            last = dict(messages[-1])
            content = str(last.get("content") or "")
            last["content"] = f"{content}\n\n{block}" if content else block
            messages = [*messages[:-1], last]
        else:
            messages = [*messages, {"role": "user", "content": block}]
        return TeacherRequest(messages, None if tools is None else list(tools))


def resolve_teacher_prompt_builder(config: Mapping[str, Any]) -> TeacherPromptBuilder:
    """The builder the processor config names, or the default when ``teacher_prompt_builder`` is empty.

    A reference is ``package.module:Builder``, naming a
    :class:`TeacherPromptBuilder` subclass (built through ``from_config``)
    or an instance.
    """
    reference = str(config.get("teacher_prompt_builder", "")).strip()
    if not reference:
        return AppendContextBuilder.from_config(config)
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"teacher_prompt_builder {reference!r} must be 'package.module:Builder'")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot import teacher_prompt_builder {reference!r}: {exc}") from exc
    if isinstance(candidate, type) and issubclass(candidate, TeacherPromptBuilder):
        candidate = candidate.from_config(config)
    if not isinstance(candidate, TeacherPromptBuilder):
        raise TypeError(f"teacher_prompt_builder {reference!r} must name a TeacherPromptBuilder subclass or instance")
    return candidate
