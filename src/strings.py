"""Extract one string or boolean parameter."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from src.llm_protocol import LLMModel
from src.classes import ParameterDefinition
from src.encode import encode_text_to_ids


def format_args(values: Mapping[str, object]) -> str:
    """Prepare the extracted args for the prompt"""

    parts = (f'{key}="{value}"'
             if isinstance(value, str)
             else f"{key}={value}"
             for key, value in values.items())

    return ", ".join(parts) + (", " if values else "")


def make_value_prompt(
    request: str,
    function: str,
    param: str,
    values: Mapping[str, object],
) -> str:
    """Create the function-call completion prompt."""
    return f'-> {request}\n{function}({format_args(values)}{param}="'


def is_escaped(text: str, pos: int) -> bool:
    """Return True when a quote at pos is escaped."""
    backslashes = len(text[:pos]) - len(text[:pos].rstrip("\\"))
    return backslashes % 2 == 1


def closed_text(text: str) -> str | None:
    """Return content before the first non-escaped closing quote."""
    text = text.lstrip()
    start = int(text.startswith('"'))

    for pos in range(start, len(text)):
        if text[pos] == '"' and not is_escaped(text, pos):
            return text[start:pos]

    return None


def extract_value(
    model: LLMModel,
    request: str,
    param_name: str,
    param_def: ParameterDefinition,
    values: Mapping[str, object] | None = None,
    max_steps: int = 64,
    function: str = "",
) -> str:
    """Extract one schema-compatible non-numeric value."""
    if param_def.type.lower() in {"boolean", "bool"}:
        return extract_bool(model, request, param_name)

    prompt = make_value_prompt(request, function, param_name, values or {})
    prompt_ids = encode_text_to_ids(model, prompt)
    generated: list[int] = []

    for _ in range(max_steps):
        logits = np.asarray(
            model.get_logits_from_input_ids(prompt_ids + generated))
        generated.append(int(np.argmax(logits)))
        text = model.decode(generated)
        value = closed_text(text)

        if value is not None:
            if value:
                return value
            raise ValueError(f"No string was extracted for {param_name!r}.")

        if "\n" in text:
            value = text.split("\n", 1)[0].strip().strip('"')
            if value:
                return value
            raise ValueError(f"No string was extracted for {param_name!r}.")

    raise ValueError(
        f"String extraction reached "
        f"its step limit for parameter {param_name!r}.")


def extract_bool(llm: LLMModel, request: str, param_name: str) -> str:
    """Select exactly one valid JSON boolean literal."""

    prompt = (
        "Extract one boolean value. /no_think\n"
        "Answer with exactly True or False.\n"
        f"Parameter: {param_name}\n"
        f"Request: {request}\n"
        "Value: ")
    prompt_ids = encode_text_to_ids(llm, prompt)
    choices = {
        word: encode_text_to_ids(llm, word) for word in ("True", "False")}
    generated: list[int] = []

    logits = np.asarray(llm.get_logits_from_input_ids(
            prompt_ids + generated))
    allowed_tokens = {ids[0] for ids in choices.values()}
    hidden = np.full_like(logits, -np.inf)
    for token in allowed_tokens:
        hidden[token] = logits[token]

    token_index = int(np.argmax(hidden))
    for name, ids in choices.items():
        if token_index in ids:
            return name

    raise ValueError(f"No boolean was extracted for {param_name!r}.")
