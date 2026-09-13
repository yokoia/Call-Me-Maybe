"""Extract one string or boolean parameter."""

from __future__ import annotations
import numpy as np
from collections.abc import Mapping

from src.llm_protocol import LLMModel
from src.classes import ParameterDefinition
from src.encode import encode_text_to_ids


def args_prompt(values: Mapping[str, object]) -> str:
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
    ex_values: Mapping[str, object]) -> str:
    """Create the function-call prompt."""

    return f'{request}\n{function}({args_prompt(ex_values)}{param}="'


def text_finished(textt: str) -> str | None:
    """Stopping once Qwen returns a quote"""

    text = textt.lstrip()  # "   john " == "john "

    if text.startswith('"'):
        text = text[1:]
    pos = text.find('"')
    if pos == -1:
        return None
    return text[:pos]

def extract_value(
    llm: LLMModel,
    request: str,
    param_name: str,
    param_def: ParameterDefinition,
    values: Mapping[str, object] | None = None,
    max_steps: int = 64,
    function: str = "") -> str:
    """Extract one schema-compatible non-numeric value."""

    if param_def.type.lower() in {"boolean", "bool"}:
        return extract_bool(llm, request, param_name)

    prompt = make_value_prompt(request, function, param_name,
                               values or {})
    prompt_ids = encode_text_to_ids(llm, prompt)
    generated: list[int] = []

    for _ in range(max_steps):
        logits = np.asarray(llm.get_logits_from_input_ids(
                prompt_ids + generated))
        generated.append(int(np.argmax(logits)))
        text = llm.decode(generated)
        value = text_finished(text)

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
        f"String extraction reached its step limit for parameter "
        f"{param_name!r}.")

def extract_bool(llm: LLMModel, request: str, param_name: str) -> str:
    """Select exactly one valid JSON boolean literal."""

    prompt = (
        "Extract one boolean value. /no_think\n"
        "Answer with exactly True or False.\n"
        f"Parameter: {param_name}\n"
        f"Request: {request}\n"
        "Value: ")
    prompt_ids = encode_text_to_ids(llm, prompt)
    choices = {word: encode_text_to_ids(llm, word) for word in ("True", "False")}
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
