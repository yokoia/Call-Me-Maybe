"""Constrained decoding for a `number` extraction parameter."""

from __future__ import annotations
from collections.abc import Sequence

import re
import numpy as np

from src.llm_protocol import LLMModel
from src.classes import ParameterDefinition
from src.encode import encode_text_to_ids


INT_TYPES = {"integer", "int"}
# the rule of what an int looks like
NUMBER_RULE = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)")


def force_one_number(llm: LLMModel,
                     userprompt: str,
                     param_name: str,
                     param_def: ParameterDefinition,
                     raw_number: str) -> float:
    """Generate one exact number using constrained decoding."""

    prompt = (
        "Extract one numeric parameter. /no_think\n"
        "Return only its value followed by a comma.\n"
        f"Parameter: {param_name}\n"
        f"Description: {param_def.description or 'numeric value'}\n"
        f"Request: {userprompt}\n"
        f"{param_name} = "
    )
    prompt_ids = encode_text_to_ids(llm, prompt)
    generated: list[int] = []
    number_ids = encode_text_to_ids(llm, f"{raw_number}")

    for token in number_ids:
        logits = np.asarray(
            llm.get_logits_from_input_ids(prompt_ids + generated)
        )
        hidden = np.full_like(logits, -np.inf)
        hidden[token] = logits[token]
        token_index = int(np.argmax(hidden))
        generated.append(token_index)

    if generated != number_ids:
        raise ValueError(
            f"Could not extract number for parameter {param_name!r}")

    value = float(raw_number)
    if param_def.type.lower() in INT_TYPES and not value.is_integer():
        raise ValueError(f"Parameter {param_name} requires an integer")

    return value


def extract_numbers(
    llm: LLMModel,
    userprompt: str,
    params: Sequence[tuple[str, ParameterDefinition]],
) -> list[float]:
    """Extract all numbers in a prompt"""

    numbers = NUMBER_RULE.findall(userprompt)

    if len(numbers) < len(params):
        raise ValueError("The request does not contain enough numeric values")

    return [
        force_one_number(llm, userprompt, name, definition, raw)
        for (name, definition), raw in zip(params, numbers)
    ]
