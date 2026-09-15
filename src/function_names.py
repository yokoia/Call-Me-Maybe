"""Extract only one function name with constrained decoding."""

from __future__ import annotations

import re
from collections.abc import Sequence
from src.llm_protocol import LLMModel
import numpy as np  # gives tools to manipulate arrays easly

from src.classes import FunctionDefinition
from src.encode import encode_text_to_ids


def make_function_prompt(userprompt: str,
                         functions: Sequence[FunctionDefinition]) -> str:
    """Create the prompt used only for function selection."""

    choices = "\n".join(
        f"- {fn.name}: {fn.description}" for fn in functions
    )
    return (
        "You extract a function name. /no_think\n"
        "Choose the single function that matches the request.\n"
        "If none matches, answer with __NO_MATCH__.\n"
        "Answer with one exact function name from the list.\n"
        "Do not explain. Finish with a newline.\n\n"
        f"Functions:\n{choices}\n\n"
        f"Request: {userprompt}\n"
        "Function name: "
    )


def unknown_function(userprompt: str,
                     functions: Sequence[FunctionDefinition]):
    """Function checking"""
    prompt_words = set(re.findall(r"[a-z0-9]+", userprompt.lower()))
    function_words = set()

    for fn in functions:
        text = f"{fn.name} {fn.description}".lower()
        words = re.findall(r"[a-z0-9]+", text)

        for word in words:
            if len(word) > 2 and word != "function":
                function_words.add(word)

    check = prompt_words & function_words
    if not check:
        raise ValueError(
            f"Prompt {userprompt} does not match any known function.")


def pick_function(llm: LLMModel, userprompt: str,
                  functions: Sequence[FunctionDefinition]) -> str:
    """Choose one exact function name with constrained decoding."""
    if not functions:
        raise ValueError("No function definitions were provided.")

    unknown_function(userprompt, functions)
    prompt_ids = encode_text_to_ids(
                llm, make_function_prompt(userprompt, functions))
    funcs_ids: dict[str, list[int]] = {
        fn.name: encode_text_to_ids(llm, f"{fn.name}\n")
        for fn in functions}
    funcs_ids["__NO_MATCH__"] = encode_text_to_ids(llm, "__NO_MATCH__\n")

    generated: list[int] = []
    max_len = max([len(ids) for ids in funcs_ids.values()])

    for _ in range(max_len):
        active = [
            ids for ids in funcs_ids.values()
            if ids[:len(generated)] == generated
            and len(ids) > len(generated)]

        if not active:  # extra safety check
            break

        logits = np.asarray(
                llm.get_logits_from_input_ids(prompt_ids + generated))
        allowed_tokens = {ids[len(generated)] for ids in active}
        hidden = np.full_like(logits, -np.inf)

        for token in allowed_tokens:
            hidden[token] = logits[token]

        token_index = int(np.argmax(hidden))
        generated.append(token_index)

        for name, ids in funcs_ids.items():
            if ids == generated:
                if name == "__NO_MATCH__":
                    raise ValueError("__NO_MATCH__:"
                                     "No function matches the request.")
                return name

    raise ValueError(
        "Function-name extraction did not produce a valid choice.")
