"""Small helper for LLM method encode()."""

from __future__ import annotations

from src.llm_protocol import LLMModel


def encode_text_to_ids(llm: LLMModel, text: str) -> list[int]:
    """Return a real integer list."""

    ids = llm.encode(text).squeeze().tolist()

    if isinstance(ids, int):
        return [ids]
    return [int(value) for value in ids]
