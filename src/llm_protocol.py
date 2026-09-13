"""Shared public model interface used by all extractors."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMModel(Protocol):
    """Public methods required from Small_LLM_Model."""

    def encode(self, text: str) -> Any:
        """Encode text into token IDs."""

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs into text."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return raw next-token logits."""
