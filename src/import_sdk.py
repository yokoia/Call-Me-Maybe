"""Safely import the LLM class from sdk"""

try:
    from llm_sdk.llm_sdk import Small_LLM_Model
except ImportError as e:
    raise ImportError("Cannot import Small_LLM_Model from llm_sdk.") from e

__all__ = ["Small_LLM_Model"]
