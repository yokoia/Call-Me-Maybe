"""Safely import the LLM class from sdk"""

try:
    from llm_sdk.llm_sdk import Small_LLM_Model

except KeyboardInterrupt:
    raise KeyboardInterrupt(" User interupt while importing Small_LLM_Model")

except ImportError:
    raise ImportError("Cannot import Small_LLM_Model from llm_sdk.")

__all__ = ["Small_LLM_Model"]
