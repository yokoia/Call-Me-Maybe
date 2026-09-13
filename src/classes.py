"""Pydantic schemas for inputs, definitions, and results."""

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


StrTypes = Literal[
    "string",
    "number",
    "float",
    "double",
    "integer",
    "int",
    "boolean",
    "bool",
]


class PromptInput(BaseModel):
    """One natural-language request."""

    model_config = ConfigDict(extra="forbid")
    prompt_text: str = Field(min_length=1)


class ParameterDefinition(BaseModel):
    """Definition of one function parameter."""

    model_config = ConfigDict(extra="forbid")
    type: StrTypes
    description: str | None = None


class ReturnDefinition(BaseModel):
    """Definition of a function return value."""

    model_config = ConfigDict(extra="forbid")
    type: StrTypes


class FunctionDefinition(BaseModel):
    """Definition of one available function."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition | None = None


class FunctionCallResult(BaseModel):
    """One generated function call."""

    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]
