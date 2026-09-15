"""Read and write project JSON files safely."""

from __future__ import annotations

import json
from typing import Any
from pydantic import ValidationError

from pathlib import Path
from src.classes import FunctionCallResult, FunctionDefinition, PromptInput


def read_json(path: Path) -> Any:
    """Read JSON file, return python data"""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Input file was not found: {path}")
    except PermissionError:
        raise ValueError(f"Permission denied while reading: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {path} at line {e.lineno}, column {e.colno}.")


def read_functions(path: Path) -> list[FunctionDefinition]:
    """Read functions JSON file"""
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")

    try:
        return [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as e:
        raise ValueError(
            f"Invalid function definition in {path}: {e}")


def read_prompts(path: Path) -> list[PromptInput]:
    """Read prompt JSON file"""
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array.")

    try:
        return [PromptInput.model_validate(item) for item in data]
    except ValidationError as e:
        raise ValueError(f"Invalid prompt in {path}: {e}")


def output_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write results as valid formatted JSON."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [item.model_dump(mode="json") for item in results]
        json_text = json.dumps(
            data, ensure_ascii=False, indent=2, allow_nan=False)

        with path.open("w", encoding="utf-8") as file:
            file.write(json_text + "\n")
    except OSError as e:
        raise ValueError(f"Could not write output file {path}: {e}")
