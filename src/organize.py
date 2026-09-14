"""Combine function, number, and string extraction stages."""
# strict=True so it raises errors while zipping data
from __future__ import annotations
from collections.abc import Sequence

import math
from pydantic import BaseModel, ConfigDict, Field

from src.function_names import pick_function
from src.llm_protocol import LLMModel
from src.classes import (FunctionCallResult, FunctionDefinition,
                        PromptInput)
from src.numbers import extract_numbers
from src.strings import extract_value


NUMERIC_TYPES = {"number", "float", "double", "integer", "int"}
INT_TYPES = {"integer", "int"}


class CallMeMaybe(BaseModel):
    """Translate natural-language prompts into function calls."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: LLMModel
    functions: Sequence[FunctionDefinition] = Field(min_length=1)

    def generate_prompt(self, prompt: PromptInput) -> FunctionCallResult:
        """Process one prompt through a pipeline."""

        fn_name = pick_function(self.model, prompt.prompt,
                                self.functions)
        fnn_def: list[FunctionDefinition] = [
            fn for fn in self.functions if fn.name == fn_name]

        if not fnn_def:
            raise ValueError(f"Unknown function: {fn_name}.")
        fn_def: FunctionDefinition = fnn_def[0]

        numeric = [(name, definition)
                    for name, definition in fn_def.parameters.items()
                    if definition.type.lower() in NUMERIC_TYPES]
        num_values = extract_numbers(self.model, prompt.prompt,
                                    numeric)

        params: dict[str, object] = {}

        for (name, definition), value in zip(
            numeric, num_values, strict=True):
            if not math.isfinite(value):
                raise ValueError(f"Parameter Error {name}")

            if definition.type.lower() in INT_TYPES:
                if not value.is_integer():
                    raise ValueError(f"Parameter {name} must be an int")
                params[name] = int(value)
            else:
                params[name] = float(value)

        for name, definition in fn_def.parameters.items():
            if definition.type.lower() in NUMERIC_TYPES:
                continue

            params[name] = extract_value(
                model=self.model,
                request=prompt.prompt,
                function=fn_def.name,
                param_name=name,
                param_def=definition,
                values=params,
            )
        ordered_params = {name: params[name] for name in fn_def.parameters}
        return FunctionCallResult(
            prompt=prompt.prompt,
            name=fn_name,
            parameters=ordered_params
        )

    def generate_all_prompts(
        self,
        prompts: Sequence[PromptInput],
    ) -> list[FunctionCallResult]:
        """Process prompts in their original order."""
        return [self.generate_prompt(prompt) for prompt in prompts]
