"""Run the complete Call Me Maybe pipeline."""

from __future__ import annotations

import sys

from src.organize import CallMeMaybe
from src.cli import make_parser
from src.read_files import read_functions, read_prompts, output_results
from src.import_sdk import Small_LLM_Model


def main() -> int:
    """Load files, run the model, and create the output JSON."""
    args = make_parser().parse_args()

    try:
        functions = read_functions(args.functions_definition)
        prompts = read_prompts(args.input)
        model = Small_LLM_Model(
            model_name=args.model,
            device=args.device,
        )
        caller = CallMeMaybe(model=model, functions=functions)
        results = caller.generate_all_prompts(prompts)
        output_results(args.output, results)
        print(f"Created: {args.output}")
        return 0
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyboardInterrupt,
        PermissionError,
    ) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
