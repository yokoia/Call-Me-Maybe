"""Run the complete Call Me Maybe pipeline."""

from __future__ import annotations
import sys
from src.cli import make_parser


def main() -> int:
    """Load files, run the model, and create the output JSON."""

    from src.organize import CallMeMaybe
    from src.read_files import read_functions, read_prompts, output_results
    from src.import_sdk import Small_LLM_Model

    args = make_parser().parse_args()
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Error: Program interrupted by user.", file=sys.stderr)
        sys.exit(1)
    except (
        OSError,
        RuntimeError,
        ValueError,
        PermissionError,
        ImportError
    ) as e:
        print(f"Ecrror: {e}", file=sys.stderr)
        sys.exit(1)
