"""Command-line argument handling."""

from __future__ import annotations

import argparse
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Convert natural-language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the available function definitions.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to natural-language prompts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to the generated result file.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, or mps. Default: automatic.",
    )
    return parser
