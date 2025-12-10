#!/usr/bin/env python3
"""Placeholder prompt merging utility.

Merges core, template, and project prompt sets and validates the result. Stub
implementation to be replaced with real merge logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge prompts from core/template/project (stub).")
    parser.add_argument("--core", type=Path, help="Path to core prompts.yaml")
    parser.add_argument("--template", type=Path, help="Path to template prompts.defaults.yaml")
    parser.add_argument("--project", type=Path, help="Path to project prompts.custom.yaml")
    parser.add_argument("--output", type=Path, default=Path("config/prompts.merged.yaml"), help="Output merged prompts path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    msg = "TODO: implement prompt merging and validation"
    print(msg)
    if args.output:
        args.output.write_text("# " + msg + "\n", encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
