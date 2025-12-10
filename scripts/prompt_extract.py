#!/usr/bin/env python3
"""Placeholder prompt extraction audit.

Scans source files for inline prompts and emits a report. Currently a stub to be
replaced with real extraction logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract inline prompts into config/prompts.yaml (stub).")
    parser.add_argument("--target-root", type=Path, default=Path("."), help="Path to target repo root")
    parser.add_argument("--report", type=Path, help="Where to write the extraction report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target_root = args.target_root.resolve()
    msg = f"TODO: implement prompt extraction audit for {target_root}"
    print(msg)
    if args.report:
        args.report.write_text(msg + "\n", encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
