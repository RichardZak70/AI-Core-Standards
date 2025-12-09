#!/usr/bin/env python3
"""Audit AI project structure against the RZ core standard."""

from __future__ import annotations

import sys
from pathlib import Path

EXPECTED_DIRS = [
    "config",
    "data",
    "data/raw",
    "data/processed",
    "data/prompts",
    "data/outputs",
    "data/cache",
    "data/embeddings",
    "docs",
]

EXPECTED_FILES = [
    "config/prompts.yaml",
    "config/models.yaml",
    "README.md",
]


def audit(path: Path) -> int:
    print(f"Auditing AI structure in: {path}\n")

    missing_dirs = [d for d in EXPECTED_DIRS if not (path / d).exists()]
    missing_files = [f for f in EXPECTED_FILES if not (path / f).exists()]

    if missing_dirs:
        print("Missing directories:")
        for directory in missing_dirs:
            print(f"  - {directory}")

    if missing_files:
        print("\nMissing files:")
        for file_path in missing_files:
            print(f"  - {file_path}")

    if not missing_dirs and not missing_files:
        print("Project matches core AI structure.")
        return 0

    print("\nProject does NOT match core AI structure.")
    print("Suggested fix: copy missing items from RZ-AI-Core-Standards/templates.")
    return 1


if __name__ == "__main__":
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
    sys.exit(audit(target))
