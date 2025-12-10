#!/usr/bin/env python3
"""
Audit AI project structure against the RZ core standard.

Usage:
    python scripts/audit_ai_project.py              # audit current directory
    python scripts/audit_ai_project.py /path/to/repo
    python scripts/audit_ai_project.py --json
    python scripts/audit_ai_project.py --validate-configs
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


# Required directories for a compliant AI project
REQUIRED_DIRS: List[str] = [
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

# Required files for a compliant AI project
REQUIRED_FILES: List[str] = [
    "config/prompts.yaml",
    "config/models.yaml",
    "README.md",
]

# Optional/recommended items (warn if missing, but do not fail)
RECOMMENDED_FILES: List[str] = [
    ".gitignore",
    ".editorconfig",
    "docs/AI_PROMPTING_STANDARDS.md",
    "docs/COPILOT_USAGE.md",
]


@dataclass
class AuditResult:
    target: str
    missing_dirs: list[str]
    missing_files: list[str]
    missing_recommended: list[str]
    config_validation_passed: bool | None = None

    @property
    def is_compliant(self) -> bool:
        return not self.missing_dirs and not self.missing_files


def _find_missing(root: Path, expected: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for rel_path in expected:
        if not (root / rel_path).exists():
            missing.append(rel_path)
    return missing


def audit(path: Path) -> AuditResult:
    missing_dirs = _find_missing(path, REQUIRED_DIRS)
    missing_files = _find_missing(path, REQUIRED_FILES)
    missing_recommended = _find_missing(path, RECOMMENDED_FILES)

    return AuditResult(
        target=str(path),
        missing_dirs=missing_dirs,
        missing_files=missing_files,
        missing_recommended=missing_recommended,
    )


def _run_config_validation(root: Path) -> bool:
    """Invoke the AJV validation script; returns True on success."""

    script_path = root / "scripts" / "ajv-validate.mjs"
    if not script_path.exists():
        print("⚠️  Config validation skipped (scripts/ajv-validate.mjs not found).")
        return False

    result = subprocess.run(
        ["node", str(script_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)

    return result.returncode == 0


def _print_block(title: str, items: Iterable[str]) -> bool:
    items = list(items)
    if not items:
        return False
    print(title)
    for item in items:
        print(f"  - {item}")
    return True


def print_human(result: AuditResult) -> None:
    print(f"Auditing AI structure in: {result.target}\n")

    printed_any = _print_block("Missing required directories:", result.missing_dirs)
    if printed_any and result.missing_files:
        print()
    printed_any = _print_block("Missing required files:", result.missing_files) or printed_any
    if printed_any and result.missing_recommended:
        print()
    _print_block("Missing recommended items (not strictly required):", result.missing_recommended)

    if result.config_validation_passed is True:
        print("\n✅ Schema validation passed (AJV).")
    elif result.config_validation_passed is False:
        print("\n❌ Schema validation failed (see output above).")

    if result.is_compliant and (result.config_validation_passed in {True, None}):
        print("\n✅ Project matches core AI structure.")
    else:
        print("\n❌ Project does NOT match core AI structure.")
        print("Suggested fix: copy or adapt missing items from RZ-AI-Core-Standards/templates.")


def main(argv: list[str]) -> int:
    json_mode = False
    run_config_validate = False
    target_arg: str | None = None

    for arg in argv[1:]:
        if arg == "--json":
            json_mode = True
        elif arg == "--validate-configs":
            run_config_validate = True
        elif target_arg is None:
            target_arg = arg
        else:
            print(f"Unexpected argument: {arg}", file=sys.stderr)
            return 2

    target = Path(target_arg).resolve() if target_arg else Path(".").resolve()

    result = audit(target)

    if run_config_validate:
        result.config_validation_passed = _run_config_validation(target)
        # If schema validation fails, treat as non-compliant
        if result.config_validation_passed is False:
            result.missing_files.append("(schema validation failed)")

    if json_mode:
        print(json.dumps(asdict(result), indent=2))
    else:
        print_human(result)

    if not result.is_compliant:
        return 1
    if run_config_validate and result.config_validation_passed is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
