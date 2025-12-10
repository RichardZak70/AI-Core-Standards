#!/usr/bin/env python3
"""Audit data/ layout and output traceability for AI projects."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_DIRS: list[str] = [
    "data",
    "data/raw",
    "data/processed",
    "data/prompts",
    "data/outputs",
    "data/cache",
    "data/embeddings",
]

# Derive allowed top-level subdirs in data/ from REQUIRED_DIRS to avoid duplication.
ALLOWED_TOP_LEVEL_IN_DATA: set[str] = {
    Path(p).name for p in REQUIRED_DIRS if p != "data"
}

# Files that are allowed directly under data/ (common housekeeping)
ALLOWED_TOP_LEVEL_FILES_IN_DATA: set[str] = {".gitkeep", ".gitignore", "README.md"}

OUTPUT_METADATA_KEYS: list[str] = ["run_id", "model", "prompt_id", "timestamp"]


@dataclass
class DataAuditResult:
    target: str
    missing_dirs: list[str]
    stray_items: list[str]
    metadata_issues: list[str]

    @property
    def is_compliant(self) -> bool:
        return not self.missing_dirs and not self.stray_items and not self.metadata_issues

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["is_compliant"] = self.is_compliant
        return payload


def _find_missing(root: Path, expected: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for rel in expected:
        if not (root / rel).exists():
            missing.append(rel)
    return missing


def _find_stray_items(data_root: Path) -> list[str]:
    """Find unexpected files/dirs directly under data/."""
    stray: list[str] = []
    if not data_root.exists():
        return stray

    for child in data_root.iterdir():
        rel = str(child.relative_to(data_root.parent))  # e.g. "data/foo"
        name = child.name

        if child.is_dir():
            if name not in ALLOWED_TOP_LEVEL_IN_DATA:
                stray.append(rel)
        else:
            if name not in ALLOWED_TOP_LEVEL_FILES_IN_DATA:
                stray.append(rel)

    return stray


def _check_output_metadata(outputs_root: Path, max_files: int | None = None) -> list[str]:
    """Check JSON files under data/outputs for required metadata keys.

    max_files: if provided, only scan up to this many files to avoid huge runs.
    """
    issues: list[str] = []
    if not outputs_root.exists():
        return issues

    count = 0
    for path in outputs_root.rglob("*.json"):
        if max_files is not None and count >= max_files:
            issues.append(
                f"{outputs_root}: metadata check truncated at {max_files} files; "
                "consider running without --max-output-files for full coverage."
            )
            break

        count += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{path}: failed to parse JSON ({exc})")
            continue

        if not isinstance(data, dict):
            issues.append(f"{path}: expected top-level JSON object with metadata")
            continue

        missing = [key for key in OUTPUT_METADATA_KEYS if key not in data]
        if missing:
            issues.append(f"{path}: missing metadata keys: {', '.join(missing)}")

    return issues


def audit(target_root: Path, max_output_files: int | None = None) -> DataAuditResult:
    missing_dirs = _find_missing(target_root, REQUIRED_DIRS)
    stray_items = _find_stray_items(target_root / "data")
    metadata_issues = _check_output_metadata(
        target_root / "data" / "outputs",
        max_files=max_output_files,
    )
    return DataAuditResult(
        target=str(target_root),
        missing_dirs=missing_dirs,
        stray_items=stray_items,
        metadata_issues=metadata_issues,
    )


def print_human(result: DataAuditResult) -> None:
    print(f"Auditing data layout in: {result.target}\n")

    if result.missing_dirs:
        print("Missing required directories:")
        for item in result.missing_dirs:
            print(f"  - {item}")
        print()

    if result.stray_items:
        print("Unexpected files/directories directly under data/:")
        for item in result.stray_items:
            print(f"  - {item}")
        print()

    if result.metadata_issues:
        print("Output metadata issues (data/outputs/**/*.json):")
        for item in result.metadata_issues:
            print(f"  - {item}")
        print()

    if result.is_compliant:
        print("✅ Data layout and outputs look compliant.")
    else:
        print("❌ Data layout issues detected. See above for details.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit data layout and traceability.")
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("."),
        help="Path to target repo root (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Where to write the JSON audit report",
    )
    parser.add_argument(
        "--max-output-files",
        type=int,
        default=None,
        help="Max number of JSON files under data/outputs to scan (for huge repos)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target_root = args.target_root.resolve()

    result = audit(target_root, max_output_files=args.max_output_files)
    json_payload = json.dumps(result.to_json(), indent=2)

    if args.json:
        print(json_payload)
    else:
        print_human(result)

    if args.report:
        args.report.write_text(json_payload + "\n", encoding="utf-8")

    return 0 if result.is_compliant else 1


if __name__ == "__main__":
    raise SystemExit(main())
