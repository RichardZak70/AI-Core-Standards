#!/usr/bin/env python3
"""Run consolidated health checks for AI-Core-Standards repositories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

from audit_data_layout import audit as audit_data_layout
from audit_tooling import audit as audit_tooling
from prompt_extract import extract_prompts


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail"
    details: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConsolidatedReport:
    target: str
    checks: List[CheckResult]

    @property
    def passed(self) -> bool:
        return all(check.status == "pass" for check in self.checks)

    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "passed": self.passed,
            "checks": [c.to_json() for c in self.checks],
        }


def run_checks(target_root: Path) -> ConsolidatedReport:
    checks: list[CheckResult] = []

    tooling_result = audit_tooling(target_root)
    checks.append(
        CheckResult(
            name="tooling",
            status="pass" if tooling_result.is_compliant else "fail",
            details=tooling_result.to_json(),
        )
    )

    data_result = audit_data_layout(target_root)
    checks.append(
        CheckResult(
            name="data_layout",
            status="pass" if data_result.is_compliant else "fail",
            details=data_result.to_json(),
        )
    )

    prompt_result = extract_prompts(target_root)
    checks.append(
        CheckResult(
            name="prompt_extract",
            status="pass",  # informational only
            details={
                "prompt_count": len(prompt_result.prompts),
                "target": prompt_result.target,
            },
        )
    )

    return ConsolidatedReport(target=str(target_root), checks=checks)


def print_human(report: ConsolidatedReport) -> None:
    print(f"Running AI-Core consolidated checks for: {report.target}\n")
    for check in report.checks:
        mark = "✅" if check.status == "pass" else "❌"
        print(f"{mark} {check.name}")
    print()
    print("Overall:", "PASS" if report.passed else "FAIL")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run consolidated AI-Core-Standards health check.")
    parser.add_argument("--target-root", type=Path, default=Path("."), help="Path to target repo root")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output")
    parser.add_argument("--report", type=Path, help="Where to write the consolidated report (JSON)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    target_root = args.target_root.resolve()

    report = run_checks(target_root)
    payload = json.dumps(report.to_json(), indent=2)

    if args.json:
        print(payload)
    else:
        print_human(report)

    if args.report:
        args.report.write_text(payload + "\n", encoding="utf-8")

    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
