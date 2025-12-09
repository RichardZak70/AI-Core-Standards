import sys
from pathlib import Path

import pytest

# Ensure the scripts directory is on the path when running tests directly
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from audit_ai_project import audit, EXPECTED_DIRS, EXPECTED_FILES  # type: ignore[import-not-found]  # noqa: E402


def _create_dirs(base: Path, dirs: list[str]) -> None:
    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)


def _create_files(base: Path, files: list[str]) -> None:
    for f in files:
        path = base / f
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_audit_passes_when_all_required_items_exist(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _create_dirs(tmp_path, EXPECTED_DIRS)
    _create_files(tmp_path, EXPECTED_FILES)

    exit_code = audit(tmp_path)
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "Project matches core AI structure." in captured


def test_audit_reports_missing_dirs_and_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Create only a subset so we trigger missing directories and files
    _create_dirs(tmp_path, ["config", "docs"])
    _create_files(tmp_path, [])

    exit_code = audit(tmp_path)
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "Missing directories:" in captured
    for expected_dir in EXPECTED_DIRS:
        # Only two directories were created; the rest should be reported as missing
        if expected_dir not in {"config", "docs"}:
            assert expected_dir in captured
    assert "Missing files:" in captured
    for expected_file in EXPECTED_FILES:
        assert expected_file in captured


def test_audit_reports_missing_files_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # All directories exist, but files are absent
    _create_dirs(tmp_path, EXPECTED_DIRS)

    exit_code = audit(tmp_path)
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "Missing files:" in captured
    for expected_file in EXPECTED_FILES:
        assert expected_file in captured
