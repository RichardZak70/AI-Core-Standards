# AI Core Standards Project Status

This document tracks the status of the RZ AI Core Standards repository.

## 1. Repository Scaffolding

- [x] Initialize `AI-Core-Standards` repo and `main` branch.
- [x] Add root `README.md` describing purpose and layout.
- [x] Add `LICENSE` (MIT).
- [x] Create core folders: `docs/`, `templates/`, `scripts/`.

## 2. Core Documentation

- [x] `docs/PROJECT_STRUCTURE.md` – RZ AI Project Structure Standard.
- [x] `docs/AI_PROMPTING_STANDARDS.md` – prompt storage and structure rules.
- [x] `docs/COPILOT_USAGE.md` – Copilot usage standard and patterns.
- [x] `docs/DATA_ORGANIZATION.md` – data directories and traceability rules.
- [x] `docs/SCHEMAS_AND_VALIDATION.md` – schema and validation requirements.
- [x] `docs/LINTING_AND_CI_STANDARDS.md` – linting/CI expectations.
- [x] `docs/AI_PROJECT_REVIEW_CHECKLIST.md` – production readiness checklist.
- [x] `docs/STATUS.md` – status and roadmap.

## 3. Templates

- [x] `templates/config/prompts.yaml` – base `summarization` and `code_debug` prompts.
- [x] `templates/config/models.yaml` – default provider/model catalog.
- [x] `templates/data_layout.txt` – standard `data/` folder layout.
- [x] `templates/.editorconfig` – editor consistency rules.
- [x] `templates/.gitignore_ai_project` – ignore patterns for AI projects.

## 4. Tooling and Scripts

- [x] `scripts/audit_ai_project.py` – checks presence of required config/data/docs.
- [x] `scripts/migrate_prompts_from_code.py` – extracts inline prompts from Python into YAML.
- [x] Add tests for `audit_ai_project.py` in a sample project (optional).

## 5. Linting and Quality

- [x] Configure markdownlint for this repo (config file + npm dependency).
- [x] Run markdownlint cleanly on all `.md` files.
- [x] Add a simple CI workflow to enforce markdownlint on pull requests.

---

### How to run markdownlint locally (once configured)

From the repo root:

```powershell
npm install --save-dev markdownlint-cli
npx markdownlint-cli "docs/**/*.md"
```

## 6. Current Assessment

- Documentation: All standards listed above are present in `docs/` and align with the repository overview in `README.md`.
- Templates: Configuration, editor, and data layout templates are available under `templates/` and match the documented expectations.
- Tooling: Validation and migration scripts exist in `scripts/`, and CI covers markdown, YAML, Ajv validation, and PowerShell linting.
- Tests: A test suite for `audit_ai_project.py` exists, satisfying the optional checklist item.

## 7. Recommended Next Steps

- Add a short "Quickstart" guide that explains how to copy templates into a new project and run `scripts/audit_ai_project.py` for an initial compliance check.
- Extend script coverage by documenting usage examples and adding lightweight tests for `migrate_prompts_from_code.py` and `validate_config.py` to keep automation in step with the standards.
- Provide a minimal sample project skeleton (using the templates and schemas) to demonstrate end-to-end alignment with the standards and to serve as a baseline for future template work.
- When additional audit scripts (e.g., `audit_llm_usage.py`, `audit_data_layout.py`, `audit_tooling.py`, `audit_docs.py`, `rz_ai_check.py`) are added, update `scripts/fix_audit_findings.py` and `tests/test_fix_audit_findings.py` to run and assert their presence in the sequence.
- TODO: Finalize new audit stubs (`prompt_extract.py`, `prompt_merge.py`, `audit_llm_usage.py`, `audit_data_layout.py`, `audit_tooling.py`, `audit_docs.py`, `rz_ai_check.py`) and then wire their real logic into `scripts/fix_audit_findings.py` plus accompanying tests.
