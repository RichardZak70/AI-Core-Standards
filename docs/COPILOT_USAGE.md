# Copilot Usage Standard

Copilot is a **code completion tool**, not a free-form reasoning engine.
To get consistent behavior:

- Use structured "Copilot Instruction" comments immediately above the code or in the relevant doc.
- Use different patterns for:
  - new code,
  - refactoring,
  - debugging,
  - documentation,
  - audit-driven remediation.

---

## New code

```python
# Copilot Instruction:
# Role: Senior engineer.
# Objective: Implement the function below.
# Constraints:
# - Follow project style (PEP 8 for Python).
# - Include type hints and a concise docstring.
# - No external dependencies unless necessary.
# Output: Only the function code.

def ...
```

## Debugging

```text
# Copilot Instruction:
# Debug this function.
# Error:
# {paste error}
# Requirements:
# - Identify root cause.
# - Propose minimal fix.
# - Output corrected function only.
```

## Refactoring

```text
# Copilot Instruction:
# Refactor for clarity and maintainability without changing external behavior.
# Keep: same function signature and return types.
# Improve: naming, structure, readability, duplication.
```

## Documentation

```text
# Copilot Instruction:
# Add concise docstrings and inline comments where helpful.
# Requirements:
# - Document inputs, outputs, side effects, and error cases.
# - Match project style (e.g., Google/NumPy docstrings for Python).
# - Do not change runtime behavior.
```

## Audit remediation

Use these when addressing findings from `audit_ai_project.py`, schema validation, or lint/type checks.

```text
# Copilot Instruction:
# Remediate the following audit finding.
# Finding source: {audit_tool}
# Finding description: {finding}
# Constraints:
# - Keep changes minimal and focused.
# - Preserve existing behavior unless the finding requires a change.
# - Add or update tests if behavior changes.
```

---

Every repo should either link to this doc or adapt it into a project-local `docs/COPILOT.md`.
