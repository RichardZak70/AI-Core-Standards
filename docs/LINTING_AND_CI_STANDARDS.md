# Linting and CI Standards

Every project must:

- Have at least one linter + formatter appropriate to the language.
- Run tests + linting in CI on every push and pull request.

Language-specific tools (examples):

- Python: Ruff + mypy + pytest
- TypeScript: ESLint + Prettier + Jest
- C/C++: clang-tidy + clang-format + ctest

Each template repo is responsible for implementing concrete tooling.
This core doc defines the **requirement**, not the specific choice per language.
