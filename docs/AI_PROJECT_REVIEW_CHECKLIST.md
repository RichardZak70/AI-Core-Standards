# AI Project Review Checklist

Before calling any project "production-ready," confirm:

- [ ] `config/prompts.yaml` exists and is actually used.
- [ ] Data directories follow the standard (`data/raw`, `processed`, `outputs`, etc.).
- [ ] There is a clear schema for all structured LLM outputs.
- [ ] LLM calls are wrapped (no raw scattered API calls).
- [ ] Linting and tests run in CI.
- [ ] Prompts used in production are explicitly identified and versioned.
- [ ] There is a README explaining the LLM usage and data flow.
