# Schemas and Validation

All AI projects must enforce schemas on:

- LLM outputs used by downstream systems.
- Any structured data ingested from external sources.

## Requirements

- Define a clear output schema (JSON schema, Pydantic model, TS interface, etc.).
- Validate LLM output against this schema before use.
- Reject or handle invalid outputs explicitly (no silent failures).

Example (Python / Pydantic):

```python
from pydantic import BaseModel


class Extraction(BaseModel):
    name: str | None
    email: str | None
    phone: str | None
```

Prompt must instruct the model to return a JSON object matching the schema.
Validation must run before data is stored or further processed.
