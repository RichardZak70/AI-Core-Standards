# Data Organization Standard

Goal: Every output must be traceable: `input → prompt → model → output → timestamp`.

## Directories

- `data/raw/` – immutable source data. Never modified in place.
- `data/processed/` – cleaned / transformed data, reproducible from raw.
- `data/prompts/` – stored prompt variants, AB tests, experiment logs.
- `data/outputs/` – ALL LLM outputs (including error logs) saved with timestamps.
- `data/cache/` – cached LLM responses keyed by input + prompt + model hash.
- `data/embeddings/` – embedding vectors + metadata and index files.

## Rules

- Outputs must be append-only (no overwrite in place).
- Every batch run must write a new file with timestamp and model information.
- Embeddings from different models or dimensions must be separated and versioned.
