# Contributing

## Principles

- `src/quintropy/` is the only active implementation. Do not add version
  numbers to package modules or public interfaces.
- Treat published `results/baselines/` directories as immutable artifacts. A
  new experiment must create a new directory and identify its method and
  evaluation period in the manifest and README.
- Never describe the repeatedly inspected development holdout as independent
  out-of-sample evidence.
- Preserve the separation between answer candidates and permitted guess
  actions, and enforce temporal cutoffs at every training boundary.

## Development workflow

```bash
python -m pip install -e ".[dev]"
python -m quintropy --help
python -m ruff format --check src benchmarks tests
python -m ruff check src benchmarks tests
python benchmarks/verify_wordlists.py
python benchmarks/verify_checksums.py
python benchmarks/audit_experiment.py results/baselines/editorial_regime_linguistic_tail_2026-05-01_2026-08-12
python -m pytest
```

Keep changes focused, add regression tests for changed behavior, and update
the README whenever a public experimental contract changes. Do
not commit local experiment output, caches, generated models, credentials, or
environment-specific files.

## Pull-request checklist

- Tests, integrity checks, and relevant experiment audits pass.
- New inputs and published artifacts have documented provenance and checksums.
- Scientific claims match the evidence status and uncertainty intervals.
- Public text, identifiers, errors, and comments are in English.
- The diff contains no secrets, temporary files, or unrelated formatting.
