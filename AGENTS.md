# pydantic-ai-app-factory

## Module layout

Public modules are the API: `deps.py`, `feature.py`, `errors.py` (the contract features and
products import) and `app.py` (the build surface products call). Every other module is a private
implementation detail and underscore-prefixed; nothing outside this package may import one.

## Commands

```bash
make install      # uv sync
make format       # ruff format (rewrites files)
make format-check # ruff format --check
make lint         # ruff check
make typecheck    # pyright strict
make test         # pytest
make testcov      # pytest with 100% branch coverage
make all          # lint + format-check + typecheck + testcov
```

Run `make all` before every commit.
