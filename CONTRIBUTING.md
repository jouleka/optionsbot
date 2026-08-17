# Contributing

Thanks for helping improve `optionsbot`.

- Keep examples and tests paper-only.
- Never add credentials, tokens, account identifiers, production endpoints, databases, or personal infrastructure details.
- Open an issue before a large architectural change.
- Keep changes focused and add regression tests for changed behavior.
- Never present simulated or paper results as evidence of live profitability.

```bash
uv sync --locked --group dev
uv run --locked pytest
uv run --locked ruff check .
uv run --locked mypy src
```

Do not open a public issue for a suspected vulnerability. Follow [SECURITY.md](SECURITY.md).
