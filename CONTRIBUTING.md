# Contributing to LoopGEPA

Thanks for your interest. LoopGEPA is pre-alpha; the shape is still moving, so
open an issue to discuss anything non-trivial before a large PR.

## Ground rules

- **Read `DECISIONS.md` first.** It records why the architecture is what it is.
  If you want to change a decision, propose a new ADR entry — don't quietly
  work around one.
- **TDD for behavior.** Write the failing test first, verify it fails, then the
  minimal implementation. Tests are close to exhaustive for models, storage,
  mutation, runner, and safety behavior.
- **No dead code.** Every line must be reachable and load-bearing. No "just in
  case" params, flags, or fallbacks.
- **Safety is not negotiable.** No change may weaken the guarantees in
  `SECURITY.md`. No telemetry. No network by default.

## Local setup

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src/loopgepa
```

## Pre-PR gate

`uv run pytest` and `uv run ruff check .` must pass with zero failures before
you push. CI enforces the same. Do not open a PR with known failures.

## Commit / PR

- Small, focused branches — one concern per branch.
- Conventional-ish commit subjects (`feat:`, `fix:`, `docs:`, `test:`).
- PRs target `main` and get a review before merge.
