## Summary

<!-- What does this change and why? -->

## Related issue

<!-- Closes #NNN -->

## Validation

- [ ] `uv run pytest` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` pass
- [ ] `uv run mypy src/whystuck` passes
- [ ] Behavior change is covered by a test written **before** the implementation (TDD)

## Safety

- [ ] No new telemetry, network calls, or non-local storage
- [ ] No write path bypasses explicit user approval
- [ ] `SECURITY.md` guarantees preserved (or updated with an ADR)
- [ ] If a decision changed, a new ADR entry was added to `DECISIONS.md`
