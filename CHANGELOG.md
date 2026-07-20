# Changelog

All notable changes to LoopGEPA are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning:
[SemVer](https://semver.org/).

## [Unreleased]

### Added
- **MVP complete (local).** Single `loopgepa` package installable via
  `uvx loopgepa`; FastMCP stdio server for Claude Code / Codex / Cursor.
- Core (`loopgepa.core`): models, XDG-scoped file store, deterministic diagnosis
  (failure-mode clustering + confidence), command+verifier runner, mutation +
  diff, safety guards (≤3-gene cap, no silent budget increase), reversible
  promotion with dry-run + rollback records.
- 8 MCP tools: record_trace, run_loop, summarize_traces, infer_failure_modes,
  propose_mutations, list_variants, promote_variant, save_loop.
- Claude Code plugin (thin): explain-loop-failure / optimize-loop / design-loop
  skills + 3 agents. Flagship `examples/bugfix-loop` with a runnable offline demo.
- 80 tests (TDD, offline); ruff + mypy strict; CI, CodeQL, Scorecard, secret
  scan, and mkdocs-material docs site. Locked architecture in `DECISIONS.md`
  (ADR-001…010); threat model in `SECURITY.md`.

_Nothing is released yet — the repo stays local until the launch gate
(`LAUNCH_CHECKLIST.md`) is fully green._
