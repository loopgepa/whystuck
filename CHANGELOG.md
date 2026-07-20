# Changelog

All notable changes to WhyStuck are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning:
[SemVer](https://semver.org/).

## [Unreleased]

### Added
- **MVP complete (local).** Single `whystuck` package installable via
  `uvx whystuck`; FastMCP stdio server for Claude Code / Codex / Cursor.
- Core (`whystuck.core`): models, XDG-scoped file store, deterministic diagnosis
  (failure-mode clustering + confidence), command+verifier runner, mutation +
  diff, safety guards (≤3-gene cap, no silent budget increase), reversible
  promotion with dry-run + rollback records.
- 6 focused MCP tools: diagnose, run_loop, propose_fix, promote, save_loop, and
  the internal record_trace.
- Claude Code plugin (thin): explain-loop-failure / optimize-loop / design-loop
  skills + 3 agents. Flagship `examples/bugfix-loop` with a runnable offline demo.
- v0.2-alpha eval layer (opt-in `whystuck[eval]`): objectives scalarizer, benchmark
  from traces/dir, loop-genome↔GEPA adapter, a GEPA-wrapping optimizer that saves
  the winner as a variant + Pareto frontier, and an LLM-backed worker so text-gene
  changes actually move the score. Optimize/frontier MCP tools register only when
  the extra is installed (default surface stays 6).
- PostToolUse thrash hook (`plugin/hooks/thrash_watch.py`, zero-dep, fail-open):
  auto-detects repeated edits, captures a session trace, and nudges the agent to
  diagnose — no need to remember to invoke WhyStuck.
- Claude Code marketplace manifest (`.claude-plugin/marketplace.json`).
- 131 tests (TDD, offline); ruff + mypy strict; CI, CodeQL, Scorecard, secret
  scan, and mkdocs-material docs site. Locked architecture in `DECISIONS.md`
  (ADR-001…010); threat model in `SECURITY.md`.

_Nothing is released yet — the repo stays local until the launch gate
(`LAUNCH_CHECKLIST.md`) is fully green._
