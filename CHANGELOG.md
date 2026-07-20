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
- **`whystuck setup` wizard** (`whystuck.setup`, also the `whystuck-setup` script):
  one command configures Claude Code / Codex / Cursor — merges each client's MCP
  config (never clobbering other servers) and injects a usage-contract rule
  (`CLAUDE.md` / Cursor `.mdc` / `AGENTS.md`) so the agent auto-diagnoses stuck
  loops. Idempotent; `--all`/`--global`/`--from-source`/`--yes` aware. No new
  prerequisite — reuses the `uv` you already have.
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
- 161 tests (TDD, offline); repo_worker (output→apply→verify), dependency-review + release-rehearsal workflows, ADR index; ruff + mypy strict; CI, CodeQL, Scorecard, secret
  scan, and mkdocs-material docs site. Locked architecture in `DECISIONS.md`
  (ADR-001…010); threat model in `SECURITY.md`.

### Changed
- `propose_fix` returns the server-computed confidence per proposal (was only
  persisted on the variant, unreachable by any tool).
- Path-escape refusals raise `PathEscapeError` (a `WhyStuckError`) instead of a
  bare `ValueError`; a malformed benchmark task file names the offending file.
- Shared helpers deduped across the tree (`utc_now_iso`, `mean`, subprocess
  capture, text-gene prompt rendering, temp-file staging); tests share factories
  and a fixed timestamp via `tests/conftest.py`.

### Fixed
- **First-promotion rollback.** The first promotion snapshots a genesis variant,
  so its rollback has a real prior state to re-promote (previously impossible).
- **`unlock_gene` mutation.** The explicit unlock every safety refusal points to
  is now a real mutation type — no more hand-editing YAML to clear a lock.
- **Honest confidence.** Citing only unknown/fabricated trace ids now forces LOW;
  raw trace volume alone no longer grants MEDIUM/HIGH.
- **Actionable errors.** The LLM worker's missing-dependency error names the real
  fix (`pip install litellm`); `list_traces` rejects a negative `limit`.
- Docs corrected: the diagnosis confidence example (15 traces → genuinely HIGH),
  the already-shipped LLM/repo workers (no longer "remaining"), the optimize
  story label, a skipped skill step, and a from-source install path pre-launch.

_Nothing is released yet — the repo stays local until the launch gate
(`LAUNCH_CHECKLIST.md`) is fully green._
