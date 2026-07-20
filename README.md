<div align="center">

# LoopGEPA

**Your coding agent re-ran `npm install` 40 times and you didn't notice.**
LoopGEPA sees *why* an agent loop keeps failing and proposes a reversible fix you approve.

`Works with` **Claude Code** · **Codex CLI** · **Cursor**
&nbsp;·&nbsp; `Local-only` · `No telemetry` · `Apache-2.0`

[![CI](https://github.com/loopgepa/loopgepa/actions/workflows/ci.yml/badge.svg)](https://github.com/loopgepa/loopgepa/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/loopgepa)](https://pypi.org/project/loopgepa/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

</div>

> [!NOTE]
> **Pre-alpha, developed in the open once it's solid.** The spec, safety model,
> and architecture are locked (`IMPLEMENTATION.md`, `DECISIONS.md`); the core is
> being implemented milestone by milestone. Not yet published.

---

## What it does

Coding agents fail the *same way* over and over — repairs that repeat the
original mistake, verifiers that pass broken patches, retries that never help.
LoopGEPA runs your repeatable agent loops, records **clean local traces**, and:

- **Diagnoses** the recurring failure mode from real trace evidence.
- **Proposes** a small, reversible patch to the loop's skills, verifier rubric,
  retry policy, or stop condition — with a diff and a confidence level.
- **Never writes without your approval.** Promotion is dry-run by default and
  always reversible.

It is an MCP server first — you drive it from inside Claude Code, Codex, or
Cursor.

## Why LoopGEPA (vs the others)

| | LoopGEPA | Latitude / LangSmith / Braintrust / Langfuse |
|---|---|---|
| Runs | **On your machine** | Cloud SaaS |
| Data | **Never leaves your machine, no telemetry** | Uploaded to a backend |
| Scope | **Your local coding-agent sessions** (Claude Code / Codex) | Production LLM apps |
| Output | **Reversible patch you approve** | Dashboards / issue→PR in the cloud |

The name nods to **GEPA** (Genetic-Pareto reflective evolution — ICLR 2026 Oral,
`gepa-ai/gepa`): the research *direction*. v0.1 does the trace-driven diagnosis
and reflective mutation; measured GEPA-style optimization arrives in v0.2 by
wrapping the official GEPA package. We do not call anything "optimized" before
it's evaluated. See `DECISIONS.md`.

## Install

> Requires [`uv`](https://astral.sh/uv). If `uvx` isn't found:
> `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Claude Code**
```bash
claude mcp add loopgepa -- uvx loopgepa
```

**Codex CLI**
```bash
codex mcp add loopgepa -- uvx loopgepa
```

**Cursor / any MCP client** — add to your `mcp.json`:
```json
{ "mcpServers": { "loopgepa": { "command": "uvx", "args": ["loopgepa"] } } }
```

No `pip install`, no virtualenv — `uvx` runs it from PyPI in a cached ephemeral
environment.

## How it works

```text
declare loop ──▶ LoopGEPA runner executes it ──▶ clean trace (local)
                                                      │
             failure diagnosis ◀── aggregate traces ◀─┘
                     │
             reflective mutation (≤3 genes) ──▶ diff preview ──▶ you approve ──▶ reversible promote
```

Because LoopGEPA *runs* the loop, every knob it tunes — skill, verifier, retry
policy, stop condition — is real, and every trace is clean by construction
(no fragile session-hook guessing). See `IMPLEMENTATION.md` §7.

## Trust model

- **Local-only.** stdio MCP server; no sockets, no cloud, no telemetry.
- **Summaries, not secrets.** Allowlist capture — step roles, statuses, tool
  names, exit codes, paths, counts. Raw I/O is opt-in per loop.
- **Outside your repo.** Traces live in `$XDG_DATA_HOME/loopgepa`, never
  committed by accident.
- **You approve every write.** Dry-run by default; every promotion is reversible.

Full threat model: `SECURITY.md`.

## Development

**80 tests**, all local and offline. Full gate mirrors CI:

```bash
uv sync
bash scripts/dev_check.sh     # ruff + format + mypy strict + pytest
uv run python scripts/verify_stdio.py   # prove `uvx loopgepa` serves MCP over stdio
```

Contributing: `CONTRIBUTING.md` · Decisions: `DECISIONS.md` · Roadmap: `ROADMAP.md`

## License

Apache-2.0 — see `LICENSE`.
