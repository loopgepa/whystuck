# LoopGEPA

**Trace-driven diagnosis and reversible improvement of Claude Code agent loops.**

Prompt optimization treats the *prompt* as the unit of improvement. LoopGEPA
treats the whole **agent loop** as the unit: skills, agents, verifier rubrics,
retry policies, context policies, and stop conditions — the machinery around the
model, not just the model's instructions.

LoopGEPA runs your declared loops, records clean execution traces, tells you
*why* a loop keeps failing, and proposes small, reversible patches you approve
before anything is written.

```text
Before:  bugfix-loop  success 8/20
         common failure: repair repeats the original mistake after tests fail

Proposed patch (you approve before it applies):
  + stronger verifier rubric
  + failure-summary-before-repair rule
  + retry policy 1 -> 2
```

## What this is — and is not

LoopGEPA is **not** a magic self-improving agent. v0.1 is deliberately
conservative: it **diagnoses** recurring failures from real traces, **proposes**
reversible patches, and **asks before promoting** them. Nothing is auto-applied.
Nothing leaves your machine.

The name nods to **GEPA** (Genetic-Pareto reflective evolution — ICLR 2026;
`gepa-ai/gepa`). That is the research *direction*. v0.1 does not yet run the
measured evaluation that would earn the word "optimized"; that lands in v0.2,
built by wrapping the official GEPA package. See `DECISIONS.md`.

## Trust model (why you can run this)

- **Local-only.** No telemetry, no cloud, no network by default.
- **Summaries, not secrets.** Allowlist capture: step roles, statuses, tool
  names, exit codes, paths, counts — never raw tool I/O unless you opt in.
- **Stored outside your repo** (`$XDG_DATA_HOME/loopgepa`), so traces never get
  committed to git by accident.
- **You approve every write.** Promotion is dry-run by default and always
  reversible. See `SECURITY.md` for the full threat model.

## Status

**Pre-alpha, scaffolding.** The monorepo layout, specs, and safety model are in
place; the core library is being implemented milestone by milestone
(see `IMPLEMENTATION.md` §11).

## Layout

```text
packages/loopgepa_core/    pure logic: models, storage, runner, mutation, diff, safety
packages/loopgepa_mcp/     thin MCP stdio server over core
packages/loopgepa_plugin/  Claude Code plugin: skills, agents, hooks, .mcp.json
examples/bugfix-loop/      flagship worked example
```

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy packages/loopgepa_core packages/loopgepa_mcp
```

## License

Apache-2.0 — see `LICENSE`.
