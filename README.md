<div align="center">

# LoopGEPA

*Diagnose the loop. Propose the fix. You approve.*

[![CI](https://github.com/loopgepa/loopgepa/actions/workflows/ci.yml/badge.svg)](https://github.com/loopgepa/loopgepa/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/loopgepa?style=flat-square)](https://pypi.org/project/loopgepa/)
[![Stars](https://img.shields.io/github/stars/loopgepa/loopgepa?style=flat-square)](https://github.com/loopgepa/loopgepa/stargazers)
[![Discussions](https://img.shields.io/github/discussions/loopgepa/loopgepa?style=flat-square)](https://github.com/loopgepa/loopgepa/discussions)

**Works with** &nbsp;
![Claude Code](https://img.shields.io/badge/Claude_Code-supported-2f855a?style=flat-square) &nbsp;
![Codex CLI](https://img.shields.io/badge/Codex_CLI-supported-2f855a?style=flat-square) &nbsp;
![Cursor](https://img.shields.io/badge/Cursor-supported-2f855a?style=flat-square) &nbsp;
`Local-only` · `No telemetry`

[Install](#install) · [How it works](#how-it-works) · [Why not X?](#why-not-just-x) · [Status](#status) · [Roadmap](#roadmap) · [Docs](#documentation) · [Discuss](https://github.com/loopgepa/loopgepa/discussions)

</div>

<p align="center">
  <img src="docs/assets/loopgepa-demo.gif" alt="LoopGEPA diagnosing a repeatedly failing bugfix loop from local traces and proposing a reversible, approval-gated fix" width="840">
</p>

<p align="center"><em>Live recording of <code>examples/bugfix-loop/demo.py</code> — reproducible via <code>vhs assets/demo/demo.tape</code>. No API key, nothing leaves the machine.</em></p>

> **Your coding agent re-ran `npm install` 40 times and you didn't notice.**
> LoopGEPA records why a repeated agent loop keeps failing, shows you the failure
> mode with a confidence level, and proposes a small, reversible patch — to the
> verifier, retry policy, stop condition, or skill — that you approve before
> anything is written.

**LoopGEPA is the diagnosis-and-repair layer for [loop engineering](docs/how-it-works.md).**
2026's agent-dev stack goes prompt → context → harness → *loop* engineering: you
stop prompting the agent turn-by-turn and design the act→verify→retry→stop cycle
that drives it ([ADTmag](https://adtmag.com/articles/2026/07/01/loop-engineering-emerges-as-developers-put-ai-coding-agents-on-repeat.aspx)).
Loop scaffolders help you *build* a loop; LoopGEPA tells you *why it keeps failing
and what to change*. Every named alternative — Latitude, LangSmith, Braintrust,
Langfuse — is cloud SaaS for production apps; **none does local, zero-telemetry,
coding-agent-session loop diagnosis.** The name nods to
[GEPA](https://arxiv.org/abs/2507.19457) (Genetic-Pareto, ICLR 2026 Oral), whose
measured optimization arrives in [v0.2](docs/v0.2-eval-benchmark.md).

**The diagnosis is deterministic — it spends ~0 model tokens; only the proposed
fix uses your agent's own model.** No API key to diagnose, nothing uploaded, and
it stays out of your request hot path.

---

## Install

> Requires [`uv`](https://astral.sh/uv). If `uvx` isn't found:
> `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
claude mcp add loopgepa -- uvx loopgepa      # Claude Code
codex  mcp add loopgepa -- uvx loopgepa      # Codex CLI
```

What this does:

1. Registers LoopGEPA as a local stdio MCP server (no `pip install`, no venv).
2. Exposes 8 tools + 3 slash-command skills to your agent.
3. Stores traces under `$XDG_DATA_HOME/loopgepa` — **outside your repo**, never
   committed by accident.

**Cursor / any MCP client** — add to your `mcp.json`:

```json
{ "mcpServers": { "loopgepa": { "command": "uvx", "args": ["loopgepa"] } } }
```

> **Tip — make your agent reach for it automatically.** Paste one line into your
> `AGENTS.md` / `CLAUDE.md` / `.cursor/rules`:
> *"When a loop keeps failing (repeated edits, retries that don't help), ask
> loopgepa why before changing code."* See [AGENTS.md](AGENTS.md).

<details><summary>Try it offline first — no API key, nothing written</summary>

```bash
git clone https://github.com/loopgepa/loopgepa && cd loopgepa
uv run python examples/bugfix-loop/demo.py
```

Prints the full flow: 8 traces → "repair repeats the original patch" (confidence:
medium, gene: `verifier_rubric`) → proposed variant → dry-run promotion.
</details>

## How it works

```text
declare loop ─▶ run it (worker + verifier) ─▶ clean local trace (JSONL)
                                                     │
             failure diagnosis ◀── aggregate ◀───────┘   (deterministic + confidence)
                     │
   reflective fix (≤3 genes, evidence-backed) ─▶ dry-run diff ─▶ you approve ─▶ reversible promote
```

1. **Declare** a loop — its *genome* of genes (skill, verifier rubric, retry
   policy, stop condition, …).
2. **Run / record** it: the offline command+verifier runner emits a clean trace,
   or your agent records session traces.
3. **Diagnose**: failures are clustered and ranked with a **deterministic**
   confidence (`n<5`→low, `≥15 ∧ cluster≥0.5`→high) — no LLM in the analysis.
4. **Propose**: your agent reflects on the evidence and proposes a ≤3-gene patch.
5. **Preview** the exact diffs (dry-run writes nothing), then **approve** — the
   write is atomic and reversible, with a rollback record.

**LoopGEPA runs the loop, so every knob it tunes is real, and every trace is
clean by construction — and it never writes without your approval.**

## Why not just X?

| | LoopGEPA | Latitude / LangSmith / Braintrust / Langfuse |
|---|---|---|
| Runs | **On your machine** | Cloud SaaS |
| Data | **Never leaves it, no telemetry** | Uploaded to a backend |
| Scope | **Your local coding-agent loops** | Production LLM apps |
| Output | **A reversible patch you approve** | Dashboards / auto-PRs |

Nobody else pairs *local + zero-telemetry + coding-agent-session scope +
reversible, approval-gated repair*. Fuller breakdown: [docs/comparison.md](docs/comparison.md).

## Status

Developed in the open once solid; the trust chain and diagnosis pipeline are built
and tested.

| Component | State |
|---|---|
| Loop genome + local store (XDG) | ✅ |
| Command+verifier runner (offline) | ✅ |
| Deterministic diagnosis + confidence | ✅ |
| Reflective mutation (≤3 genes, safety-guarded) | ✅ |
| Dry-run + reversible promotion + rollback | ✅ |
| FastMCP stdio server (8 tools) | ✅ |
| Claude Code plugin (3 skills, 3 agents) | ✅ |
| Evaluation + benchmark (wraps `gepa-ai/gepa`) | 📋 v0.2 |
| Pareto frontier over variants | 📋 v0.2 |

**102 tests**, all local and offline; `ruff` + `mypy --strict` + docs `--strict` clean.

## Configure

The server is zero-config. Point any MCP client at it:

```json
{ "mcpServers": { "loopgepa": { "command": "uvx", "args": ["loopgepa"] } } }
```

Storage location is overridable with `LOOPGEPA_HOME` (default
`$XDG_DATA_HOME/loopgepa`). No API keys — the v0.1 runner is fully offline.

## MCP tools & safety boundary

Eight approval-gated tools: `record_trace`, `run_loop`, `summarize_traces`,
`infer_failure_modes`, `propose_mutations`, `list_variants`, `promote_variant`,
`save_loop`. The boundary is mechanical: **`promote_variant` is dry-run by
default and only writes on an explicit approval flag; proposals are capped at 3
genes; budget caps can't be weakened; promotion paths can't escape the project
directory.** Full threat model: [SECURITY.md](SECURITY.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md).

- ✅ **v0.1** — diagnose recurring loop failures, propose reversible fixes
- 📋 **v0.2** — measured evaluation + benchmark + Pareto, wrapping `gepa-ai/gepa`
- 📋 **v0.3** — plugin marketplace listing, mkdocs site, benchmark writeup

## Documentation

- [How it works](docs/how-it-works.md) — loop engineering, the genome, the pipeline
- [Getting started](docs/getting-started.md)
- [Trust & safety](docs/safety.md) · [SECURITY.md](SECURITY.md)
- [v0.2 eval/benchmark design](docs/v0.2-eval-benchmark.md)
- [Comparison](docs/comparison.md) · [Competitive & launch notes](docs/COMPETITIVE-AND-LAUNCH.md)
- [Design records (ADRs)](DECISIONS.md) · [Implementation spec](IMPLEMENTATION.md)

## Where to find LoopGEPA

| Channel | Install | Notes |
|---|---|---|
| PyPI | `uvx loopgepa` | primary; zero-install via `uv` |
| MCP registry | auto-discovered | via `server.json` (feeds mcp.directory, Glama) |
| GitHub | `git clone` | source + examples |

## Contributing

Highest-impact area right now: **real loop examples** beyond the bugfix loop
(an orchestrator/triage loop). See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[`good first issue`](https://github.com/loopgepa/loopgepa/labels/good%20first%20issue) label.

## License

Apache-2.0 — see [LICENSE](LICENSE).

---

<div align="center">
Built by <a href="https://github.com/vladimirrott">Vladimir Rotariu</a>. · Ideas, war stories — <a href="https://github.com/loopgepa/loopgepa/discussions">come say hi</a>.
</div>
