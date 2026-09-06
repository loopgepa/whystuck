<div align="center">

# WhyStuck

*Diagnose the loop. Propose the fix. You approve.*

[![CI](https://github.com/loopgepa/whystuck/actions/workflows/ci.yml/badge.svg)](https://github.com/loopgepa/whystuck/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/whystuck?style=flat-square)](https://pypi.org/project/whystuck/)
[![Stars](https://img.shields.io/github/stars/loopgepa/whystuck?style=flat-square)](https://github.com/loopgepa/whystuck/stargazers)
[![Discussions](https://img.shields.io/github/discussions/loopgepa/whystuck?style=flat-square)](https://github.com/loopgepa/whystuck/discussions)

**Works with** &nbsp;
![Claude Code](https://img.shields.io/badge/Claude_Code-supported-2f855a?style=flat-square) &nbsp;
![Codex CLI](https://img.shields.io/badge/Codex_CLI-supported-2f855a?style=flat-square) &nbsp;
![Cursor](https://img.shields.io/badge/Cursor-supported-2f855a?style=flat-square) &nbsp;
`Local-only` · `No telemetry`

[Install](#install) · [How it works](#how-it-works) · [Why not X?](#why-not-just-x) · [Status](#status) · [Roadmap](#roadmap) · [Docs](#documentation) · [Discuss](https://github.com/loopgepa/whystuck/discussions)

</div>

<p align="center">
  <img src="docs/assets/whystuck-demo.gif" alt="WhyStuck diagnosing a repeatedly failing bugfix loop from local traces and proposing a reversible, approval-gated fix" width="840">
</p>

<p align="center"><em>Live recording of <code>examples/bugfix-loop/demo.py</code> — reproducible via <code>vhs assets/demo/demo.tape</code>. No API key, nothing leaves the machine.</em></p>

> **Your coding agent re-ran `npm install` 40 times and you didn't notice.**
> WhyStuck records why a repeated agent loop keeps failing, shows you the failure
> mode with a confidence level, and proposes a small, reversible patch — to the
> verifier, retry policy, stop condition, or skill — that you approve before
> anything is written.

**WhyStuck is the diagnosis-and-repair layer for [loop engineering](docs/how-it-works.md).**
2026's agent-dev stack goes prompt → context → harness → *loop* engineering: you
stop prompting the agent turn-by-turn and design the act→verify→retry→stop cycle
that drives it ([ADTmag](https://adtmag.com/articles/2026/07/01/loop-engineering-emerges-as-developers-put-ai-coding-agents-on-repeat.aspx)).
Loop scaffolders help you *build* a loop; WhyStuck tells you *why it keeps failing
and what to change*. Every named alternative — Latitude, LangSmith, Braintrust,
Langfuse — is cloud SaaS for production apps; **none does local, zero-telemetry,
coding-agent-session loop diagnosis.** WhyStuck's reflective-mutation approach
builds on [GEPA](https://arxiv.org/abs/2507.19457) (Genetic-Pareto, ICLR 2026
Oral); the measured GEPA optimization itself arrives in
[v0.2](docs/v0.2-eval-benchmark.md), which wraps `gepa-ai/gepa`.

**The diagnosis is deterministic — it spends ~0 model tokens; only the proposed
fix uses your agent's own model.** No API key to diagnose, nothing uploaded, and
it stays out of your request hot path.

---

## Install

> Requires [`uv`](https://astral.sh/uv). If `uvx` isn't found:
> `curl -LsSf https://astral.sh/uv/install.sh | sh`

**One command — configures Claude Code, Codex, and Cursor:**

```bash
uvx whystuck setup
```

The wizard writes each client's MCP config **and** a short usage rule — so your
agent knows to diagnose a stuck loop *before* it thrashes — merges into any
existing config, and is safe to re-run. Choose clients/scope with
`--all` · `--claude` · `--cursor` · `--codex` · `--global` · `--yes`.

It registers WhyStuck as a local stdio MCP server (no `pip install`, no venv) and
stores traces under `$XDG_DATA_HOME/whystuck` — **outside your repo**, never
committed by accident.

<details><summary>Prefer to wire a single client up by hand?</summary>

```bash
claude mcp add whystuck -- uvx whystuck      # Claude Code
codex  mcp add whystuck -- uvx whystuck      # Codex CLI
```

Cursor / any MCP client — add to your `mcp.json`:

```json
{ "mcpServers": { "whystuck": { "command": "uvx", "args": ["whystuck"] } } }
```

Then paste one line into `AGENTS.md` / `CLAUDE.md` / `.cursor/rules` so the agent
reaches for it automatically (the wizard does this for you). See [AGENTS.md](AGENTS.md).
</details>

> **Pre-launch:** `whystuck` isn't on PyPI yet, so `uvx …` won't resolve. Run the
> wizard from a clone instead:
> ```bash
> git clone https://github.com/loopgepa/whystuck && cd whystuck
> uv run whystuck setup --from-source "$PWD"
> ```

<details><summary>Try it offline first — no API key, nothing written</summary>

```bash
git clone https://github.com/loopgepa/whystuck && cd whystuck
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

**WhyStuck runs the loop, so every knob it tunes is real, and every trace is
clean by construction — and it never writes without your approval.**

## Why not just X?

| | WhyStuck | Latitude / LangSmith / Braintrust / Langfuse |
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
| FastMCP stdio server (6 tools) | ✅ |
| Claude Code plugin (3 skills, 3 agents) | ✅ |
| Auto-trigger hook (thrash detection + session capture) | ✅ |
| Evaluation library + GEPA wrapper (`whystuck[eval]`) | 🟡 v0.2-alpha |
| LLM worker + output→apply→verify wiring (real-repo scoring) | 🟡 v0.2-alpha |
| Optimize + frontier MCP tools (gated on the extra) | 🟡 v0.2-alpha |

**2045 tests**, all local and offline; `ruff` + `mypy --strict` (on `src/`) + docs `--strict` clean.

## Configure

The server is zero-config. Point any MCP client at it:

```json
{ "mcpServers": { "whystuck": { "command": "uvx", "args": ["whystuck"] } } }
```

Storage location is overridable with `WHYSTUCK_HOME` (default
`$XDG_DATA_HOME/whystuck`). No API keys — the v0.1 runner is fully offline.

## MCP tools & safety boundary

Six focused tools: `diagnose`, `run_loop`, `propose_fix`, `promote`, `save_loop`,
and the internal `record_trace`. The boundary is mechanical: **`promote` is
dry-run by default and only writes on an explicit approval flag; proposals are
capped at 3 genes; budget caps can't be weakened; promotion paths can't escape
the project directory.** Full threat model: [SECURITY.md](SECURITY.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md).

- ✅ **v0.1** — diagnose recurring loop failures, propose reversible fixes; a
  PostToolUse hook auto-detects thrash and captures session traces
- 🟡 **v0.2-alpha** — eval library + GEPA wrapper + LLM-backed worker built
  (`whystuck[eval]`, gated MCP tools); remaining: the output→apply→verify wiring
  for real repos + the benchmark writeup
- 📋 **v0.3** — plugin marketplace listing (manifest ready), MCP-registry submission

## Research artifact

This repository is the supporting artifact for a **three-paper series** on optimizing
generate–critique–refine loops. Start at **[docs/papers/README.md](docs/papers/README.md)** — it
says which paper answers which question.

| | paper | in one line |
|---|---|---|
| **I** | *Whether Optimizing the Loop Beats Optimizing the Prompt Depends on One Number* | Evolving a loop's verifier beats evolving its generation prompt **only when enough first drafts are already correct**: +0.060 held-out at p₀ = 0.267, nothing measurable at p₀ = 0.067, a 0.102 swing across the axis |
| **II** | *What a Checker Cannot Buy Is the Repair* | A **perfect** inference-time checker scores **0.042 below** a mere text-statistics tool and is indistinguishable from a near-random one; concrete measured repair instructions are worth +0.0613. No optimizer anywhere in the design |
| **III** | *Your Loop Optimizer Is Not Optimizing Your Loop* | Pointing an optimizer at a component is not the same as optimizing it. Our first fifteen contrast runs had a critique gene **byte-identical to the seed in 15/15** winning genomes. Print one integer per component |

- **[Every experiment, and the command that produced it](docs/EXPERIMENTS.md)** — one script spends
  money and it refuses to without `--yes`; everything else is $0 and offline.
- **[How to reproduce it](docs/papers/REPRODUCE.md)** — offline checks, all $0: every number in
  every paper re-derived from the artifacts, bit-for-bit cassette replay, a positive control proving
  the harness detects an effect when one exists, and a proof that the three-way split lost no claim.
- Results, cassettes and pre-registrations for every run are committed, including the superseded
  ones, labelled as such.

## Documentation

- [The three papers](docs/papers/README.md) — which paper answers which question
- [Every experiment and its command](docs/EXPERIMENTS.md) — the map from a claim to a script
- [How it works](docs/how-it-works.md) — loop engineering, the genome, the pipeline
- [Getting started](docs/getting-started.md)
- [Trust & safety](docs/safety.md) · [SECURITY.md](SECURITY.md)
- [v0.2 eval/benchmark design](docs/v0.2-eval-benchmark.md)
- [Comparison](docs/comparison.md) · [Competitive & launch notes](docs/COMPETITIVE-AND-LAUNCH.md)
- [Design records (ADRs)](DECISIONS.md) · [Implementation spec](IMPLEMENTATION.md)

## Where to find WhyStuck

| Channel | Install | Notes |
|---|---|---|
| PyPI | `uvx whystuck` | primary; zero-install via `uv` |
| MCP registry | auto-discovered | via `server.json` (feeds mcp.directory, Glama) |
| GitHub | `git clone` | source + examples |

## Contributing

Highest-impact area right now: **real loop examples** beyond the bugfix loop
(an orchestrator/triage loop). See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[`good first issue`](https://github.com/loopgepa/whystuck/labels/good%20first%20issue) label.

## License

Apache-2.0 for the code — see [LICENSE](LICENSE).

The papers under `docs/papers/` are licensed **CC BY 4.0** instead — see
[docs/papers/shared/LICENSE](docs/papers/shared/LICENSE). Two separate grants: a patent grant and NOTICE
requirement make no sense for a document, and the papers are meant to be redistributed and
built upon. CC BY 4.0 is also the licence selected for the arXiv submissions.

---

<div align="center">
Built by <a href="https://github.com/vladimirrott">Vladimir Rotariu</a>. · Ideas, war stories — <a href="https://github.com/loopgepa/whystuck/discussions">come say hi</a>.
</div>
