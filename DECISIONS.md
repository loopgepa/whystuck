# LoopGEPA — Architecture Decision Record

This file records the decisions that shaped v0. It is the answer to
"why is it built this way." New decisions append; superseded ones are marked,
not deleted.

---

## ADR-001 — Name: `loopgepa`

**Status:** accepted (2026-07-20)

**Context.** The working title was "LoopForge." `loopforge` is taken on PyPI,
npm, and GitHub. It also collides with existing audio-loop and gamedev tools.

**Decision.** Name the project **LoopGEPA** (package/handle `loopgepa`).

**Evidence.** `loopgepa` is free on PyPI, npm, and as a GitHub handle
(all 404 on 2026-07-20). Web search returns no project by that name.

**Why GEPA in the name is honest, not hype.** GEPA (Genetic-Pareto) is a real,
current method — ICLR 2026 Oral, arXiv 2507.19457, official package
`gepa-ai/gepa`, integrated in DSPy. Naming the project after its research
lineage signals *direction*, not a shipped capability. The discipline rule
("do not call v0 output 'optimized', do not lead marketing with self-improvement")
is enforced in the README copy and the plugin UI, **not** by avoiding the word
in the project name.

**Consequence.** v0.2's evaluation/Pareto layer should **wrap `gepa-ai/gepa`**
(map a loop genome onto GEPA's candidate/evaluation interface) rather than
reimplement evolutionary search. This makes the name literal over time.

---

## ADR-002 — Product shape: LoopGEPA owns the loop runner (option b)

**Status:** accepted (2026-07-20). Supersedes the original spec's implicit
"optimize any Claude Code session."

**Context.** The original genome lists 11 gene types. Only four —
`skill`, `agent`, `prompt`, `hook` — map to files Claude Code reads. The other
seven (`retry_policy`, `stop_condition`, `context_policy`, `tool_policy`,
`escalation_policy`, `budget_policy`, `verifier_rubric`) have **no write target**
in an arbitrary Claude Code session; they live inside the model's turn logic.
A promotion that "writes" those genes would change nothing the runtime obeys.

Two candidate products:

- **(a)** Optimize arbitrary Claude Code sessions. → non-file genes are not
  editable → the core value claim is a lie for 7 of 11 gene types.
- **(b)** Optimize **LoopGEPA-declared loops** that LoopGEPA itself executes via
  a runner. → every gene is a real knob, because the runner reads and enforces
  it.

**Decision.** Adopt **(b)**. A "loop" is a declared, versioned genome
(`loop.yaml`) executed by the LoopGEPA runner. The runner:

- feeds `skill`/`agent`/`prompt` genes into each step,
- enforces `retry_policy`/`stop_condition`/`context_policy`/`budget_policy` in
  its own control flow,
- runs `verifier_rubric` as an explicit verification step,
- stamps every run with `loop_id`, `variant_id`, and an auto-derived outcome
  label → **clean traces by construction** (resolves the original spec's
  hand-waved trace segmentation and binding).

**Honest scope of the two surfaces.**

1. **Managed loops (the real product).** Declared + run by LoopGEPA → all genes
   real, traces clean, promotion effective.
2. **Session import (lite, opportunistic).** A hook may capture raw Claude Code
   session data, but mutation/promotion is restricted to **file-backed genes
   only** (`skill`/`agent`/`prompt`/`hook`) and every such trace is marked
   low-confidence. Non-file genes are never claimed to be optimizable outside
   the runner.

**Rule.** A gene not backed by a file is only mutatable/effective for loops run
through the LoopGEPA runner. The UI must never offer to "promote" a non-file
gene for an imported session.

---

## ADR-003 — v0.1 scope cut: no Pareto, no eval, no frontier

**Status:** accepted (2026-07-20)

**Context.** The trust ladder is Diagnose → Suggest → Generate → **Evaluate** →
Promote. Pareto selection needs *measured* objectives, which only the Evaluate
stage produces. The original spec defers Evaluate ("Level 4-5, risky, later")
yet schedules `pareto.py` at Milestone 2. That builds machinery with no fuel —
a candidate set with no real metrics — violating the repo rule "no code just in
case."

**Decision.** v0.1 = **Diagnose → Suggest → Generate diff → Approve → Promote**,
nothing further. Explicitly **deferred to v0.2**:

- `pareto.py` / Pareto frontier
- `loopforge_score_variant`, `loopforge_get_pareto_frontier`
- live multi-variant evaluation
- benchmark harness

**Consequence.** v0.1 is **not** GEPA, and must never call its output
"optimized." Correct v0.1 framing: *trace-driven diagnosis + evidence-backed
reflective mutation with reversible, approved promotion.* The GEPA claim is
earned in v0.2 when eval + `gepa-ai/gepa` land (see ADR-001).

---

## ADR-004 — Trust model: allowlist storage, not denylist redaction

**Status:** accepted (2026-07-20)

**Context.** For a tool whose pitch is "it records your agent sessions," trust
is the product. The original spec relies on **denylist redaction** (strip known
secret patterns). Denylists always leak — any secret whose shape you did not
anticipate is stored in cleartext.

**Decision.**

1. **Allowlist by default.** Store structured *summaries* (step roles, statuses,
   tool names, exit codes, file paths, metric counts). Never store raw tool
   input/output or file contents unless the user explicitly opts a loop into
   `capture: raw`. Redaction is a second line of defense, not the first.
2. **Store location.** Default `LOOPGEPA_HOME=$XDG_DATA_HOME/loopgepa`
   (fallback `~/.local/share/loopgepa`), **not** the project's cwd. A repo-local
   `.loopgepa/` is opt-in and, if chosen, LoopGEPA writes a `.gitignore` entry
   for it. This removes the "traces committed to git with secrets" footgun.
3. **Concurrency.** Trace writes use `O_APPEND` single-line JSONL writes;
   concurrent sessions each write their own line atomically. No shared mutable
   file handles.
4. **Schema versioning.** Every persisted record carries `schema_version`.
   The trace schema *will* change within the first month; migration must be
   possible from day one.
5. **Promotion reconciliation.** At promote time, read the live file for each
   file-backed gene and diff against it — never trust the genome's stored
   `before` snapshot, which can be stale.

---

## ADR-005 — Confidence is deterministic, not vibes

**Status:** accepted (2026-07-20)

**Context.** The original spec emits `confidence: 0.65` without saying whether
it is computed or LLM-guessed, and mixes both.

**Decision.** `confidence` on a failure mode / mutation proposal is a
**deterministic function** of (trace count, failure-cluster size, outcome-label
completeness). The reflective LLM step produces the *diagnosis and mutation
text*; it does not invent the confidence number. Bands:

    n_traces < 5                          -> low   (cap 0.4)
    5 <= n_traces < 15                    -> medium
    n_traces >= 15 AND cluster >= 0.5     -> high

where `cluster` = share of failing traces attributed to the same failure mode.

---

## ADR-006 — One PyPI package; `uvx loopgepa` is the entire setup story

**Status:** accepted (2026-07-20). Supersedes the 3-package uv workspace in the
original scaffold.

**Context.** The lowest-friction way to install an MCP server in 2026 is a single
zero-install command:

    claude mcp add loopgepa -- uvx loopgepa      # Claude Code
    codex  mcp add loopgepa -- uvx loopgepa      # Codex CLI

`uvx` runs a console entry point straight from PyPI in an ephemeral cached env —
no `pip install`, no venv. That requires **one** importable package named
`loopgepa` exposing `[project.scripts] loopgepa = "loopgepa.server:main"`. A
3-package workspace (`loopgepa_core` / `loopgepa_mcp` / `loopgepa_plugin`) fights
this and triples release complexity for zero user benefit.

**Decision.** Ship a **single `loopgepa` package**. Internal structure:

    src/loopgepa/
      server.py        # FastMCP app + main() stdio entrypoint
      tools.py         # thin MCP tool handlers over core
      core/            # vendor-neutral pure logic (models, storage, runner, ...)

The `core/` subpackage preserves the ADR-002 hedge (keep the defensible logic
vendor-neutral) without a separate distribution.

**Consequence.** README's first code block is the one-liner above. Prerequisite
callout: if `uvx` is missing, `curl -LsSf https://astral.sh/uv/install.sh | sh`
(the #1 reported MCP setup failure). `.mcpb` desktop bundle is deferred — our
audience is CLI-agent users whose shells already expose `uvx`.

---

## ADR-007 — FastMCP v3, stdio only

**Status:** accepted (2026-07-20). Refines original spec §4.5.

**Decision.** Build on **FastMCP v3** (de-facto standard, ~¼ the code of the raw
`mcp` SDK, stdio + streamable-HTTP built in). v1 uses **stdio only** — the
correct transport for local Claude Code / Codex / Cursor, which spawn the server
as a subprocess. Do not add HTTP/SSE until a user needs it.

---

## ADR-008 — The MCP server is the product; the plugin is a thin ergonomics layer

**Status:** accepted (2026-07-20). Refines original spec §1.1.

**Context.** Two possible surfaces: the MCP server (tools) and a Claude Code
plugin (slash commands, agents, hooks). A plugin needs a marketplace install; the
MCP server installs with one line. Under ADR-002 (runner-owned loops), trace
capture belongs to the **runner inside the MCP server**, not to session hooks.

**Decision.** The **MCP server is the primary, sufficient artifact** — it does
diagnosis, mutation, promotion, and (via the runner) trace capture. The plugin
under `plugin/` is an **optional thin layer** adding slash-command ergonomics
(`/loopgepa:explain-loop-failure`) that call MCP tools. Build and validate the
server first; the plugin is polish, not a dependency of the core value.

---

## ADR-009 — OSS-quality standards adopted from SysKnife (deferred to launch gate)

**Status:** accepted (2026-07-20)

**Context.** SysKnife (`/home/user/Desktop/lacs`) is the reference for OSS
polish. Its hardening is valuable but must not slow local MVP development.

**Decision.** Adopt these, but **stage them**: cheap community files now; heavy
security/release/docs infrastructure at the pre-public launch gate (see
`LAUNCH_CHECKLIST.md`). The standards:

- SHA-pin every GitHub Action with a `# vX` comment; commit `uv.lock`; run CI
  with `uv sync --frozen`.
- Split a **fast, always-on CI gate** (lint/type/test) from **probabilistic
  LLM-behavior tests** (`workflow_dispatch` only, documented as "guidance, not
  merge criteria").
- Separate lightweight security workflows: CodeQL (python), OpenSSF Scorecard,
  TruffleHog `--results=verified`, dependency-review.
- `mkdocs-material` docs site (Python-idiomatic; SysKnife uses mdBook) deployed
  to GitHub Pages on a paths-filtered trigger; ADRs mirrored from this file.
- Idempotent, resumable PyPI publish (check-then-skip if the version exists).
- A `check_public_claims` script that verifies README claims (test counts, "works
  with X") against reality.
- Demo GIF generated from a checked-in VHS `.tape` (reproducible, not a binary
  blob).
- Keep-a-Changelog `CHANGELOG.md`, structured YAML issue templates,
  `blank_issues_enabled: false`, PR template with a trust-boundary checkbox,
  `dependabot.yml`, `docs/local/` for internal notes.

**Not adopted:** SysKnife's heavy `npx` binary-installer wizard — LoopGEPA is pure
Python via `uvx`, so the one-liner *is* the installer. An optional `uvx loopgepa
init` helper may come later.

---

## ADR-010 — Positioning: name the pain, promise reversible repair, headline "local"

**Status:** accepted (2026-07-20)

**Context.** 2026 sentiment is sharply anti-hype (the "agent reliability gap");
every named competitor — Latitude (which markets its own "GEPA algorithm"),
Braintrust, LangSmith, Langfuse — is **cloud SaaS for production LLM apps**.
LoopGEPA's whitespace is *local, zero-telemetry, coding-agent-session loop
diagnosis*.

**Decision.**

1. Lead with the **concrete failure**, not the abstraction: "your agent re-ran
   `npm install` 40 times and you didn't notice" before any word like
   "observability."
2. Headline verbs are **"diagnoses"** and **"proposes a reversible fix you
   approve"** — never "self-improving" / "auto-fix" as a headline (allowed only
   in the literal, approval-gated sense).
3. **"Local-only, no telemetry, traces never leave your machine"** is a top-two
   README bullet — the sharpest differentiator vs the cloud incumbents, stated as
   a trust claim.
4. A **"works with Claude Code · Codex · Cursor"** signal sits directly under the
   title (highest-value 5-second trust cue observed across viral MCP servers).
5. In comparison content, name **Latitude's "GEPA algorithm"** explicitly and
   contrast: they close issue→PR for *production* traces in the cloud; LoopGEPA
   closes loop→fix for *local coding-agent sessions* with nothing leaving the
   machine.
