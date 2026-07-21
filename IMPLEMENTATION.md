# WhyStuck — Implementation Spec (v0.1)

**Supersedes** the original `IMPLEMENTATION(2)(1).md` LoopForge draft.
Read `DECISIONS.md` first — it records *why* the shape below differs from that
draft (name, runner-owned loops, scope cut, trust model).

**Audience:** autonomous coding agents (Codex / Claude Code) and human
contributors.

**One sentence.** WhyStuck is a Claude Code plugin + local MCP server that
executes *declared* agent loops, captures clean execution traces, diagnoses
recurring failure modes with trace evidence, and proposes small, reversible,
human-approved mutations to the loop's genes.

**What v0.1 is NOT.** Not "GEPA," not "optimized," not "self-improving." Those
words are reserved for v0.2 when measured evaluation + `gepa-ai/gepa` land
(ADR-003).

---

## 1. Product ladder (mandatory)

    Level 1  Diagnose why a declared loop keeps failing.        v0.1  ✔
    Level 2  Suggest gene mutations with trace evidence.        v0.1  ✔
    Level 3  Generate reversible variant diffs.                 v0.1  ✔
    Level 4  Promote a variant after explicit approval.         v0.1  ✔
    Level 5  Evaluate variants on tasks (wrap gepa-ai/gepa).    v0.2
    Level 6  Pareto selection / frontier across objectives.     v0.2

Do not implement Levels 5–6 before Levels 1–4 are reliable and tested.

## 2. Core concepts

- **Loop genome** — a declared, versioned workflow (`loop.yaml`): its genes.
- **Gene** — one evolvable unit. File-backed (`skill`/`agent`/`prompt`/`hook`)
  or runner-enforced (`verifier_rubric`/`retry_policy`/`context_policy`/
  `tool_policy`/`stop_condition`/`escalation_policy`/`budget_policy`).
- **Runner** — WhyStuck executes a genome step-by-step and enforces the
  runner-backed genes. Because it owns execution, every gene is a real knob and
  every trace is clean by construction (ADR-002).
- **Trace** — one recorded run of a genome; stamped with `loop_id`,
  `variant_id`, auto-derived `final_outcome`.
- **Mutation** — a proposed change to 1–3 genes, with evidence trace IDs,
  rationale, risk, deterministic confidence (ADR-005), and a rollback path.
- **Variant** — a genome plus the mutations applied to it; `promoted` flag.

(Pareto frontier is a v0.2 concept — omitted here on purpose.)

## 3. Repository layout

Single distributable package so `uvx whystuck` works (ADR-006).

    whystuck/
      README.md  LICENSE  SECURITY.md  CONTRIBUTING.md  CODE_OF_CONDUCT.md
      IMPLEMENTATION.md  DECISIONS.md  LAUNCH_CHECKLIST.md  ROADMAP.md  CHANGELOG.md
      pyproject.toml  .mcp.json.example
      .github/workflows/ci.yml  .github/{dependabot.yml,ISSUE_TEMPLATE/,PULL_REQUEST_TEMPLATE.md}
      src/whystuck/
        __init__.py
        server.py        # FastMCP app + main() stdio entrypoint (the console script)
        tools.py         # thin MCP tool handlers over core
        core/            # vendor-neutral pure logic
          models.py storage.py diagnosis.py runner.py mutation.py diff.py safety.py
      tests/             # unit + integration; no network
      plugin/            # optional Claude Code plugin (thin): skills, agents, hooks, .mcp.json
      examples/bugfix-loop/
      docs/              # mkdocs-material site (built at launch gate)
      scripts/

## 4. Tech stack

- Python 3.12+, **uv** (single package, `uv.lock` committed).
- **FastMCP v3**, **stdio transport only** for v1 (ADR-007).
- Console entry point `whystuck = "whystuck.server:main"` → `uvx whystuck`.
- **Pydantic v2** at every boundary.
- File store: YAML genomes/variants, JSONL traces, JSON indexes, under
  `$XDG_DATA_HOME/whystuck` (ADR-004).
- Tests: pytest + pytest-cov; ruff; mypy strict. No network in tests.
- Distribution: PyPI (`uv build` + `uv publish`). No `.mcpb` bundle in v1.

## 5. Data models (`whystuck_core/models.py`)

Keep the original draft's `LoopGene`, `LoopGenome`, `ToolCall`, `LoopStepTrace`,
`LoopMetrics`, `LoopTrace`, `MutationProposal`, `LoopVariant` — with these
**required changes**:

- Add `schema_version: str` to `LoopTrace`, `LoopGenome`, `LoopVariant`
  (ADR-004.4).
- Drop `ParetoCandidate` from v0.1 (v0.2).
- `MutationProposal.evidence_trace_ids: list[str]` is **required** (may be `[]`
  with forced low confidence — never omitted).
- `MutationProposal.confidence` is set by the deterministic function in
  ADR-005, not by the LLM.
- `LoopGene` gains `file_backed: bool` (derived from `type`): only file-backed
  genes are promotable for imported sessions (ADR-002).

## 6. Storage (`storage.py`)

- Root: `WHYSTUCK_HOME`, default `$XDG_DATA_HOME/whystuck`
  (fallback `~/.local/share/whystuck`). **Not** cwd (ADR-004.2).
- Layout: `loops/ traces/ variants/ promotions/ config.json`.
  (`frontiers/` is v0.2.)
- Traces: `O_APPEND` one-JSON-object-per-line; concurrent-safe (ADR-004.3).
- Allowlist capture: store summaries only unless the loop sets `capture: raw`
  (ADR-004.1).

## 7. Runner (`runner.py`) — the piece that makes genes real

**v0.1 default = command+verifier ingest runner (offline, free, deterministic).**
This is the honest MVP; the `claude -p` worker is a labeled stretch (§7.2).

`run_command_loop(...)`:

1. Run the declared **worker command** (e.g. a repair script), capturing exit
   code + a summarized stdout/stderr.
2. Run the declared **verifier command** (deterministic; exit 0 = pass). This is
   free and needs no API key — matching ADR-005's "confidence is deterministic."
3. On verifier failure, retry up to `retry_policy.max_retries`; stop when the
   `stop_condition` is met or retries exhaust.
4. Derive `final_outcome`: `success` if the verifier passed within retries,
   `failure` if retries exhausted, else `unknown`.
5. Return a `LoopTrace`. The caller (or the MCP `record_trace` handler) persists
   it via `core.storage.append_trace` — the runner does **not** call its own MCP
   tool (S8).

Command execution is injected (`execute: Callable[[list[str]], tuple[int, str]]`)
so tests are deterministic and offline.

### 7.2 `claude -p` worker (stretch, opt-in, network)

An optional worker step may shell out to an LLM CLI. When it does, it MUST run
with `--bare` and explicit `--agents` / `--append-system-prompt` /
`--mcp-config` built from the genome's genes — never ambient discovery — so the
trace reflects only declared genes (resolves B5). This path requires an API key
and makes network calls; it is disclosed in SECURITY.md, never the default.

## 8. MCP tools (v0.1) — 6, consolidated

`whystuck_diagnose` (merges summarize + infer; deterministic, no LLM in the
server), `whystuck_run_loop`, `whystuck_propose_fix` (validates target genes,
recomputes confidence server-side, saves a draft variant + diff summary),
`whystuck_promote` (dry-run default; writes only on explicit approval; reconciles
against live files per ADR-004.5; records promotion history), `whystuck_save_loop`,
and the internal `whystuck_record_trace`.

v0.2 adds two eval-gated tools — `whystuck_optimize_loop` (runs `gepa.optimize`,
returns the frontier) and `whystuck_get_frontier` — registered only when the
`eval` extra is installed.

Deferred to v0.2: `whystuck_score_variant`, `whystuck_get_pareto_frontier`.

## 9. Claude Code plugin

- `.claude-plugin/plugin.json`, `.mcp.json` → `whystuck-mcp`.
- Skills: `explain-loop-failure` (first-class, diagnose-only),
  `optimize-loop` (propose + diff + ask; never auto-promote),
  `design-loop` (declare a new genome).
- Agents: `loop-optimizer`, `loop-critic`, `loop-variant-reviewer`.
- Hooks: best-effort session import → summaries only, fail-open, local-only.

## 10. Safety (see SECURITY.md for the threat model)

Suggest-only default. Dry-run promotion. Explicit approval before any file
write. `safety_locked: true` genes are commentable but not editable without an
explicit unlock. Local-only, no telemetry, no network by default. Reversible
changes only; every promotion writes a rollback record.

## 11. Milestones (v0.1)

    M0  Skeleton: monorepo imports, pytest+ruff green.        (scaffolded)
    M1  models.py + tests (TDD).
    M2  storage.py (XDG root, JSONL append, schema_version) + tests.
    M3  mutation.py + diff.py + safety.py (apply, diff, guard) + adversarial tests.
    M4  runner.py (minimal) + trace emission + tests.
    M5  MCP server wrapping core (stdio) + smoke tests.
    M6  plugin (skills/agents/hooks/.mcp.json/bin) + README install.
    M7  promote dry-run + real promote + rollback record + tests.
    M8  examples/bugfix-loop end-to-end walkthrough.

v0.2 (separate track): eval harness wrapping `gepa-ai/gepa`, then Pareto.

## 12. Definition of done (v0.1)

1. Install the plugin locally; it starts the MCP server.
2. Declare a loop; run it; a clean trace is recorded (loop_id, variant_id,
   outcome).
3. Ask Claude to explain failures → diagnosis with trace count + deterministic
   confidence.
4. Claude proposes ≤3-gene mutations with evidence trace IDs.
5. MCP saves a draft variant and shows a diff.
6. Promotion is dry-run by default; real promote requires explicit approval and
   writes a rollback record; it reconciles against live files.
7. All WhyStuck state stays local (XDG dir); WhyStuck opens no network sockets.
   The v0.1 command+verifier runner is fully offline; only an opt-in `claude -p`
   worker step makes (disclosed) network calls (§7.2, SECURITY.md).
8. Nothing is ever labeled "optimized" without evaluation.

## 13. Non-negotiables

Local-only default · no telemetry · no silent promotion · no silent budget-cap
increase (the one enforced-in-code safety guard — see `core/safety.py`) ·
reversible only · explicit approval before writes · ≤3 genes per proposal ·
show diffs first · low trace count → low confidence · diagnosis before
suggestion · never call a proposal "optimized" until evaluated (v0.2).
