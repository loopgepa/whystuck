# LoopGEPA — Implementation Spec (v0.1)

**Supersedes** the original `IMPLEMENTATION(2)(1).md` LoopForge draft.
Read `DECISIONS.md` first — it records *why* the shape below differs from that
draft (name, runner-owned loops, scope cut, trust model).

**Audience:** autonomous coding agents (Codex / Claude Code) and human
contributors.

**One sentence.** LoopGEPA is a Claude Code plugin + local MCP server that
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
  or runner-enforced (`verifier_rubric`/`retry_policy`/`stop_condition`/
  `context_policy`/`budget_policy`).
- **Runner** — LoopGEPA executes a genome step-by-step and enforces the
  runner-backed genes. Because it owns execution, every gene is a real knob and
  every trace is clean by construction (ADR-002).
- **Trace** — one recorded run of a genome; stamped with `loop_id`,
  `variant_id`, auto-derived `final_outcome`.
- **Mutation** — a proposed change to 1–3 genes, with evidence trace IDs,
  rationale, risk, deterministic confidence (ADR-005), and a rollback path.
- **Variant** — a genome plus the mutations applied to it; `promoted` flag.

(Pareto frontier is a v0.2 concept — omitted here on purpose.)

## 3. Repository layout

    loopgepa/
      README.md  LICENSE  SECURITY.md  CONTRIBUTING.md  CODE_OF_CONDUCT.md
      IMPLEMENTATION.md  DECISIONS.md  pyproject.toml
      .github/workflows/ci.yml
      packages/
        loopgepa_core/    # pure logic: models, storage, runner, mutation, diff, safety
        loopgepa_mcp/     # thin MCP stdio server wrapping core
        loopgepa_plugin/  # Claude Code plugin: skills, agents, hooks, .mcp.json, bin
      examples/bugfix-loop/
      docs/
      scripts/

## 4. Tech stack

- Python 3.12+, **uv** workspace.
- **Pydantic v2** at every boundary.
- Official **`mcp` / FastMCP** for the stdio server (pin the version).
- File store: YAML genomes/variants, JSONL traces, JSON indexes.
- Tests: pytest + pytest-cov; ruff; mypy. No network in tests.

## 5. Data models (`loopgepa_core/models.py`)

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

- Root: `LOOPGEPA_HOME`, default `$XDG_DATA_HOME/loopgepa`
  (fallback `~/.local/share/loopgepa`). **Not** cwd (ADR-004.2).
- Layout: `loops/ traces/ variants/ promotions/ config.json`.
  (`frontiers/` is v0.2.)
- Traces: `O_APPEND` one-JSON-object-per-line; concurrent-safe (ADR-004.3).
- Allowlist capture: store summaries only unless the loop sets `capture: raw`
  (ADR-004.1).

## 7. Runner (`runner.py`) — the piece that makes genes real

Minimal v0.1 runner:

1. Load genome + selected variant.
2. Execute steps in order. A worker step invokes Claude Code (headless
   `claude -p` or a subagent) with the step's `skill`/`agent`/`prompt` genes.
3. Enforce runner-backed genes: retry up to `retry_policy.max_retries` on the
   declared conditions; stop when `stop_condition` is met; prune context per
   `context_policy`; abort on `budget_policy`.
4. Run `verifier_rubric` as a verification step; derive `final_outcome`
   (`success` if stop condition + verifier pass; `failure` if budget/retries
   exhausted; else `unknown`).
5. Emit a `LoopTrace` via `loopgepa_record_trace`.

## 8. MCP tools (v0.1 only)

`loopgepa_run_loop`, `loopgepa_record_trace`, `loopgepa_summarize_traces`,
`loopgepa_infer_failure_modes` (deterministic; no LLM inside the server),
`loopgepa_propose_mutations` (accepts Claude's structured proposal, validates
target genes, saves draft variant, returns diff summary),
`loopgepa_list_variants`, `loopgepa_promote_variant` (dry-run default; writes
only on explicit approval; reconciles against live files per ADR-004.5; records
promotion history).

Deferred to v0.2: `loopgepa_score_variant`, `loopgepa_get_pareto_frontier`.

## 9. Claude Code plugin

- `.claude-plugin/plugin.json`, `.mcp.json` → `loopgepa-mcp`.
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
    M3  mutation.py + diff.py (apply to genome, render diff) + tests.
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
7. All data stays local (XDG dir); works offline after install.
8. Nothing is ever labeled "optimized" without evaluation.

## 13. Non-negotiables

Local-only default · no telemetry · no silent promotion · no silent safety
weakening · reversible only · explicit approval before writes · ≤3 genes per
proposal · show diffs first · low trace count → low confidence · diagnosis
before suggestion · never call a proposal "optimized" until evaluated (v0.2).
