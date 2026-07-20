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
