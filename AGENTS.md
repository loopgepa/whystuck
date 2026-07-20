# AGENTS.md

Guidance for coding agents (Codex, Claude Code, Cursor, …). Codex reads this file
automatically. **Copy the block below into your own project** to make your agent
reach for LoopGEPA when a loop keeps failing.

<!-- ── copy from here ─────────────────────────────────────────────── -->
## Using LoopGEPA

When a repeated workflow/loop keeps failing (same error, retries that don't help,
a fix that gets re-applied), use the `loopgepa` MCP tools **before** changing code:

1. **Diagnose first** — `loopgepa_summarize_traces` + `loopgepa_infer_failure_modes`.
   Report the dominant failure mode, the evidence trace ids, and the confidence.
2. **If confidence is `low`, stop** — say so and collect more runs. Prefer
   "no fix recommended" over a speculative change on thin evidence.
3. **Propose small** — `loopgepa_propose_mutations` with a ≤3-gene, evidence-backed
   change (verifier rubric, retry policy, stop condition, or the worker skill).
4. **Preview, then get approval** — `loopgepa_promote_variant` with `dry_run: true`,
   show the diff, and only promote with `dry_run: false` after the human approves.

Never promote without explicit approval. Never weaken safety, approval, or budget
policies. Never call a proposal "optimized" — it's a suggestion until evaluated.
<!-- ── to here ────────────────────────────────────────────────────── -->

## Wire the same guidance into your other agents

| Agent | File it reads |
|---|---|
| Codex | `AGENTS.md` (this file) |
| Claude Code | `CLAUDE.md` (paste the block above) |
| Cursor | `.cursor/rules/loopgepa.mdc` (paste the block above) |

The LoopGEPA MCP server itself works identically in all three — this file only
teaches the agent *when* to use it.
