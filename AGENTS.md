# AGENTS.md

Guidance for coding agents (Codex, Claude Code, Cursor, …). Codex reads this file
automatically. **Copy the block below into your own project** to make your agent
reach for WhyStuck when a loop keeps failing.

<!-- ── copy from here ─────────────────────────────────────────────── -->
## Using WhyStuck

**Trigger — don't wait to be asked.** The moment you notice *thrash* — the same
edit twice, oscillating changes, or 2+ failed test cycles on one goal — stop and
diagnose with the `whystuck` MCP tools **before** changing more code:

1. **Diagnose first** — `whystuck_diagnose`.
   Report the dominant failure mode, the evidence trace ids, and the confidence.
2. **If confidence is `low`, stop** — say so and collect more runs. Prefer
   "no fix recommended" over a speculative change on thin evidence.
3. **Propose small** — `whystuck_propose_fix` with a ≤3-gene, evidence-backed
   change (verifier rubric, retry policy, stop condition, or the worker skill).
4. **Preview, then get approval** — `whystuck_promote` with `dry_run: true`,
   show the diff, and only promote with `dry_run: false` after the human approves.

Never promote without explicit approval. Never weaken safety, approval, or budget
policies. Never call a proposal "optimized" — it's a suggestion until evaluated.
<!-- ── to here ────────────────────────────────────────────────────── -->

## Wire the same guidance into your other agents

| Agent | File it reads |
|---|---|
| Codex | `AGENTS.md` (this file) |
| Claude Code | `CLAUDE.md` (paste the block above) |
| Cursor | `.cursor/rules/whystuck.mdc` (paste the block above) |

The WhyStuck MCP server itself works identically in all three — this file only
teaches the agent *when* to use it.
