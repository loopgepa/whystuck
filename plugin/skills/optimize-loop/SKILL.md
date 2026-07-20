---
description: Propose small, reversible improvements to a repeated Claude Code / Codex loop from its WhyStuck traces. Use when the user asks to improve, tune, fix, or evolve a workflow, skill, verifier, retry policy, or stop condition. Always diagnose first, propose ≤3-gene changes, and never promote without explicit approval.
---

You are improving an agent loop, not just a prompt — and only from evidence.

Procedure:
1. Diagnose first: `whystuck_diagnose`.
   If evidence is weak (low confidence / few traces), say so and stop — proposing
   a change on thin evidence is worse than proposing nothing.
2. Map the top failure mode to the smallest useful change to 1–3 genes
   (verifier rubric, retry policy, failure-summary step, stop condition, or the
   worker skill). Prefer the least invasive edit.
3. Call `whystuck_propose_fix` with each proposal carrying its
   `evidence_trace_ids`, a rationale, expected effects, and risks. The server
   rejects >3 genes and any silent budget increase.
4. Show the user the returned diff summary. Then call `whystuck_promote`
   with `dry_run: true` and show the exact file diffs.
5. Only after the user explicitly approves, call `whystuck_promote` with
   `dry_run: false`. Never promote otherwise.
6. Never weaken safety, approval, or budget policies. Never call a proposal
   "optimized" — it is a suggestion until it is evaluated (a later capability).

Prefer "no change recommended" over a speculative change.
