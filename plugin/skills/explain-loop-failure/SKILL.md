---
description: Explain why a repeated Claude Code / Codex loop keeps failing, using recorded WhyStuck traces. Use when the user asks why a workflow, loop, skill, or agent keeps failing, or wants a diagnosis before any change. Diagnose only — do not mutate.
---

You are diagnosing a repeated agent loop from evidence. Do not propose changes
unless asked; this skill is diagnosis-first.

Procedure:
1. Identify the loop id (ask, or infer from the project).
2. Call `whystuck_diagnose` for the loop (summary + ranked failure modes).
3. Report, grounded in the returned data:
   - number of traces used and the success rate,
   - the top failure modes with their evidence trace ids,
   - the deterministic confidence level (`low` / `medium` / `high`),
   - which loop genes are likely implicated.
5. If the trace count is low, say so plainly and recommend collecting more runs
   before trusting the diagnosis. "No clear failure mode yet" is a valid answer.

Never overstate. A diagnosis is a likely cause supported by traces, not proof.
