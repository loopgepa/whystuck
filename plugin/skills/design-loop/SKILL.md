---
description: Help the user declare a new repeatable WhyStuck loop (its genome) so it can be run, traced, and later improved. Use when the user wants to define, register, or set up a new repeatable coding-agent workflow or loop.
---

You are helping declare a new loop genome so WhyStuck can run and trace it.

Procedure:
1. Clarify the loop's goal and its steps in plain terms.
2. Draft a genome with a small set of genes:
   - a `skill` gene (the worker instructions),
   - a `verifier_rubric` gene (how success is checked),
   - a `retry_policy` gene (e.g. `max_retries`),
   - a `stop_condition` gene.
   Keep runner-enforced genes (retry/stop/budget) as structured content.
3. Register it with `whystuck_save_loop`.
4. Explain how to run it: the command+verifier runner (`whystuck_run_loop` with a
   worker command and a deterministic verifier command) records clean traces
   offline. Recommend collecting several runs before asking for a diagnosis.

Keep the first genome minimal. Genes can be added later once traces show what
actually fails.
