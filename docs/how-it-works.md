# How it works

## Where WhyStuck sits: loop engineering

In mid-2026 the AI-dev skill ladder gained a new rung:

| Discipline | Unit of work |
|---|---|
| Prompt engineering | the words you send |
| Context engineering | everything the model sees |
| Harness engineering | the environment the agent runs in |
| **Loop engineering** | the repeating **act → observe → verify → retry → stop** cycle |

Loop engineering is *designing the system that prompts the agent for you* — it
discovers work, hands tasks to (sub-)agents, verifies results, keeps state, and
decides the next action until a goal is met. Claude Code, Codex, and Devin are
themselves loops.

**WhyStuck is the diagnosis-and-repair layer for loop engineering.** Where a loop
scaffolder helps you *build* a loop, WhyStuck answers the next question: *this
loop keeps failing — why, and what is the smallest reversible change that helps?*

## The loop genome

A loop is a declared **genome** — a set of evolvable *genes*:

```text
skill · verifier_rubric · retry_policy · stop_condition · context_policy · budget_policy · hook
```

Genes are either **file-backed** (`skill`/`agent`/`prompt`/`hook` — they live in
files your agent reads) or **runner-backed** (`retry`/`stop`/`context`/`budget` —
enforced by the WhyStuck runner and persisted in the genome). This is what makes
every knob real: WhyStuck runs the loop, so it can both observe and change it.

## The pipeline

```text
1. run / record   run a declared loop (worker + verifier), or ingest a session
                  ──▶ a clean local trace (JSONL, in $XDG_DATA_HOME/whystuck)
2. diagnose       cluster failures, rank them, attach DETERMINISTIC confidence
3. propose        the agent reflects on the diagnosis and proposes a ≤3-gene fix,
                  each carrying evidence trace ids
4. preview        dry-run promotion shows exact diffs and writes nothing
5. approve+apply  you approve → atomic, reversible write + a rollback record
```

### Deterministic analysis, LLM reflection

The split is deliberate:

- **Analysis is deterministic** (no LLM). `infer_failure_modes` groups failing
  traces and computes confidence as a pure function of evidence:

  ```text
  n_traces < 5                          → low     (score capped at 0.4)
  5 ≤ n_traces < 15                     → medium
  n_traces ≥ 15 ∧ cluster_share ≥ 0.5   → high
  ```

  The server recomputes confidence from the referenced traces and **overrides**
  any value a model supplies — so a "high-confidence" fix can never be conjured
  from two runs.

- **Reflection is the LLM's job.** Your agent reads the deterministic diagnosis
  and writes the diagnosis prose + the concrete mutation. This mirrors GEPA's
  "reflective mutation from traces" pillar, scoped to loop genes.

### Safety is structural

- ≤ 3 genes per proposal (keeps a result attributable to a change)
- budget caps can't be raised, dropped, or removed silently
- `safety_locked` genes need an explicit unlock
- promotion paths can't escape the project directory
- dry-run by default; every promotion is reversible via the promotions log
- local-only, no telemetry, stdio transport, stored outside your repo

## What it does *not* do yet (v0.1)

v0.1 **diagnoses and proposes**. It does **not** run experiments, measure that a
fix works, or build benchmarks — so nothing is ever called "optimized." That
measured-evaluation loop is [v0.2](v0.2-eval-benchmark.md), which wraps the
official GEPA optimizer.

## Sources

- Loop engineering overview — [ADTmag](https://adtmag.com/articles/2026/07/01/loop-engineering-emerges-as-developers-put-ai-coding-agents-on-repeat.aspx),
  [Tosea](https://tosea.ai/blog/loop-engineering-ai-agents-complete-guide-2026),
  [Cobus Greyling](https://cobusgreyling.medium.com/loop-engineering-62926dd6991c)
- GEPA (Genetic-Pareto) — [arXiv 2507.19457](https://arxiv.org/abs/2507.19457),
  [gepa-ai/gepa](https://github.com/gepa-ai/gepa)
