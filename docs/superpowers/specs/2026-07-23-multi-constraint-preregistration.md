# Pre-registration — multi-constraint loop-level GEPA study

**Registered:** 2026-07-23, *before* the paid run (commit precedes any results).
**Follow-up to:** the single-constraint run (`docs/validation-study.md`), which was
NOT SUPPORTED but on the least-favorable regime (all tasks single-constraint, where
the critique/verify gene has nothing to decompose). This run tests the thesis where
its mechanism should actually matter: multiple, independent, diagnosable constraints.

## Hypothesis

**H1 (directional):** On composable multi-constraint tasks, loop-level GEPA (arm D,
evolves the `generate` + `ensure_correct` genes) achieves a higher mean held-out
`mc_score` than prompt-only GEPA (arm B, evolves only `generate`) under a matched
budget.

Null (H0): mean(D) ≤ mean(B), or any positive gap is within seed noise.

## Frozen configuration (do not change after registration)

- **Task set:** `--task-set multi` — 6 independent IFEval-style constraints
  (include_keyword, min_words, all_lowercase, end_with, no_comma, num_bullets),
  k ∈ {3, 4} per task, satisfiability-guaranteed (every task's witness scores 1.0).
- **Arms:** A (seed baseline), B (prompt-only), C (genome-blind command control),
  D (loop-level). Identical seed genome; matched budget.
- **Seeds:** 0, 1, 2, 3, 4 (n = 5, for statistical power).
- **Tasks:** n_train = 8, n_heldout = 10 (per seed; train/heldout disjoint).
- **GEPA budget:** max_metric_calls = 28 per optimizing arm; reflection_calls = 6
  (hard proposal cap). loop_max_retries = 1 (generate→critique→refine, ≤3 calls/eval).
- **Models:** worker `openai/gpt-4.1-mini` (max_tokens 320), reflection
  `openai/gpt-5.6-terra` (max_tokens 1500).
- **Budget:** pre-flight estimate ≈ $6.2; `--max-usd 8` lets the guard pass, but the
  API key's own **$5 hard cap** is the true ceiling (actual expected ≈ $2–3.6; if it
  hits $5 the run stops and is resumable from the cassette at $0).

## Verdict rule (decided in advance)

Uses the existing `eval/report.py` `_verdict` rule, unchanged:
- **SUPPORTED** iff mean(D) − mean(B) > pooled per-arm std (sum of the two arms'
  population stdev across seeds) AND the gap is positive.
- **INCONCLUSIVE** iff the gap is positive but within that pooled noise, or < 2 seeds.
- **NOT SUPPORTED** iff mean(D) ≤ mean(B).

Expected-but-not-required sanity: controls A and C flat (Δ ≈ 0). If A or C move
materially, the run is suspect regardless of the D-vs-B outcome.

## Commitments

- The result is reported **as-is**, whatever it is — a second null is a legitimate,
  publishable outcome and will be published alongside the first.
- No post-hoc changes to arms, budget, seeds, task distribution, or the verdict rule.
- Outputs are written to **separate** paths so the single-constraint result is not
  overwritten: report `docs/validation-study-multi.md`, raw `docs/validation-study-multi.json`,
  cassette `tests/fixtures/cassettes/study-multi.json` (committed for offline replay).
