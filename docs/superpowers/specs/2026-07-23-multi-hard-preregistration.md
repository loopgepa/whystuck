# Pre-registration — HARD multi-constraint loop-level GEPA study (run 3)

**Registered:** 2026-07-23, *before* the paid run (this commit precedes any results).
**Follow-up to** two inconclusive runs, each blocked by a different artifact:
- Run 1 (single-constraint `docs/validation-study.md`): the verify gene was inert —
  nothing to decompose, so D could not differ from B.
- Run 2 (easy multi-constraint `docs/validation-study-multi.md`): **seed saturated at
  0.97** — no headroom for any arm.

This run removes the run-2 failure mode by construction.

## Hypothesis

**H1 (directional):** On HARD multi-constraint tasks, loop-level GEPA (arm D, evolves
`generate` + `ensure_correct`) achieves a higher mean held-out `mc_score` than
prompt-only GEPA (arm B, evolves only `generate`) under a matched budget.
Null (H0): mean(D) ≤ mean(B), or any positive gap is within seed noise.

## Why headroom is guaranteed this time (structural)

Every hard task MANDATES `exact_word_count` (a genuine LLM skill gap — models cannot
reliably self-count words during generation) plus ≥1 more brutal constraint
(`forbidden_word` / `word_max_length`). Because a model essentially never hits an
exact word count, the seed is **structurally capped below saturation**: an otherwise-
perfect answer proxies to ~0.68, a careless one to ~0.08, so gpt-4.1-mini is expected
to land ~**0.4–0.6**. Pre-registered expected seed band: **0.2 ≤ seed ≤ 0.85**. If the
seed lands outside that band the run is inconclusive-by-saturation/floor (the report's
`report.py` ceiling/floor flag fires automatically at ≥0.9 / ≤0.1) and H1 is untestable
regardless of the mechanical verdict.

All constraints are individually diagnosable and fixable (the critique can say "you
have 23 words, need 20" or "you used the banned word 'the'"), so if the loop-level
thesis is true, arm D has a real mechanism to win here.

## Frozen configuration (do not change after registration)

- **Task set:** `--task-set multi-hard` (6 shared + exact_word_count / forbidden_word /
  word_max_length; k ∈ {4,5}; exact_word_count + ≥1 more brutal always present;
  satisfiability guaranteed — 19,632-case exhaustive proof).
- **Arms:** A / B / C / D (identical seed genome, matched budget).
- **Seeds:** 0,1,2,3,4 (n = 5).
- **Tasks:** n_train = 8, n_heldout = 10 per seed (disjoint).
- **GEPA budget:** max_metric_calls = 28/arm; reflection_calls = 6; loop_max_retries = 1.
- **Models:** worker `openai/gpt-4.1-mini` (max_tokens 320), reflection
  `openai/gpt-5.6-terra` (max_tokens 1500).
- **Budget:** estimate ≈ $6.2; `--max-usd 8`; the key's own **$5 hard cap** is the true
  ceiling (hard tasks fail more → more refine calls → actual likely higher than run 2's
  $0.67 but expected < $3; if it hits $5 it stops and is resumable from the cassette).

## Verdict rule (decided in advance)

The existing `eval/report.py` rule, unchanged: **SUPPORTED** iff mean(D) − mean(B) > the
summed per-arm stdev and positive; **INCONCLUSIVE** if positive-but-within-noise or < 2
seeds, or if the ceiling/floor flag fires; **NOT SUPPORTED** iff mean(D) ≤ mean(B).
Controls A and C are expected flat.

## Commitments

- Report **as-is**, whatever it is; publish alongside the first two runs.
- No post-hoc changes to arms, budget, seeds, task distribution, or the verdict rule.
- Outputs to separate paths: report `docs/validation-study-multihard.md`, raw
  `docs/validation-study-multihard.json`, cassette
  `tests/fixtures/cassettes/study-multihard.json` — earlier runs' artifacts untouched.
