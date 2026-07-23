# Pre-registration — complexity (third-module) loop-level GEPA study (run 5)

**Registered:** 2026-07-23, *before* the paid run (this commit precedes any results).

**Follow-up to** run 3 (`docs/validation-study-multihard.md`), the fair null. Run 3's loop
had only two modules (generate, verify) and one refine round — the low-complexity end.
DSPy work (Khattab et al. 2023, arXiv:2310.03714; and follow-ups) reports that optimizing
multiple *interacting* prompt modules pays off **as pipeline complexity grows**, because
the modules must be made to cohere end-to-end — something single-module (prompt-only)
optimization cannot do. Run 5 tests whether adding a third interacting module gives
loop-level GEPA a lever it lacked at two modules.

## The added module — a `plan` gene

With `--plan-gene` the loop becomes **plan → generate → critique → refine**. A new `plan`
gene runs first (seed content: "list each explicit constraint in the task as a short
checklist"), and its output is threaded into the generate prompt. This adds one worker call
per eval (2 + 2·retries total).

- **Arm B (prompt-only):** still evolves only `generate` — one module — even though the
  plan gene exists. This keeps B the genuine single-module baseline.
- **Arm D (loop-level):** evolves the whole genome — `generate` + `ensure_correct` +
  `plan` — three interacting modules to co-tune.

So Δ = mean(D) − mean(B) isolates whole-genome (multi-module) optimization against
single-prompt optimization, at higher structural complexity than run 3.

## Hypothesis

**H1 (directional):** With three interacting modules, loop-level GEPA (D) achieves a higher
mean held-out `mc_score` than prompt-only GEPA (B). Null (H0): mean(D) ≤ mean(B), or any gap
is within seed noise.

## Frozen configuration (do not change after registration)

- **Task set:** `--task-set multi-hard` (same generator as run 3).
- **Complexity:** `--plan-gene` (adds the third module and its plan step).
- **Arms:** A / B / C / D (identical 3-gene seed genome, matched budget).
- **Seeds:** 0, 1, 2 (**n = 3** — lower-powered *exploratory* run; a clear D > B warrants a
  5-seed confirmation; a null means "no signal at this power", not a tight refutation).
- **Tasks:** n_train = 8, n_heldout = 10 per seed (disjoint).
- **GEPA budget:** max_metric_calls = 20/arm; reflection_calls = 4; loop_max_retries = 1
  (worker makes 2 + 2·1 = 4 calls/eval — the extra +1 is the plan step, priced in the guard).
- **Models:** worker `openai/gpt-4.1-mini` (max_tokens 320), reflection
  `openai/gpt-5.6-terra` (max_tokens 1500).
- **Budget:** conservative estimate ≈ $3.78; `--max-usd 5`; the key's **$5 hard cap** is the
  true ceiling. Actual expected far lower (run 3 spent ~0.13× its estimate); resumable from
  the cassette if the cap is hit.

## Expected seed band and flags

Pre-registered band **0.2 ≤ seed ≤ 0.85** (a plan step may modestly help the seed). Ceiling
(seed ≥ 0.9) or floor (≤ 0.1) fires `report.py`'s flag → H1 untestable, reported as-is.

## Verdict rule (decided in advance)

Unchanged `eval/report.py` rule: **SUPPORTED** iff mean(D) − mean(B) > summed per-arm stdev
and positive; **INCONCLUSIVE** if positive-but-within-noise, < 2 seeds, or a ceiling/floor
flag fires; **NOT SUPPORTED** iff mean(D) ≤ mean(B). Controls A and C expected flat (C is
genome-blind, so the plan gene cannot help it either).

## Commitments

- Report **as-is**, whatever it is; publish alongside runs 1–4.
- No post-hoc changes to arms, budget, seeds, task distribution, module set, or the verdict
  rule.
- Outputs to separate paths: report `docs/validation-study-plan.md`, raw
  `docs/validation-study-plan.json`, cassette `tests/fixtures/cassettes/study-plan.json` —
  earlier runs' artifacts untouched.
