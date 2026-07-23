# Pre-registration — cross-family critic loop-level GEPA study (run 6, D1)

**Registered:** 2026-07-23, *before* the paid run (this commit precedes any results).

**Follow-up to** the five-run null (`docs/validation-study-summary.md`) and its measured
mechanism: the intrinsic self-critique is a near-random failure detector (F1 ≈ 0.11), so
evolving its rubric cannot help. The literature says *why* and points to the fix: a judge
from the **same family as the generator** inflates agreement by 5–7% and carries a
self-preference bias, and **cross-family** verification is more reliable (Valmeekam et al.
2023, arXiv:2310.08118; LLM-as-judge 2026 surveys; "Reliability without Validity"
arXiv:2606.19544). If a more reliable critic is what the loop lacks, giving the critique
step a *different model* than the generator is the most theory-driven way to make the
loop-level lever bite.

## ⚠ Cross-family caveat (read this)

We run this on **OpenAI models only**: the critique routes to `openai/gpt-5.6-luna` while
the worker generates with `openai/gpt-4.1-mini`. This is **cross-lineage** (a different
model generation verifying), which breaks the *self-grading* failure mode (the critic is
not the generator and does not share its exact weights/preferences) — but it is **NOT a
true cross-provider family test**. gpt-5.6 and gpt-4.1 share a vendor, tokenizer lineage,
and likely training-data and stylistic priors, so residual self-preference bias may remain.
A stronger follow-up — critic = `gemini/gemini-2.5-flash-lite` (cheapest true cross-family,
$0.10/$0.40) or `anthropic/claude-haiku-4-5` — is pre-priced in `MODEL_PRICES` and needs
only that provider's API key. Interpret a null here as "cross-lineage is insufficient",
not "cross-family cannot help"; interpret a win as strong evidence for the reliable-critic
hypothesis.

## Hypothesis

**H1 (directional):** With the critique run on a different-lineage model
(`gpt-5.6-luna`), loop-level GEPA (arm D, evolves `generate` + `ensure_correct`) achieves a
higher mean held-out `mc_score` than prompt-only GEPA (arm B, evolves only `generate`).
Null (H0): mean(D) ≤ mean(B), or any gap is within seed noise.

**H2 (secondary, observational):** the cross-lineage critic is a better failure detector
than the same-model intrinsic critique of run 3 (measured post-hoc from the cassette with
`scripts/analyze_critique_detector.py`, extended to read the `critic` call namespace).
Reported, not used for the verdict.

## Frozen configuration (do not change after registration)

- **Task set:** `--task-set multi-hard` (same generator as run 3).
- **Critic:** `--critique-model openai/gpt-5.6-luna` (critique step only; generate/refine
  stay on the worker).
- **Arms:** A / B / C / D (identical seed genome, matched budget).
- **Seeds:** 0, 1, 2 (**n = 3** — lower-powered exploratory run; a clear D > B warrants a
  5-seed, true-cross-family confirmation; a null means "no signal at this power/lineage").
- **Tasks:** n_train = 8, n_heldout = 10 per seed (disjoint).
- **GEPA budget:** max_metric_calls = 20/arm; reflection_calls = 4; loop_max_retries = 1
  (1 + 2·1 = 3 worker-side calls/eval; the 1 critique/eval routes to the critic model).
- **Models:** worker `openai/gpt-4.1-mini` (320), critic `openai/gpt-5.6-luna` (320),
  reflection `openai/gpt-5.6-terra` (1500).
- **Budget:** conservative estimate ≈ $4.32 (all worker-side calls priced at the pricier
  critic model — an upper bound; only ~1/3 actually hit the critic, so real spend is far
  lower, ~\$0.5); `--max-usd 5`; the key's **$5 hard cap** is the true ceiling; resumable
  from the cassette.

## Expected seed band and flags

Pre-registered band **0.2 ≤ seed ≤ 0.85**. The seed loop uses the same cross-lineage critic,
so its held-out may differ from run 3's 0.765. Ceiling (≥ 0.9) or floor (≤ 0.1) fires the
`report.py` flag → H1 untestable, reported as-is.

## Verdict rule (decided in advance)

Unchanged `eval/report.py` rule: **SUPPORTED** iff mean(D) − mean(B) > summed per-arm
stdev and positive; **INCONCLUSIVE** if positive-but-within-noise, < 2 seeds, or a
ceiling/floor flag fires; **NOT SUPPORTED** iff mean(D) ≤ mean(B). Controls A/C expected flat.

## Commitments

- Report **as-is**, whatever it is; publish alongside runs 1–5.
- No post-hoc changes to arms, budget, seeds, task distribution, critic model, or the
  verdict rule.
- Outputs to separate paths: report `docs/validation-study-crossfamily.md`, raw `.json`,
  cassette `tests/fixtures/cassettes/study-crossfamily.json` — earlier runs untouched.
