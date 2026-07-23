# Pre-registration — oracle-critique ceiling study (run 7, D6)

**Registered:** 2026-07-23, *before* the paid run (this commit precedes any results).

**Purpose.** This is a deliberate **ceiling / sanity** arm, not a D-vs-B test. The five-run
null could have two explanations: (a) the loop *mechanism* is incapable of turning a good
detector into higher held-out score, or (b) the loop is fine and the intrinsic critique
simply is not a good detector (our measured F1 ≈ 0.11 supports (b)). Run 7 discriminates:
we replace the LLM self-critique with the **true programmatic metric** — the oracle. This is
exactly Huang et al.'s "with oracle" condition (arXiv:2310.01798): they show self-correction
gains that vanish without oracle labels. If the loop CAN use a perfect detector, its seed
held-out should jump well above the intrinsic (run 3, 0.765) and tool-fed (run 4, 0.837)
baselines, completing the detector-quality → performance curve:
intrinsic (F1≈0.11) → tool (F1≈0.83) → oracle (F1=1.0).

## What "oracle critique" means

The critique gate uses `mc_score(draft) ≥ 1.0` as its PASS/FAIL and `mc_feedback(draft)` as
the fix list, instead of an LLM self-check. There is **no critique LLM call** (1 + retries
calls/eval), and the `ensure_correct` verify gene is **inert** (never sent to any model).
This intentionally consults the scorer inside the loop — the opposite of every other run —
which is why it is a clearly-labeled control, not a headline result.

## Hypothesis

**H1 (ceiling):** the oracle-critique seed loop scores substantially higher on held-out
`mc_score` than the intrinsic run-3 seed (0.765), demonstrating the loop mechanism can
convert a perfect detector into higher performance. Pre-registered success threshold:
oracle seed ≥ 0.85 (above the run-4 tool baseline). A result near or below the intrinsic
baseline would instead indicate a loop-mechanism limit (explanation (a)).

**Not tested:** Δ(D − B). Under a perfect oracle the verify gene is inert, so D cannot
differ from B by construction; we report the arm table but the meaningful readout is the
*seed* held-out (the ceiling), compared across runs 3/4/7.

## Frozen configuration (do not change after registration)

- **Task set:** `--task-set multi-hard` (same generator as run 3).
- **Gate:** `--oracle-critique` (no critique LLM call; verify gene inert).
- **Arms:** A / B / C / D (identical seed genome; A's seed held-out is the key number).
- **Seeds:** 0, 1, 2 (n = 3).
- **Tasks:** n_train = 8, n_heldout = 10 per seed (disjoint).
- **GEPA budget:** max_metric_calls = 20/arm; reflection_calls = 4; loop_max_retries = 1
  (1 + 1·1 = 2 worker calls/eval; no critique call).
- **Models:** worker `openai/gpt-4.1-mini` (320); reflection `openai/gpt-5.6-terra` (1500).
- **Budget:** conservative estimate ≈ $2.28; `--max-usd 5`; key's $5 cap is the true
  ceiling; real spend far lower (~\$0.25); resumable from the cassette.

## Verdict framing (decided in advance)

Report the oracle seed held-out and place it on the detector-quality curve next to runs
3 and 4. The `report.py` ceiling/floor flag still annotates the arm table. We expect a
ceiling flag to fire (oracle likely pushes the seed high) — that is the intended outcome
of a ceiling arm, not an inconclusive failure.

## Commitments

- Report **as-is**; publish alongside runs 1–6.
- No post-hoc changes to configuration or framing.
- Outputs to separate paths: report `docs/validation-study-oracle.md`, raw `.json`,
  cassette `tests/fixtures/cassettes/study-oracle.json` — earlier runs untouched.
