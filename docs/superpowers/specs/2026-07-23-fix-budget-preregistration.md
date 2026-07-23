# Pre-registration — fix-budget-under-oracle study (run 8, D1′)

**Registered:** 2026-07-23, *before* the paid run (this commit precedes any results).

**Follow-up to** run 7, which flipped the paper's thesis: a *perfect* in-loop detector
(oracle) did NOT lift the ceiling (oracle seed 0.748, below the tool's 0.837). We concluded
the binding constraint is the **fix step**, not detection. Run 8 tests the most benign
fix-side hypothesis: maybe the fixer just needs **more attempts**. We hold detection perfect
(oracle) and raise the refine budget from 1 to 3.

## Hypothesis

**H1 (fix-attempts):** with a perfect detector and a larger refine budget
(`loop_max_retries = 3`), the seed held-out `mc_score` rises meaningfully above run 7's
oracle-at-1-retry baseline of 0.748. Pre-registered success threshold: seed ≥ 0.85.

**H0 / the interesting null:** if the seed stays near 0.748 even with a perfect detector AND
three fix attempts, the bottleneck is not detection *or* attempts but the generator's raw
**capability** — `gpt-4.1-mini` cannot produce an output satisfying the hard constraints
(e.g. an exact word count) even when told exactly what is wrong, repeatedly. That would be
the deepest form of the loop-level null: no amount of loop optimization helps because the
base model lacks the capability the loop is trying to orchestrate.

## Frozen configuration (do not change after registration)

- **Task set:** `--task-set multi-hard` (same generator as runs 3/7).
- **Gate:** `--oracle-critique` (perfect detector; no critique LLM call; verify gene inert).
- **Fix budget:** `--loop-max-retries 3` (vs run 7's 1) — the single changed variable.
- **Arms:** A / B / C / D (seed held-out of A is the readout, as in run 7; not a D-vs-B test).
- **Seeds:** 0, 1, 2 (n = 3, exploratory), matching run 7 so the 1-vs-3 comparison is clean.
- **Tasks:** n_train = 8, n_heldout = 10 per seed (disjoint) — identical to run 7.
- **GEPA budget:** max_metric_calls = 20/arm; reflection_calls = 4.
- **Worker calls/eval:** 1 + 3 = 4 (oracle, no critique calls).
- **Models:** worker `openai/gpt-4.1-mini` (320); reflection `openai/gpt-5.6-terra` (1500).
- **Budget:** conservative estimate ≈ $3.8 (all calls at worker rate); `--max-usd 5`; the
  key's $5 cap is the true ceiling; real spend far lower (~\$0.4); resumable from cassette.

## Verdict framing (decided in advance)

Report the oracle seed held-out at `retries = 3` next to run 7's `retries = 1` value
(0.748), forming a 2-point fix-budget curve under perfect detection. seed ≥ 0.85 supports
H1 (attempts were the constraint); seed ≈ 0.748 supports the capability-limit null. The
`report.py` ceiling/floor flag annotates the arm table as usual.

## Commitments

- Report **as-is**; publish alongside runs 1–7.
- No post-hoc changes to configuration or framing.
- Reproducible via `--replay` (added this session); results to separate paths: report
  `docs/validation-study-fixbudget.md`, raw `.json`, cassette
  `tests/fixtures/cassettes/study-fixbudget.json` — earlier runs untouched.
