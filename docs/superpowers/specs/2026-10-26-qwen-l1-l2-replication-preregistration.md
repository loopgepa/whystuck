# Pre-registration: Paper I's L1-versus-L2 contrast, powered, on a third worker

Registered 2026-10-26, before any spend on this program. Budget: the remainder of the $60 Groq
authorisation (about $47 left after the target-p₀ program). Anything reported that is not defined
here will be labelled exploratory.

## Why

Paper I's answer was that loop-level optimization beats prompt-only conditionally on p₀, from one
swing: Δ(D−B) = −0.042 at L1 (hard tier, p₀ = 0.067) and +0.060 at L2 (`exact_word_count`
excluded, p₀ = 0.267), three seeds each, on gpt-4.1-mini. Round-3 review showed the release does
not support it: nine corrected-harness cells at L1 span −0.042 to +0.062 (mean +0.006), the L1
endpoint is the lowest of them, and the L2 cell equals one of them. The author has decided to
withdraw the p₀ headline. This run asks the narrower question the swing was standing in for, with
the power the original lacked: **does the L1/L2 design itself produce a difference in Δ(D−B) on
another worker at ten seeds a level?**

## What this run is not

Not a p₀ dial: as Paper I's D9 says, L1 and L2 are two different generators, so a difference
between them is measured along p₀ and not identified as caused by it, and the baselines are not
matched. Not an extension of any released curve: the worker is qwen3.8-27b, the series' third.

## Design (identical across the two levels except the exclusion)

`--task-set multi-hard` (L1) and `--task-set multi-hard --exclude-constraint exact_word_count`
(L2); `--arms ABCD --seeds 0,1,2,3,4,5,6,7,8,9 --n-train 8 --n-heldout 10 --max-metric-calls 28
--reflection-calls 6 --module-selector all --loop-max-retries 1 --worker-model
groq/qwen/qwen3.8-27b --reflection-model groq/openai/gpt-oss-120b --empty-attempts 6`. This is
Paper I's p₀-run configuration with only the worker, the reflector and the seed count changed.
Seeds regenerate the task pool, so they are independent draws. Bound $9.9744 per level.

Expected from the target-p₀ calibration on this worker (pool means, not this run): L1 p₀ ≈ 0.02,
baseline ≈ 0.72; L2 p₀ ≈ 0.72, baseline ≈ 0.93.

## Quantities and verdict rules

Per level and seed, Δ = D.opt_heldout − B.opt_heldout. Primary: **S = mean Δ_L2 − mean Δ_L1**,
95 % percentile bootstrap resampling seeds independently within level (10 000 resamples, seed 0,
`scripts/analyze_p0target.swing`), with Welch's t.

- **Manipulation checks.** M1: realized p₀ (arm A first drafts on held-out, replayed) is higher at
  L2 than at L1. M2: if arm A's held-out mean at L2 exceeds 0.95 the level is flagged SATURATED
  and S is reported as uninformative about any gain at L2. M3: arm D changed `ensure_correct` on
  at least 5 of 10 seeds per level.
- **H1 (Paper I's direction).** SUPPORTED if the interval excludes 0 with S > 0; CONTRADICTED if
  it excludes 0 with S < 0; otherwise INCONCLUSIVE, reported with what the interval rules out.
- **Secondary, descriptive.** Each level's Δ with its own bootstrap interval; the evolved
  critique's PASS rate and false PASSes on held-out first critiques (Paper I's mechanism).

## Stopping

No seeds are added after looking. If cumulative Groq spend reaches $55, everything stops and what
was bought is reported as bought. Deviations are appended below, dated, never edited in place.

## Execution, 2026-10-26 (appended after the runs; nothing above is edited)

Both levels ran with the design above. L2 lost three attempts to provider timeouts
(`litellm.Timeout: Connection timed out after 600.0 seconds`); each attempt resumed from the
cassette, no result was read between attempts, and the fourth attempt finished. This changes no
design parameter. Calls in flight when an attempt died were bought and left no cassette entry.

## Outcome, 2026-11-04

M1, M2 and M3 pass at both levels (realized p₀ 0.04 and 0.76; arm A held-out 0.721 and 0.936;
`ensure_correct` changed on 8 and 7 of 10 seeds). H1 is INCONCLUSIVE: S = −0.0100, 95 % CI
[−0.0575, +0.0355], Welch t = −0.40 (df 15.2, p = 0.70). The interval excludes a swing above +0.036,
but arm B scores 0.957 at L2, which caps any gain there at 0.043, so the run could not show an L2
gain of Paper I's size. Both levels replay at $0 with 0 mismatches. Spend floor $2.23. No seeds were
added. The full record, including exploratory per-arm PASS counts labelled as such, is in
`docs/qwen-l1l2/REPORT.md`.

## Errata, 2026-11-04 (nothing above is edited)

- The "Execution" heading carries the date of the runs; the section itself was written on
  2026-11-04, with the outcome.
- The outcome's "excludes a swing above +0.036" is the registered percentile bootstrap. A Welch t
  interval for S is [−0.064, +0.044]. Given the L2 ceiling, the design could hardly have returned
  SUPPORTED; the report says so.
