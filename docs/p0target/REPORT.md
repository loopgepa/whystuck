# Target-p₀ study on a second worker: result

Pre-registration: `docs/superpowers/specs/2026-10-25-target-p0-preregistration.md` (with its three
deviations, each written before the spend it governs). Worker `groq/qwen/qwen3.8-27b`, reflection
`groq/openai/gpt-oss-120b`, four arms, 10 independent seeds per level, configuration of Paper I's
p₀ runs otherwise. Analysis: `scripts/analyze_p0target.py` → `docs/p0target/analysis.json`.
Spend for the whole program ≈ $13 of the $60 authorised (outputs counted exactly from the
cassettes, inputs at a 300-token-per-call upper bound).

## Registered verdict

**The levels were not built.** Manipulation check M1 fails: realized p₀ on the runs' own draws is
0.30 / 0.30 / 0.38 for low / mid / high, against calibrated 0.08 / 0.28 / 0.50, so the levels do
not order. By the registered rule the contrast between them is not a test of p₀, and H1 is
reported for completeness only: S = Δ_mid − Δ_low = +0.0405, 95 % bootstrap CI [−0.0105, +0.1000],
Welch t = 1.35 (df 16.3, p = 0.195), INCONCLUSIVE.

| level | realized p₀ (M1) | arm A held-out (M2) | D changed `ensure_correct` (M3) | B | D | Δ(D−B) | sd |
|---|---|---|---|---|---|---|---|
| low | 0.300 | 0.821 | 7/10 | 0.8325 | 0.8110 | −0.0215 | 0.0551 |
| mid | 0.300 | 0.8185 | 7/10 | 0.8180 | 0.8370 | +0.0190 | 0.0772 |
| high | 0.380 | 0.8155 | 7/10 | 0.8300 | 0.8405 | +0.0105 | 0.0383 |

M2 passes (arm A within 0.006 across levels) and M3 passes on every level, so the treatment was
applied and the baselines were matched. Only the p₀ manipulation failed.

## Why the levels collapsed: selection on a 3-draw estimate

A beta-binomial prior fitted per generator to the calibration counts predicts what happened. Within
the L2 generator the prior is Beta(3.51, 1.38): tasks are far more alike than three draws make
them look. The L2 tasks the low level needed (0 or 1 of 3 first drafts right) were chosen on noise;
their posterior mean pass rate is 0.506 and they realized 0.50. Level means predicted from the
posteriors are 0.263 / 0.338 / 0.408 against realized 0.30 / 0.30 / 0.38.

With the L2 share held at 0.5, the reachable range of true p₀ on this worker is roughly 0.26 to
0.44, too narrow to separate three levels. Between generators the gap is large (pool p₀ 0.023 hard
tier, 0.718 L2 tier); within a generator it is small. **On this worker p₀ is mostly a property of
the generator, not of the task**, so D9's candidate (ii), sampling tasks to a target p₀, cannot
build a ladder at a fixed generator mix. Doing so would need per-task heterogeneity that this
worker does not show, and dropping the share match reintroduces the generator confound D9 set out
to remove. No further phase was bought.

## Exploratory (not registered): the contrast at p₀ ≈ 0.33

Because every level realized p₀ between 0.30 and 0.38 with matched baselines, the 30 seeds are 30
independent draws of Δ(D−B) at roughly one operating point. Pooled: **Δ = +0.0027, sd 0.0597,
95 % bootstrap CI [−0.0175, +0.0245]**; 12 seeds positive, 10 negative, 8 zero. Seed-level realized
p₀ ranges from 0.1 to 0.7 and correlates with Δ at r = −0.089 (OLS slope −0.035 per unit p₀).

Paper I reports Δ = +0.060 at p₀ = 0.267 on gpt-4.1-mini (n = 3). This interval excludes that
value. It is a different worker, a different generator mix (half hard tier, half L2) and a
different reflector, so it does not contradict Paper I's measurement; it says the gain Paper I
found at that p₀ did not appear on a second worker at a comparable p₀, with ten times the seeds.

## Other registered quantities

- H2 (monotonicity): not monotone (−0.0215, +0.0190, +0.0105); S_high = +0.032, CI [−0.006,
  +0.0715]. Moot given M1.
- H3: the evolved critique emits PASS on 6/8/9 of 10 seeds per level, PASS rate 0.19 / 0.28 / 0.35,
  with 11 / 11 / 13 pooled false PASSes. Unlike Paper I's evolved genes, these accept, and they
  accept wrongly about as often as the seed critique does.
- H4: draft term +0.001 / +0.0345 / +0.005, loop term −0.0225 / −0.0155 / +0.0055.

## What a reader should take from it

1. The p₀ knob D9 asked for does not exist at the task level on this worker. That is a finding
   about the design problem, and it is measured, not argued.
2. Building a level from noisy per-task estimates regresses to the generator mean. Any target-p₀
   design needs either many draws per task or a shrinkage estimate; the registered K = 3 was not
   enough, and the empirical-Bayes check shows by how much.
3. At p₀ ≈ 0.33 on qwen3.8, loop-level optimization neither helped nor hurt measurably over 30
   seeds, with an interval narrow enough to exclude the gain Paper I measured at 0.267.
