# Target-p₀ study on a third worker: result

Pre-registration: `docs/superpowers/specs/2026-10-25-target-p0-preregistration.md` (three dated
deviations, each committed before the spend it governs, and an outcome section). Worker
`groq/qwen/qwen3.8-27b`, reflection `groq/openai/gpt-oss-120b`, four arms, 10 independent seeds per
level, configuration of Paper I's p₀ runs otherwise. This is the series' third worker: the
released second-worker line is `llama-3.3-70b`, which Groq has since retired. Analysis:
`scripts/analyze_p0target.py` → `docs/p0target/analysis.json`; every number below is in that file
or in the calibration stats it reads.

Spend for the whole program ≈ $13 of the $60 authorised. That is an estimate: outputs are counted
from the cassettes, inputs are reconstructed, and an independent recompute in review gave
$12.4 to $12.9.

**Reproducible at $0.** All three levels were replayed under `--replay` from their committed
cassettes on 2026-10-26: 10 seeds × 4 arms per level, `seed_heldout`, `opt_heldout` and
`genes_changed` identical to the released JSON in every record (0 mismatches of 120 per level).

## What ran, and what did not

The registered design, candidate (ii) of Paper I's D9 on the hard tier, never ran: its gate failed
on this worker (pool p₀ 0.019 on the pilot). What ran is the deviation-3 design: levels drawn
half from the hard tier and half from the `exact_word_count`-excluded (L2) tier, matched on
calibrated p₀, on mean final score and on that half-and-half share.

## Registered verdict

**The levels were not built.** Manipulation check M1 fails: realized p₀ on the runs' own first
drafts is 0.30 / 0.30 / 0.38 for low / mid / high, against calibrated 0.08 / 0.28 / 0.50, so the
levels do not order. By the registered rule the contrast between them is not a test of p₀. H1 is
reported for completeness: S = Δ_mid − Δ_low = +0.0405, 95 % bootstrap CI [−0.0105, +0.1000],
Welch t = 1.35 (df 16.3, p = 0.195), INCONCLUSIVE.

| level | realized p₀ (M1) | arm A held-out (M2) | D changed `ensure_correct` (M3) | B | D | Δ(D−B) | sd |
|---|---|---|---|---|---|---|---|
| low | 0.300 | 0.821 | 7/10 | 0.8325 | 0.8110 | −0.0215 | 0.0551 |
| mid | 0.300 | 0.8185 | 7/10 | 0.8180 | 0.8370 | +0.0190 | 0.0772 |
| high | 0.380 | 0.8155 | 7/10 | 0.8300 | 0.8405 | +0.0105 | 0.0383 |

M2 passes (arm A within 0.006 across levels) and M3 passes on every level.

## Why the levels did not separate

**Selection on three draws per task regressed to the tier mean.** A method-of-moments
beta-binomial prior fitted to each tier's calibration counts gives Beta(3.51, 1.38) for the L2
tier and Beta(0.59, 25.44) for the hard tier. On the L2 half, the held-out tasks chosen for the low
level had a calibrated mean of 0.160, a posterior mean of 0.506 and realized 0.50; mid 0.553 /
0.655 / 0.60; high 0.893 / 0.785 / 0.74. The hard half sits near zero throughout (realized 0.10,
0.00, 0.02).

**Tasks do differ within a tier, but less than three draws suggest.** On the runs' fresh draws, L2
tasks that passed 0, 1, 2 or 3 of their calibration drafts passed 0.42, 0.47, 0.63 and 0.82 of the
time. Selection on per-task estimates does move p₀; with K = 3 and a share held at one half, it
moved the realized level means only across 0.30 to 0.38.

**The selection also moved the constraint mix.** The registered composition report: within the
L2 tier, four constraint types differ in frequency across levels (χ² of presence by level, df 2):
`all_lowercase` 0.24 / 0.42 / 0.59 (p = 1.7e-05), `num_bullets` 0.19 / 0.37 / 0.52 (p = 1.9e-05),
`end_with` 0.77 / 0.61 / 0.44 (p = 5.5e-05), `no_comma` 0.66 / 0.54 / 0.40 (p = 2.6e-03). The hard
tier is stable apart from `num_bullets` (p = 0.039). So selecting tasks for p₀ selected constraint
types, and these levels differ in constraint mix as well as in p₀. That is the confound D9 raises
against exclusion, re-created at the task level.

**What this says about D9's candidate (ii).** It is not refuted, but two things it needs were
missing here: enough draws per task (or selection on a shrinkage estimate rather than on the raw
rate), and a constraint-mix match alongside the p₀ match. Whether a mix-matched ladder is even
reachable on this worker is open; at K = 3 it was not attempted, and no further phase was bought.

## Exploratory (not registered): the contrast near one operating point

The 30 seeds share a matched baseline, and the three level means of realized p₀ lie between 0.30 and 0.38.
Pooled: **Δ(D−B) = +0.0027, sd 0.0597, 95 % bootstrap CI [−0.0175, +0.0245]**; 12 seeds positive,
10 negative, 8 zero. Two cautions. The pooled p₀ is a mixture, roughly 0.04 on the hard half and
0.61 on the L2 half, not a homogeneous operating point. And seed-level realized p₀ ranges from 0.1
to 0.7, which is what ten binomial draws per seed produce around those means.

Paper I reports Δ = +0.060 at p₀ = 0.267 on gpt-4.1-mini from three seeds (+0.005, +0.070,
+0.105), whose own t-interval is [−0.066, +0.186]. The pooled interval here does not contain
+0.060, but the two estimates differ by Welch t = 1.83 (df 2.6, p = 0.178). The worker, the
reflector and the generator mix all differ. So this run does not reproduce the gain Paper I
measured, and it is not evidence against it.

## Other registered quantities

- H2 (monotonicity): not monotone (−0.0215, +0.0190, +0.0105); S_high = +0.032, CI [−0.006,
  +0.0715]. Moot given M1.
- H3: the evolved critique emits PASS on 6 / 8 / 9 of 10 seeds per level, PASS rate 0.19 / 0.28 /
  0.35, with 11 / 11 / 13 pooled false PASSes, against 27 / 17 / 17 for the seed critique on the
  same held-out tasks (35 against 61 overall). Unlike Paper I's evolved genes, these accept, and
  they false-accept less often than the seed critique.
- H4: draft term +0.001 / +0.0345 / +0.005, loop term −0.0225 / −0.0155 / +0.0055.
- Empty-completion resampling (deviation 1) touched about 1.3 % of Phase 3 calls (18 / 15 / 20
  resamples), too few to bear on M1.

## What a reader should take from it

1. A target-p₀ level built from three draws per task regresses toward its tier's mean, by an
   amount an empirical-Bayes fit predicts. Any design of this kind needs more draws or a shrinkage
   estimate, and the fit says how many.
2. Selecting tasks on p₀ also selects constraint types. A ladder built that way must match the
   constraint mix too, or it repeats the confound it was meant to remove.
3. Near p₀ ≈ 0.33 on qwen3.8, loop-level optimization neither helped nor hurt measurably over 30
   seeds. That neither reproduces nor contradicts Paper I's three-seed gain at 0.267.
