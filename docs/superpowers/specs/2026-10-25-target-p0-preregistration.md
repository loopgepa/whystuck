# Pre-registration: p₀ levels built by measurement, on a second worker (Paper I, D9 (ii))

Registered 2026-10-25, before any spend on this program. Budget authorised by the credential's
owner: **$60** on the Groq key. Every number below that the study will report is defined here
first; anything reported that is not defined here will be labelled exploratory.

## Why this run, and why now

Paper I's answer is that loop-level optimization beats prompt-only *conditionally on p₀*, the
base rate of first drafts that already satisfy every constraint. Its evidence is one swing on one
worker at n = 3 seeds per level: Δ(D−B) = −0.042 at p₀ = 0.067 and +0.060 at p₀ = 0.267, built
by excluding constraint types from the generator. Paper I's own D9 names two problems with that
evidence, and both are what this run addresses:

1. **Exclusion is not a p₀ knob.** It re-assigns which constraints are compulsory, so each level
   names a different generator; on the llama-3.3-70b worker the same three levels gave p₀ = 0.095,
   0.905, 0.810, a non-monotonic "ladder". The registered Groq Phase 2 (2026-08-08) was therefore
   gated off, and the conditional lever has never been measured on a second worker.
2. **The high level saturated**, because a difficulty knob raises the baseline with p₀. The
   design problem D9 states is a knob that raises p₀ *without* raising the baseline.

D9's candidate (ii) is to measure p₀ per task and sample tasks to a target. This run does that,
and adds the constraint the paper says is missing: every level is matched on the seed baseline.
Mixing tasks that are always right first time with tasks that stay wrong after repair raises p₀ at
a fixed mean final score, which is the separation exclusion could not make.

## What this run is not

- **It does not extend any released curve.** The worker is new. Groq retired
  `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` (both `model_not_found` on 2026-10-25), so
  the released Groq line cannot be continued on its own worker either.
- **It is not the openai stack.** A different worker has a different ceiling; we predict signs and
  orderings, never magnitudes carried over from Paper I.

## Models (verified by call on 2026-10-25)

| role | model | why |
|---|---|---|
| worker (draft, critique, refine) | `groq/qwen/qwen3.8-27b` | default mode answers without a reasoning trace (12 completion tokens on a constraint probe) and misses constraints, so there is headroom. Preview tier, $0.80 / $4.00 per M |
| GEPA reflection | `groq/openai/gpt-oss-120b` | reasoning model; at the 3000-token reflection cap its reasoning (~1000 chars on a probe) leaves room for content. $0.15 / $0.60 per M |

Rejected, and recorded so the choice is a decision: `gpt-oss-20b` and `gpt-oss-120b` as worker
(both returned **empty content** at the 320-token worker cap, `finish_reason: length`, even at
`reasoning_effort=low`); `qwen3.6-27b` (default mode emits an inline `<think>` block that fills the
cap). Sampling temperature: the provider's default, as in every committed run.

## Phase 0: pilot calibration, a GATE ($2.42 bound)

`calibration_pool(120, seed=1000)`: 120 hard-tier tasks from the stock generator (4–5 constraints,
`exact_word_count` forced into every task), ids `p0cal-1000-*`, laid out as six 20-task arm-A
"seeds" in manifest level `calib`. Three arm-A runs with `--sample 0`, `1`, `2`. Per task,
`scripts/p0_task_stats.py` replays the cassettes and reports `p_first` (share of the K = 3 first
drafts that pass every constraint) and `final` (mean score after the loop).

**Gate:** Phase 1 is bought only if the pilot pool can supply all three levels below at a common
baseline in a pool scaled to the full study. Operationally: `select_levels` succeeds on the 120
pilot tasks for the three targets at 1 seed with n_train 8 and n_heldout 10 (54 tasks), matched to
the rule in Phase 2. If it fails, we report that this worker cannot host a baseline-matched ladder
and stop, or drop the infeasible level and say so.

## Phase 1: full calibration (bound ≈ $15, expected ≈ $3)

Pool size N chosen from the pilot so the high level is feasible at 10 seeds with a 50 % margin:
`N = ceil(1.5 × 180 / share)`, where `share` is the pilot fraction of tasks with `p_first ≥ 2/3`,
capped at 1000 tasks. Generator seed 1001, so no pilot task is reused. K = 3 salts.

## Phase 2: level construction (offline, $0)

`select_levels` (src/whystuck/eval/task_manifest.py), `rng_seed = 20261025`, on the Phase 1 stats:

- **Targets:** p₀ ∈ {**low 0.07, mid 0.27, high 0.50**}. The first two reproduce Paper I's two
  measured points; the third is the point Paper I could not reach unsaturated.
- **Baseline B\*:** the mean `final` over the whole Phase 1 pool. Every train and held-out split
  of every level and seed must land within **±0.03 of B\*** in mean `final` and within **±0.03 of
  its target** in mean `p_first`. If any level is infeasible at B\*, B is moved to the nearest
  value in [B\* − 0.05, B\* + 0.05] (one common B for all levels) at which all three are
  feasible. If none exists, the high level is dropped and that is reported as a result about the
  worker, not hidden.
- **Disjointness:** no task appears in two levels, seeds or splits (enforced at manifest build and
  load). Seeds are therefore independent draws; Paper III's defect 5 cannot occur.
- 10 seeds per level, `n_train = 8`, `n_heldout = 10`. Levels filled in the order high, mid, low,
  because the high level competes for the scarcest tasks.

The manifest is committed before Phase 3 is run.

## Phase 3: the four-arm runs (bound $9.83 per level, $29.49 total)

Per level: `--task-set multi-hard-manifest --manifest-level <L> --arms ABCD --seeds 0-9
--n-train 8 --n-heldout 10 --max-metric-calls 28 --reflection-calls 6 --module-selector all
--loop-max-retries 1`. This is the configuration of Paper I's p₀ runs (`c2b`, `p0-L2`, `p0-L3`)
with only the worker, the reflector, the task source and the seed count changed.

Known underpricing, registered: `--reflection-calls` caps *proposals*, and arm D under
`module_selector=all` issues one reflection call per gene per proposal, so reflection calls can
reach 2× the priced 60 per level. At gpt-oss-120b prices that is < $0.30 per level.

## Quantities and verdict rules

Per level L and seed s: Δ_L,s = D.opt_heldout − B.opt_heldout. Δ_L = mean over s.

**Manipulation checks (must pass, or the level is reported as not built):**
- M1, realized p₀: measured from each level's own arm-A first-critique events (fresh salts, not
  the calibration draws). Must order low < mid < high. Regression to the mean is expected and
  the realized values, not the targets, are what get reported.
- M2, matched baseline: arm A held-out means of the three levels must lie within 0.05 of each
  other. If the high level's arm A exceeds 0.95, it is flagged as saturated.
- M3, treatment applied: `genes_changed` includes `ensure_correct` for arm D on ≥ 5/10 seeds per
  level, as Paper III requires before a contrast is read.

**Primary hypothesis H1 (Paper I's answer):** the swing S = Δ_mid − Δ_low > 0.
Test: seeds are independent within and across levels, so S's 95 % interval is a percentile
bootstrap (10 000 resamples, seed 0) over seeds resampled independently within each level, plus
Welch's t. **Supported** if the interval excludes 0 with S > 0. **Contradicted** if it excludes 0
with S < 0. Otherwise **inconclusive**, and we will say the run could not detect a swing of the
size Paper I reports (0.102), with the interval stating what it rules out.

**Secondary H2 (monotonicity, Paper I's falsified prediction (iii), re-tested unsaturated):**
Δ_low ≤ Δ_mid ≤ Δ_high, read from point estimates, with S_high = Δ_high − Δ_low and its interval
reported. Monotonicity failed in Paper I only because L3 saturated; if it fails here with M2
passing, that is a real falsification.

**Secondary H3 (mechanism):** the evolved `ensure_correct` gene's PASS rate on held-out first
critiques rises with level (Paper I: 0/3 → 1/3 → 2/3 seeds with PASS > 0). Reported as seeds with
PASS > 0 per level and pooled false-PASS count.

**Secondary H4 (decomposition):** Paper I's identity Δ = draft term + loop term, per level. Paper I
found the treated share 40 % at low p₀ and 100 % at mid.

The noise note carried from Paper I: σ(Δ) = 0.0291 there is re-scoring noise of a fixed genome and
omits search-path variance. Intervals here are over independent seeds, so they include it.

## Stopping and deviations

No seeds are added after looking. If the key's spend reaches $50 cumulative, everything stops and
what was bought is reported as bought. Any deviation from this file is recorded in a dated section
appended below, never by editing the text above.

## Deviation 1, 2026-10-25, during Phase 0 (before any level was built)

Pilot salt 2 aborted twice on the harness's empty-completion guard ("empty completion twice in a
row"). Probing the second prompt directly, ten identical calls to `qwen/qwen3.8-27b` returned a
one-token empty answer four times (`finish_reason: stop`, no reasoning field). The guard was
written for reasoning models that exhaust `max_tokens`; here an empty answer is an ordinary draw.

Change: every call in this program from here on runs with `--empty-attempts 6` (new flag; the
default stays 2, which every committed run used). An empty answer is still never passed
downstream; it is resampled, and each resample is counted into the run's metadata
(`empty_resamples`), because resampling conditions the output on being non-empty. Salts 0 and 1
were recorded under the default 2 and completed; any single empty they met was resampled once and
is not counted. Salt 2 resumes from its partial cassette under 6.

## Deviation 2, 2026-10-25: the Phase 0 gate failed on the hard tier, and what replaces it

**Registered result.** On `qwen/qwen3.8-27b` the 120-task hard-tier pilot (K = 3 salts) gives pool
p₀ = **0.019** and mean final **0.723**. 114 of 120 tasks never passed on a first draft; 5 passed on
one draw in three and 1 on two. No selection from this pool can reach mean p₀ 0.27 over 18 tasks, so
the mid and high levels are infeasible and the gate fails as written. This worker cannot host a
baseline-matched ladder on the hard tier. (`docs/p0-manifests/pilot-1000-stats.json`.)

**A worker swap does not rescue it.** An exploratory probe (labelled as such: first drafts only,
24 pilot tasks, seed `generate` gene, `max_tokens` 2000, `reasoning_effort` low) gave
`gpt-oss-120b` 24/24 first drafts passing and `gpt-oss-20b` 22/24. The two families bracket the band
from opposite sides: qwen cannot count to 15–25 words, a reasoning model always can.

**Replacement design, registered before its pilot is run.** Hold ONE generator fixed and let task
selection be the only knob: the pool is the hard tier with `exact_word_count` excluded (Paper I's
L2 generator, where its p₀ was 0.267 on gpt-4.1-mini), generator seed 1000 for the pilot and 1001
for the full pool, ids `p0cal-<seed>-x-exact_word_count-*`. Every level of the study draws from this
one generator, so the D9 objection to exclusion (a different generator per level) does not apply:
exclusion here defines the pool, not the level. Everything else in Phases 0 to 3 is unchanged:
same targets, tolerances, baseline rule, gate, seeds, arms and verdict rules, applied to this pool.

The cost of the change, stated now: results speak about the L2 generator only, and a constraint mix
can still differ between levels, because selecting for p₀ may select for some constraint types. The
level-by-level constraint composition will be reported next to the results.

## Deviation 3, 2026-10-25: two fixed generators, share matched, before Phase 1

**Registered result of the deviation-2 pilot.** The L2 generator alone (120 tasks, K = 3) gives pool
p₀ = **0.736** and mean final **0.936**: 54 tasks right on every first draft, 5 never. Its low end
is too thin for a 0.07 level and its baseline sits at the M2 saturation flag. So on qwen3.8 the hard
tier is all floor (p₀ 0.019) and the L2 tier nearly all ceiling, and neither alone hosts a ladder.
Per task, `p_first` and `final` correlate at r = 0.795 over the union of the two pilots and 0.556
within L2: on this worker, raising p₀ by choosing tasks drags the baseline with it unless the
selection holds the baseline explicitly.

**Design for Phases 1 to 3.** The pool is the union of two fixed generators, the hard tier and the
L2 tier, both at generator seed 1001: **1000 hard-tier and 2000 L2-tier tasks**, K = 3 salts.
`select_levels` matches three things in every split: p₀ to target (±0.03), mean final to one
common B (±0.03), and the **share of L2-tier tasks to 0.5 (±0.07)**, so a level cannot reach its
p₀ by changing its generator mix. Sizes come from a $0 simulation that bootstrap-inflates the two
pilots: 800 + 1600 was the smallest tested size at which all three levels were feasible at 10
seeds for B ∈ {0.78, 0.82} (600 + 1200 was not), and 1000 + 2000 adds a 25 % margin.

Changes to Phase 2 rules, each forced by the pilots: levels are filled **low, mid, high** (low is
the scarcest under the share constraint; the registered order high-first was chosen when high was
believed scarcest). **B\*** is the mean `final` of a 50/50 mixture of the two pools, i.e. the mean
of the two pool means, consistent with the matched share; the registered fallback (nearest common B
within ±0.05) is unchanged. Targets, tolerances on p₀ and baseline, seeds, arms, budgets and every
verdict rule are unchanged.

Expected Phase 1 spend ≈ $9 (≈ 24 000 worker calls at the pilot's measured ≈ $0.0004 per call).

## Phase 1 and 2 record, 2026-10-26 (before any Phase 3 call)

Phase 1 calibrated 1000 hard-tier and 2000 L2-tier tasks at K = 3 (about 30 000 worker calls, 0 empty
outputs stored). Pool p₀ 0.023 and 0.718, pool mean final 0.7215 and 0.9318. Two runs died on
Groq's output-token rate limit and one on a VPN reconnect; all resumed from their cassettes, and
the harness now waits out rate limits (commit 0b05e88).

Phase 2 (`scripts/build_p0_levels.py`) found every level feasible at the registered B\* = 0.8266,
so no fallback was used. Mean over seeds of the achieved per-split values:

| level | split | p₀ | final | L2 share |
|---|---|---|---|---|
| low | train / held-out | 0.083 / 0.080 | 0.817 / 0.818 | 0.500 / 0.500 |
| mid | train / held-out | 0.287 / 0.280 | 0.815 / 0.822 | 0.500 / 0.500 |
| high | train / held-out | 0.500 / 0.500 | 0.823 / 0.822 | 0.500 / 0.500 |

The gate passes. Manifest `docs/p0-manifests/study.json`, sha256 prefix `1b5289c9d1ce5b84`,
540 tasks. These are calibration-draw values; M1 and M2 are judged on the runs' own fresh draws.

## Outcome, 2026-10-26

Phase 3 ran all three levels. M1 failed (realized p₀ 0.30 / 0.30 / 0.38), so by the rule above the
levels are reported as not built and H1 is not a test of p₀. The full record, the regression-to-
the-mean diagnosis and an explicitly exploratory pooled estimate are in `docs/p0target/REPORT.md`.
No seeds were added and no further phase was bought.

## Errata and review notes, 2026-10-26 (after the outcome; nothing above is edited)

An adversarial review of the program found every reported number reproducible and the deviations
committed before the runs they govern. Corrections to this file's own record:

- The Phase 1 record says "about 30 000 worker calls". The six Phase 1 cassettes store 23 264
  entries; the 30 000 figure counted the pilots and Phase 3 as well.
- The spend estimate used a 300-token-per-call input bound that is not an upper bound for
  Phase 3 reflection-heavy runs. An independent recompute gives $12.4 to $12.9 for the program.
- Deviation 3 promised "the level-by-level constraint composition will be reported next to the
  results". It was not, in the first outcome report. It is now (`docs/p0target/REPORT.md`,
  `analysis.json` → `diagnosis.composition`): within the L2 tier four constraint types differ in
  frequency across levels at χ² p ≤ 0.003, so the levels differ in constraint mix as well as in
  calibrated p₀.
- The registered hard-tier candidate (ii) never ran (its gate failed); the two-generator design of
  deviation 3 is what produced the outcome. The outcome report now says so in its first section.
