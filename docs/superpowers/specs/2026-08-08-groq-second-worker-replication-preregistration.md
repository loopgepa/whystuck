# Pre-registration — a second-worker replication line on Groq

Registered 2026-08-08, before any spend on this program. Budget authorised by the credential's
owner at $400; the program below is priced at **≈ $37** and the reasons for not spending the rest
are recorded here rather than discovered later.

## What this line is, and what it is not

Every run in the released series uses `worker = openai/gpt-4.1-mini`. This program runs on a
borrowed Groq credential, so the worker changes, and **no run here extends any released curve**.
It is a parallel replication line on an independent worker.

That is worth buying because "single worker/reflection model" is an open row in Paper I's
closability table and a stated limitation in all three papers. It is *not* a substitute for the
existing numbers and no paper sentence will present it as one.

**Worker: `groq/llama-3.3-70b-versatile`.** Deliberately not the 8× cheaper
`llama-3.1-8b-instant`: a calibration run on the 8b worker returned an intrinsic critique that
**never emits PASS (0/30)**, i.e. the degenerate always-refine verifier. A stack whose detector
axis has no variance cannot replicate a result about detector quality. With the budget authorised
there is no reason to economise into that corner.

## Models available, and one that is not

Verified against the live `/v1/models` on 2026-08-08 and by call:

| model | usable as critic at the 320-token cap? |
|---|---|
| `groq/llama-3.3-70b-versatile` | yes, 85 completion tokens on a realistic critique |
| `groq/qwen/qwen3.6-27b` | yes, verbose (hits the cap) |
| `groq/llama-3.1-8b-instant` | yes, but degenerate (0/30 PASS) |
| `groq/openai/gpt-oss-120b` | **no** |
| `groq/openai/gpt-oss-20b` | **no** |

Both `gpt-oss` models are reasoning models: on a realistic critique prompt they consume the entire
320-token allowance on reasoning and return an empty completion, which Paper III's blank-completion
guard (Evidence 8) correctly refuses. Measured: 0 characters at `max_tokens=320` even with
`reasoning_effort=low`; 257 characters at 1024 with low effort; 390 at 2048 with default effort.
Using them would require a per-model output cap and would multiply critic output cost 3–6×, and
the cap is the quantity the spend guard prices. **They are excluded from this program** and that
exclusion is registered, not decided after a failure.

The cross-lineage critic is therefore `qwen/qwen3.6-27b` (Alibaba lineage, different vendor of
origin from the Meta worker) and the same-vendor stronger critic is the worker itself at 70b where
a size contrast is wanted.

## Phase 0 — p₀ calibration, and it is a GATE

Paper I's headline is that the loop-level effect is **conditional on p₀**, and its design needs a
wide p₀ span: it measured 0.067 → 0.267 → 0.667, a 10× span, across three exclusion levels. p₀ is
a property of the worker, so the span must be re-measured, never assumed.

Three cheap `--arms A` passes, 20 seeds, at the same three levels (`L1` none,
`L2 -exact_word_count`, `L3 -exact_word_count -word_max_length`). Cost **$1.73**. p₀ is then read
offline from the recorded cassettes at $0 via `measure_p0.py --stack groq`.

**Gate, decided now:** the sweep in Phase 2 is bought only if the measured span satisfies
`max(p₀)/min(p₀) ≥ 3` and `max(p₀) ≤ 0.90`. A worker that saturates the easy level, or whose
levels barely separate, cannot show a conditional effect, and buying the $27 sweep anyway would be
spending to produce a foregone "inconclusive". If the gate trips, Phase 2 is not bought and the
recorded result is that this worker cannot host the comparison.

## Phase 1 — Paper II's detector table on a second worker ($4.90)

Every cell is `--arms A`, so no optimizer runs and nothing here can be an artifact of search.
20 seeds. Cells: intrinsic (70b judging itself), same-lineage smaller (8b), cross-lineage
(qwen3.6-27b), perfect oracle (`--oracle-critique`), and the fix-side levels
(`--feedback-level` lo/mid/hi) that Paper II uses to separate detection from repair.

This also discharges D3's registered *Remaining* item — "a second cross-provider family to show the
degeneracy is not one model's quirk" — since qwen is a third lineage after `gpt-5.6-luna` and
`llama-3.3-70b`.

**Registered prediction (Paper II's thesis):** detector quality will not order the loops. The
oracle, a perfect detector, will not be the best cell. If it is, that is evidence against the
thesis on an independent worker and will be reported as such.

## Phase 2 — Paper I's conditional lever on a second worker ($26.83)

Four-arm `ABCD`, `--module-selector all`, `--max-metric-calls 28`, `--reflection-calls 6`,
`--loop-max-retries 1`, at the three p₀ levels, **20 seeds each** against the released study's 3.

**Registered prediction:** Δ(D−B) is larger at the middle level than at the extremes, reproducing
the sign of the released swing. We do *not* predict the magnitude: a different worker has a
different ceiling.

**This phase is conditional on the Phase 0 gate.**

## Phase 3 — the verdict/prose splice, done properly ($3.17)

The 2×2 registered in `2026-08-08-groq-verdict-prose-splice-preregistration.md`, re-run on this
worker with `qwen3.6-27b` as the cross-lineage critic in place of the reasoning-starved
`gpt-oss-120b` that aborted it, at 20 seeds. `--temperature 0` throughout, which now actually
reaches the critique and verdict calls (it did not until today; see that file's deviation record).

## Budget

| phase | bound |
|---|---|
| 0 — p₀ calibration, 3 levels | $1.73 |
| 1 — detector table | $4.90 |
| 2 — conditional lever, 3 levels (gated) | $26.83 |
| 3 — verdict/prose 2×2 | $3.17 |
| **total** | **$36.63** |

Prior spend on this credential is ≈ $0.10, so the running total lands near **$37**.

## Why not spend the other $363

Recorded now so that declining to spend is a decision on the record rather than an omission.

1. **The task pool caps it.** `multi-hard` holds **64 tasks** and each seed draws 18. Seeds
   resample one small pool, so past roughly 20 seeds the marginal power is small and no budget
   enlarges the pool. Widening the generator pools is D8, and it costs engineering, not dollars.
2. **The remaining sweeps replicate closed results.** A depth sweep ($78.56) and a split-size
   sweep ($35.77) on this worker would re-derive findings Paper I already closed on OpenAI. That
   buys robustness, not a claim, and it is available later at a known price if a reviewer asks.
3. **Throughput.** The credential reports 300 000 tokens/minute. At 70b prices $400 is ≈ 1.5 days
   of fully saturated traffic and several days at achievable utilisation, for results that are
   supplementary to every paper's spine.

## Commitments

- Phase 0 is read **only** against its stated gate. Its numbers do not reselect the design.
- Every phase's cassettes and per-task score vectors are committed, so each replays at $0.
- Predictions above are registered before the data and will be reported as hits or misses.
- Nothing from this line will be presented as extending, correcting or replacing a released
  number measured on the `gpt-4.1-mini` worker.
- If a phase's result contradicts a paper's claim on this worker, that goes in the paper as a
  limitation of the claim's scope, not into a footnote.

## RESULT — Phase 0, and the gate TRIPPED (2026-08-08)

Three calibration passes bought at 20 seeds, $0.5765 each, **$1.73** total.

| level | n | p₀ | intrinsic critique PASS rate | seed-rubric detector |
|---|---|---|---|---|
| L1 none | 200 | **0.095** | 0.01 | tp=0 fp=1 fn=19 tn=180 |
| L2 −`exact_word_count` | 200 | **0.905** | 0.01 | tp=3 fp=0 fn=178 tn=19 |
| L3 −`exact_word_count` −`word_max_length` | 200 | **0.810** | 0.01 | tp=1 fp=1 fn=161 tn=37 |

**The gate was `max(p₀) ≤ 0.90` and `max/min ≥ 3`. max(p₀) = 0.905, so Phase 2 is NOT bought.**
That is the registration doing its job: it declined a $26.83 purchase whose outcome was already
determined.

The deeper reason is worse than the threshold. On `gpt-4.1-mini` the three levels form a ladder,
0.067 → 0.267 → 0.667. On this worker they do not: **0.095 → 0.905 → 0.810**, non-monotonic, with
nothing between 0.095 and 0.810. Dropping `exact_word_count` alone takes p₀ from 0.095 to 0.905,
so on a 70B worker that single constraint is very nearly the *entire* difficulty of the task
family, and every level that relaxes it lands at the ceiling. There is no middle level to measure
a conditional effect at.

Alternative levels cannot be found for free: excluding a constraint changes the task **prompt**, so
the recorded L1 drafts cannot be re-scored under a different exclusion set (`collect_events` raises
`CassetteMiss`, correctly). Each candidate level is a fresh $0.58 pass, and the search space is a
handful of constraint families none of which look promising, since only `exact_word_count` binds.

**Reportable finding, not merely a blocked phase.** Paper I's conditional lever is measured on a
p₀ ladder built by relaxing constraints, and that ladder is a property of the **worker**, not of
the task family. On a stronger worker the same relaxations collapse to a two-point set at the
floor and the ceiling. This is direct empirical support for D9 — "a high-p₀ level that does not
saturate" — being the open design problem Paper I says it is, and it is evidence that the p₀ axis
needs a worker-independent construction before the lever can be replicated anywhere.

**A second finding, free.** The intrinsic critique on this worker emits PASS on **1%** of drafts,
and at L2 demands fixes on 178 of 181 already-correct drafts (fn=178). It is a false-**alarm**
machine, where `gpt-4.1-mini`'s was a false-**pass** machine (precision 0.08). Both are degenerate
and neither is a usable detector. That is now three lineages — `gpt-4.1-mini`, `llama-3.3-70b`,
`llama-3.1-8b` — on which the seed rubric yields a degenerate in-loop verifier.

**Phases 1 and 3 proceed**: neither needs a p₀ ladder. Revised program cost **≈ $10** against the
$36.63 registered, and the $26.83 is not spent.

**Bonus: Phase 1's intrinsic cell is already bought.** `study-groq-p0cal-L1` is exactly the
intrinsic detector cell at L1 — arms A, 20 seeds, worker judging itself — so it is reused rather
than re-purchased.

## RESULT — Phase 1, and the registered prediction FAILED AS WRITTEN (2026-08-08)

Five cells at 20 seeds on `groq/llama-3.3-70b-versatile`, `--arms A` throughout, so no optimizer
ran anywhere. The intrinsic cell is `study-groq-p0cal-L1`, reused rather than re-bought.

| cell | held-out mean | vs intrinsic | se | t |
|---|---|---|---|---|
| intrinsic, own prose (PASS rate 1%) | 0.7895 | — | — | — |
| same-lineage smaller critic (8b) | 0.7860 | −0.0035 | 0.0093 | −0.38 |
| cross-lineage critic (`qwen3.6-27b`) | 0.7973 | +0.0077 | 0.0095 | +0.82 |
| **perfect oracle**, mid prose | 0.8333 | **+0.0437** | 0.0075 | **+5.80** |
| **perfect oracle**, hi prose | 0.8688 | **+0.0792** | 0.0082 | **+9.68** |
| perfect oracle, lo prose (verdict only) | 0.8005 | +0.0110 | 0.0080 | +1.37 |

Paired on seeds (n = 20). The oracle-vs-intrinsic contrast also holds at the task level:
+0.0437, se 0.0093, **t = +4.70** over n = 200 tasks.

**The prediction was: "the oracle, a perfect detector, will not be the best cell." It is the best
cell, at t = +5.80. The prediction is refuted as written, and the fault is in how I wrote it.**

The oracle cell is not a detection-only intervention. It replaces the verdict with the true metric
**and** the repair text with the metric's feedback function. I had documented that exact
two-at-once confound for model swaps hours earlier and then registered a prediction about a bundled
cell anyway. The decomposition Paper II already owns — `--feedback-level` under a fixed oracle
gate — separates them:

- **verdict alone** (oracle + `lo`, a perfect gate with no actionable text): **+0.0110, t = +1.37**.
  Not significant. A perfect detector, by itself, buys nothing.
- **prose alone, under a perfect gate held fixed** (`hi` − `lo`): **+0.0682, se 0.0073,
  t = +9.40**.

So of the +0.0792 the oracle appears to buy, **14% is the verdict and 86% is the repair text**.

**Paper II's thesis is confirmed on an independent worker, and more sharply than on the original.**
"What a checker cannot buy is the repair" is exactly what these numbers say: detection changes buy
nothing (8b −0.0035; qwen +0.0077; a *perfect* verdict +0.0110, none significant), and repair text
buys everything (+0.0682 at t = 9.40, against the released study's +0.0613 at t = 8.08 on
`gpt-4.1-mini`). The ordering that looked like a refutation was the bundle.

**D3's registered *Remaining* item is discharged.** "A second cross-provider family to show the
degeneracy is not one model's quirk": `qwen3.6-27b` is a third lineage after `gpt-5.6-luna` and
`llama-3.3-70b`, and moving the critic to it buys +0.0077 (t = +0.82) — nothing, the same answer
both prior lineages gave.

## Phase 3 is CANCELLED, and the reason is a result rather than a budget

The verdict/prose splice existed to attribute a **+0.048 cross-lineage lift**. On this worker the
cross-lineage critic's total effect is **+0.0077 at t = +0.82**. There is no lift to decompose, so
the splice would be partitioning zero. Cancelled, $3.17 not spent.

The oracle `lo`/`mid`/`hi` ladder also answers the verdict-versus-prose question *better* than the
splice could: it holds the verdict perfectly fixed while only the prose varies, with no second
sampled call, no fallback rule, and no mechanism artifact of the kind that moved V1 by +0.027.

## Final spend on this credential

| item | bound |
|---|---|
| Phase 0, three p₀ calibration passes | $1.7295 |
| Phase 1, `det-8b` | $0.5765 |
| Phase 1, `det-qwen` | $1.0080 |
| Phase 1, oracle × {lo, mid, hi} | $1.1529 |
| D3 (2026-08-06, measured actual) | $0.0740 |
| G1 twice, probes, aborted cells | ≈ $0.03 |
| **total, bounds** | **≈ $4.57** |

Against $400 authorised. The remaining $395 is not spent because the two phases it would have
funded were closed by results rather than by money: Phase 2 by the p₀ gate, Phase 3 by a null it
was designed to explain.
