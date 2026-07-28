# GEPA validation study — results note (eleven runs + cross-domain code)

> ## ⚠️ SUPERSEDED HEADLINE — read this first (added 2026-07-28)
>
> An adversarial audit of this harness established that **the loop-level arm never received
> its treatment**, so the Δ(D−B) numbers below do not measure loop-level optimization.
> Evidence, all verified by offline cassette replay at $0:
>
> 1. **The verify gene never changed.** Replaying run 3 (the flagship n=5 multi-hard run) and
>    dumping every winning genome: `generate` changed in 5/15 winners, `ensure_correct` in
>    **0/15**. GEPA proposed critique rewrites and they were rejected — and the proposals were
>    *generation* instructions written into a critique slot.
> 2. **The reflector could not see the critique.** `adapter.py` returned the *same* records for
>    every component, with "Generated Outputs" defined as the last *worker* step (verifier
>    steps deliberately excluded). So the optimizer rewrote a detector while being shown the
>    generator's answer.
> 3. **The two-gene arm was starved per gene.** `optimize.py` passed no `module_selector`, so
>    GEPA used its default `round_robin` — one component per proposal — meaning arm D searched
>    the shared productive gene about half as hard as arm B under a matched cap. This also
>    *reverses* the over-fitting mechanism the writeup asserted.
> 4. **A percentile-index error.** `exact_resample_ci` took the upper bound at
>    `int(0.975*N) - 1`, dropping the largest resample atom, which at n=3 carries
>    1/27 = 3.70 % > 2.5 %. Corrected: the actuator run's Δ(D−B) = −0.10 has interval
>    **[−0.150, +0.000]**, which *includes* zero (sign test p = 0.50). It was published as
>    [−0.150, −0.050] "excluding zero".
> 5. **The flatness controls could not fail.** Arm A copied its own seed score with δ
>    hardcoded to 0.0; arm C's command loop scored a saturated 1.0, so GEPA proposed nothing.
> 6. **The seeds are not independent replications** (added 2026-07-28, second audit). Every
>    loader ends in "shuffle the pool, split at mid", but `load_multi_constraint` reseeds the
>    *pool* first while the vendored loaders (ifbench, code, bcb) shuffle a fixed file — there a
>    seed picks a *partition*, not a sample. Measured pairwise held-out overlap at the sizes
>    used: 0 % for multi/multi-hard, 20 % for the bcb null, 20 % and 25 % for the two actuator
>    runs, 7 % for the code-actionability pair. Correlated units make the CI too narrow and the
>    sign-test p too small, both toward overstating significance. So the bcb results, including
>    the actuator claim, may not be read as replication; the text nulls and +0.063 are clean.
>    Overlap must be measured on task *content*: the generator numbers tasks by index, so two
>    seeds reuse 15–17 of 32 ids for different tasks and an id-based check reports the one clean
>    family as ~50 % overlapping. `analyze_deltas.py` now marks the affected lines.
> 7. **The fix shipped with the same class of defect** (second audit, found before any spend).
>    `_detector_records` rebuilt the task as `BenchmarkTask(id=trace.task_id)` with no payload;
>    every text scorer returns 0.0 for an empty constraint list by design, so `verify_truth` was
>    a *constant* `(False, "no verifiable constraints on this task")` — every PASS labelled a
>    false pass, every rejection labelled correct, i.e. directional pressure toward a verifier
>    that never accepts, in the channel that *is* arm D's treatment. All three unit tests passed
>    because each `verify_truth` double ignored its `task` argument.
> 8. **A "strict" replay silently reported a different number** (second audit). GEPA catches
>    exceptions from its reflection LM, so a `CassetteMiss` there does not abort the run.
>    Replaying run 11 under the corrected harness: **70 swallowed misses, exit 0**, a written
>    report, 11 of 12 arm-seed scores matching, and seed 1 arm D returning 0.681 against a
>    recorded 0.666 — 0.681 being exactly that seed's baseline. The failure mode converts
>    "accepted a change that hurt" into "accepted nothing", with no visible signal. The cassette
>    now counts misses and raises on context exit.
> 9. **The metric has a do-nothing floor.** The empty string scores 0.548 on the vendored
>    IFBench slice and 0.420 on generated multi-hard ('x' scores 0.452 / 0.499); five of eleven
>    IFBench verifiers pass vacuously on text with no whitespace, quotes or bullets. Faithful to
>    upstream, equal across arms (so Δ is unbiased), but on IFBench the floor sits at or above
>    the reported seed baselines, so ±0.10 is measured against a much smaller effective range
>    than [0,1] suggests.
> 10. **`text_stats` is verdict-equivalent for 57 % of constraint instances.** The paper's rule
>    ("a field is the oracle iff its verifier has no per-task parameter") is too weak: given the
>    task statement in the prompt, `words`/`longest_word_letters`/`bullet_lines`/`last_words`
>    settle `exact_word_count`/`word_max_length`/`num_bullets`/`end_with` by comparison — 4 of 8
>    families, 165 of 288 instances, at least one on all 64 tasks (only 2 tasks fully). Held-out
>    scores are uncontaminated (they never see the tool), but the F₁ 0.11→0.83 lift is largely
>    the tool *replacing* the detector, not a prompt that learned to detect.
>
> **FIRST RESULT UNDER THE CORRECTED HARNESS — C1 verifier ablation (2026-07-28, $0.043).**
> Two arm-A conditions on multi-hard seed 0, 32 held-out tasks, differing ONLY in
> `ensure_correct` (seed rubric vs `docs/genes/verify_measure_v2.txt`), sharing one
> content-addressed cassette so each task's first draft is byte-identical across them.
> Analysis pre-registered in `scripts/analyze_c1.py`, committed *before* the run was paid for.
> Paired by TASK: n=32, 17 ties, hand better on 6 / seed better on 9, mean Δ = −0.033,
> 95 % CI [−0.095, +0.028], sign p = 0.61 → **INCONCLUSIVE, not "no effect"** (the 90 % CI
> passes the ±0.05 SESOI, so equivalence fails too).
>
> The detector numbers are the finding. On the same 32 drafts (base rate 0.16):
>
> | critique gene | n | tp | fp | fn | PASS rate | refines |
> |---|---|---|---|---|---|---|
> | run 3 intrinsic (F₁ = 0.11) | 50 | 1 | 12 | 4 | 0.26 | — |
> | C1 seed gene | 32 | 0 | 3 | 5 | 0.09 | 29/32 |
> | C1 hand gene | 32 | 0 | 0 | 5 | **0.00** | **32/32** |
>
> Neither C1 gene produced a single true positive, so F₁ is undefined for both; the
> hand-written gene changed *when* the detector fires, not *whether it can tell*. Catching the
> 3 false passes moved those tasks 0.800 → 0.867 (one repaired to 1.00), worth +0.006 on the
> mean; the 5 already-compliant drafts give −0.100, and since BOTH genes demanded fixes on all
> 5 that is one damaged draft, not a false-alarm difference. ⇒ prose-only rewriting moves the
> threshold, not the discrimination, a mechanism for the loop-level null that does not involve
> the optimizer. Evidence, not proof: one hand-written point in prose space. Caveat: only 5
> positive cases per split, so tp = 0 cannot separate "no" from "weak" discrimination.
>
> **`verify_reproducibility.py` now reports ALL 2 RUNS REPRODUCE FROM CASSETTE**, not
> NOTHING PROVEN. Measured cost realization: $0.043 actual vs $0.190 guard bound = **23 %**.

> **C2b — THE CLEAN FOUR-ARM RE-RUN (2026-07-28). This is the run the paper reports.**
> Same config as C2 below, re-run after the blank-gene guard and with the reflection token
> cap raised 1500 → 3000. **No empty completions anywhere** (shortest reflection output 968
> chars), 542 calls, and all 24 arm-scores + gene-change sets **reproduce exactly** under
> strict replay.
>
> **Treatment applied on 3 of 3 seeds** — arm D wrote detector rubrics of 3235 / 3022 / 2920
> chars against a 127-char seed. **ties = 0/3**: for the first time in the study every seed
> produced a genuinely different B-vs-D contrast.
>
> | arm | evolves | seed | optimized | Δ |
> |---|---|---|---|---|
> | A | nothing | 0.788 | 0.788 | +0.000 |
> | C | genome ignored | 0.788 | 0.788 | +0.000 |
> | B | `generate` | 0.788 | 0.788 | **+0.000** |
> | D | `generate` + `ensure_correct` | 0.788 | **0.747** | **−0.042** |
>
> Δ(D−B) = **−0.042**, per-seed [−0.095, −0.080, +0.050], 95 % CI [−0.095, +0.050],
> sign p = 1.00. Point estimate at the edge of the ±0.05 SESOI in the WRONG direction.
> **Prompt-only optimization is exactly flat (+0.000) while loop-level costs −0.042** — so
> C2's "optimization hurts BOTH arms" does NOT survive; the cost is specific to the arm with
> the extra gene. Task-level: 19/30 exact ties, 11 flips (7 favour B, 4 favour D).
>
> **Positive control (free, real GEPA):** on a landscape where evolving the critique is the
> only path to a passing answer, the same harness returns A/B/C 0.000, D 1.000,
> Δ(D−B) = **+1.000**. The instrument responds; the nulls are about the tasks.
>
> **C2 — THE CONTAMINATED FIRST ATTEMPT, retained only as the evidence for Evidence 8
> (2026-07-28).** multi-hard, arms ABCD, seeds 0/1/2, 8 train / 10 held-out, mmc 28,
> 6 reflection calls, 1 refine round, `--module-selector all`, reflection `gpt-5.6-luna`.
> Both parts of the prediction pre-registered in the paper held.
>
> **(i) The treatment was applied.** Arm D's `genes_changed` includes `ensure_correct` on
> **2 of 3 seeds**, against **0 of 15** across the released runs. This is the check no
> released run could perform.
>
> | arm | evolves | seed | optimized | Δ |
> |---|---|---|---|---|
> | A | nothing | 0.790 | 0.790 | +0.000 |
> | C | genome ignored | 0.790 | 0.790 | +0.000 |
> | B | `generate` | 0.790 | 0.750 | **−0.040** |
> | D | `generate` + `ensure_correct` | 0.790 | **0.735** | **−0.055** |
>
> **(ii) The null holds.** Per-seed Δ(D−B) = [+0.000, +0.020, −0.065], mean **−0.015**,
> 95 % CI [−0.065, +0.020], sign p = 1.00. Point estimate inside the ±0.05 SESOI as
> predicted; the interval is not, so this is not equivalence either, and at n = 3 the sign
> test floors at p = 0.25. What separates this null from the eleven it replaces is not its
> width but that **the arms provably differ** — different genomes, different scores.
>
> **The unpredicted finding: optimization made both arms worse.** Both controls sit exactly
> on the seed baseline while B and D fall below it, so the loss is attributable to
> optimization, not scoring noise; D loses more because it has two genes of surface to
> overfit on 8 training tasks. Seed 0 is the one tie: GEPA accepted nothing in *any* arm and
> returned the seed genome (the documented GEPA plateau).
>
> **Arm C is now evidence rather than an identity.** Its +0.000 is unchanged from the
> retracted runs, but there it was saturated at 1.0 and *could not* propose; here it starts
> at 0.36 with headroom, GEPA proposes against it, and it still lands on the baseline.
>
> **⚠️ CONTAMINATED ON SEED 2 — see Evidence 8.** Of 31 reflection completions **2 were
> empty**, both proposals for `ensure_correct` in arm D. On seed 0 the empty proposal was
> rejected on score; on seed 2 it was **accepted** (a loop with no critique beats one with a
> near-random critique on the training subsample, 2.25→2.5) and became the winner, so arm D
> on seed 2 ran with **no critique at all** — and that is the seed carrying the −0.160 drop
> and the −0.065 that dominates Δ. So the treatment was genuinely applied on **seed 1 only**
> (a real 3101-char rubric); seed 2's "change" was gene deletion. `genome_from_candidate` now
> refuses blank gene text, which makes C2's own cassette unreplayable (replay re-serves the
> empty completion and the guard raises), so C2 is marked SUPERSEDED until re-run.
> Likely cause: `REFLECTION_MAX_OUTPUT_TOKENS = 1500` consumed by a reasoning model's
> reasoning tokens before any content is emitted.
>
> Integrity: 461 distinct calls, 0 empty *worker* outputs, and all 24 arm-scores plus
> gene-change sets reproduced exactly under strict replay *at the time of the run*. Note the
> first integrity pass wrongly reported 0 empty outputs overall: it serialised each cassette
> entry to JSON before measuring length, so it sized a dict and could never see an empty
> completion. Cost ≈ $0.31 estimated against the $1.7358 priced bound.
>
> **Released artifacts do NOT carry the provenance fields.** `genes_changed`, `train_score` and
> `final_genes` postdate every recorded run and are empty in all 17 result JSONs, so the
> reporting minimum this study recommends cannot be checked against its own release — only
> against runs made after it. Runs now also release the per-task held-out score vectors
> (`seed_heldout_by_task` / `opt_heldout_by_task`), which is what a paired task-level test needs
> given finding 6.
>
> **Retracted:** "no measurable advantage for loop-level optimization"; the null being "tight
> rather than underpowered" on the strength of [0,0] intervals (those are *ties* — B and D
> producing byte-identical genomes, i.e. the same calls served from the same cassette entries);
> the train-split over-fitting mechanism; "Δ = −0.10 excludes zero"; "rules out under-powering".
>
> **Actuator run, partially recovered (2026-07-28, $0, from the released JSON):** per-seed
> B/D held-out are seed0 0.55→0.60 / 0.55→**0.45**; seed1 0.45→**0.60** / 0.45→0.45 unmoved;
> seed2 0.65→0.65 / 0.65→0.65. A moved score proves the genome changed, so on seed 0 arm D
> DID ship a change — "treatment never applied" does not extend uniformly to this experiment
> (it evolves generate+actuator with the detector fixed). Which gene changed is unrecoverable:
> bcb replay depends on container test outcomes, and a visible test times out under load here,
> changing the critique prompt and hence the cassette key. Seed 1 is the tell: B accepted a
> transferring change while D never moved — the round-robin starvation signature. So half the
> headline −0.10 is about search budget per gene, not about optimizing the fixer. NOTE the
> asymmetry: an unmoved score does NOT prove an unchanged genome, so seeds 1–2 are formally
> inconclusive on that point.
>
> **Survives:** everything not depending on arm D having been optimized — detector F₁ = 0.11
> (0.83 with the measurement tool), the per-constraint residual table, the fix-budget effect,
> and the actionability effects (+0.063 text, +0.20 code). Two caveats: runs 7–10 gate on the
> true metric at *held-out* scoring time too, so those are **ceiling results** conditional on a
> perfect checker at inference; and at n=3 a sign test cannot go below p = 0.25, so +0.063 rests
> on magnitude, not on replication across seeds.
>
> **Harness now fixed** (component-specific reflective dataset; explicit `module_selector`
> defaulting to `all`; non-saturating arm C; arm A re-measured; per-gene accepted-change counts,
> train aggregates and final gene text released per arm; detector truth now receives the real
> task and raises on an unmatchable trace; replay fails on a swallowed miss; per-task held-out
> score vectors released; `--seed-gene GENE=FILE` injects a hand-written gene for ablations). The corrected wiring changes every
> reflection prompt, hence every cassette key, so **the eleven committed cassettes cannot replay
> under it** — `verify_reproducibility.py` now reports them as skipped/superseded rather than
> verified, and `EXPECTED_RUNS = 0` until the study is re-run.
>
> **Pre-audit replication worth keeping:** worker swap to `gpt-4.1-nano` (multi-hard, 3 seeds,
> 16/16, mmc 64) → Δ(D−B) = −0.014, per-seed (+0.000, −0.041, +0.000), corrected interval
> [−0.041, +0.000], 2/3 seeds tied; B = 0.70 ± 0.03, D = 0.69 ± 0.03, seed 0.67, actual spend
> ≈ $0.42. Shows the pattern is not one worker model's artifact, but inherits the same harness.
> An mmc = 160 budget replication was designed, priced (~$0.90 projected), started, and then
> **stopped mid-run** once the audit landed: spending on superseded wiring stopped being
> worthwhile. It consumed ≈ $0.33 (10 reflection + 271 worker calls) before being killed and
> produced no results JSON; its partial cassette was deleted rather than committed, since a
> cassette with no corresponding result is a misleading artifact.

*Question:* does **loop-level** GEPA (evolve the self-refine loop's `generate` **and**
`ensure_correct` verify genes) beat **prompt-only** GEPA (evolve `generate` alone), on
instruction-following tasks, under a matched budget?

*Answer:* **unknown** — see the superseded-headline box above. The runs below are retained as
the record of what was measured, not as an answer to that question.

## The three runs

All runs share one design: 4 arms on an identical seed genome — A = no optimization,
B = prompt-only (evolve `generate`), C = genome-blind command control (flatness check),
D = loop-level (evolve `generate` + `ensure_correct`). The worker is a
generate→critique→refine loop; the critique is *intrinsic* (the constraint verifier is
the scoring metric and is never consulted mid-loop — gating on it would be oracle
contamination). Only what GEPA is allowed to evolve differs between B and D, so the
held-out gap Δ = mean(D) − mean(B) isolates the loop-level lever, not prompt length.
Worker `gpt-4.1-mini`, reflection `gpt-5.6-terra`.

| Run | Task regime | seed held-out | B (prompt-only) | D (loop-level) | Δ = D − B | Verdict | Why |
|-----|-------------|---------------|-----------------|----------------|-----------|---------|-----|
| 1 | single-constraint (IFBench slice) | 0.47 | 0.59 | 0.50 | **−0.09** | NOT SUPPORTED | **uninformative** — one constraint ⇒ nothing to decompose ⇒ verify gene inert |
| 2 | easy multi-constraint (k = 3–4) | **0.97** | 0.97 | 0.97 | **0.00** | NOT SUPPORTED | **uninformative** — seed saturated, no headroom (ceiling flag) |
| 3 | **hard multi-constraint** (k = 4–5, exact-count + brutal) | **0.765** | 0.770 | 0.770 | **0.00** | NOT SUPPORTED | **the fair test** — real headroom, no ceiling/floor flag |

Run 3 is the one that counts. Seed held-out ≈ 0.765 sits inside the pre-registered band
0.2 ≤ seed ≤ 0.85, so neither the ceiling nor floor flag fires — the loop had genuine
room to help and the verify gene had multiple independent constraints to decompose.

The run-3 tie is not a dead line: per-seed, D and B trade wins (seed 1: D +0.05, B −0.005;
seed 4: B +0.055, D flat) and net to Δ = 0.000. Control C is flat on every seed, which
confirms the harness is sound — a real signal *could* have shown up and didn't.

## How this fits the literature

- The result is exactly what the self-correction literature predicts. Huang et al.,
  [*LLMs Cannot Self-Correct Reasoning Yet*](https://arxiv.org/abs/2310.01798) (ICLR 2024),
  find that **intrinsic** self-correction — refining with no external feedback signal —
  fails to reliably improve outputs and can degrade them (CommonSenseQA 75.8 → 41.8 after
  one intrinsic round). They show prior "self-correction works" gains relied on **oracle**
  stopping signals: "the improvements in these studies result from using oracles to guide
  the self-correction process, and the improvements vanish when oracle labels are not
  available." Self-correction genuinely helps only with a real external verification
  mechanism (a tool, a code executor, unit tests). Arm D's verify gene is precisely an
  intrinsic critique (no oracle). A null is the expected outcome, not a surprise.
- [Self-Refine](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)
  (Madaan et al., NeurIPS 2023) reports ~20 % gains from self-feedback, but on open-ended
  generation quality, not on tasks with an *external verifiable* metric where the model
  cannot see its own score. Our tasks are the hard case for self-refinement.
- [GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026 oral) claims reflective prompt
  evolution beats RL — a claim about **prompt** evolution vs policy gradients, *not* about
  loop-level vs prompt-level. Our null does not contradict GEPA; it answers a question GEPA
  does not.
- DSPy pipeline-optimization work reports that optimizing multiple interacting modules pays
  off **as pipeline complexity grows**. Our loop is minimal: 2 genes, 1 refine round. That
  is the low-complexity end where the literature would least expect a loop-level edge.

## Bottom line

Within scope — one worker model, one small budget (max_metric_calls = 28, 1 refine round),
one task family — **loop-level GEPA gives no edge over prompt-only GEPA on verifiable
instruction-following.** This is a clean, publishable negative result, and it agrees with
the strongest prior on intrinsic self-correction. The burden now shifts: the interesting
question is no longer "does the loop help?" (default: no) but "under what conditions
*would* it?"

## Next step (as of run 10 + the $0 analyses)

The mechanism story is **complete and, within its scope, publication-ready**: null (tight
CIs) → near-random detector → non-monotonic ceiling → fix step → actionability lever
(CI excludes 0) → residual = a single mechanical skill (exact counting). The cheap
experiments are exhausted; the two remaining paid ideas are low-value:

- **D7 (more seeds): low value.** The exact-resample CIs are already tight (the paired Δ has
  near-zero variance), so more seeds would only marginally narrow them. The high-value part
  of D7 was free and is done.
- **D1′ (deterministic word-count actuator): low value.** The decomposition already made its
  point (residual = exact_word_count); a counter fixing counting is unsurprising. Its only
  residual value is confirming the other constraints don't break when the count is enforced.

- **D5 (cross-domain): DONE — null corroborates.** Replicated the 4-arm design on code (unit
  tests), where the verify step catches *real* semantic errors. MBPP+ saturated its train split
  (all arms == seed, degenerate Δ=0 — kept as a documented failed attempt, paper appendix), so
  we moved to **BigCodeBench-Hard** (Docker sandbox; visible unittest methods in-loop, hidden
  methods scored). There the loop had genuine headroom (seed 0.40 / 0.50) and GEPA *ran*: on
  seed 1 it accepted training-improving candidates for **both** B (generate) and D
  (generate+ensure_correct), each lifting the 8-task train aggregate 0.375 → 0.625 — yet every
  arm scored exactly its seed baseline on held-out, so **Δ(D−B) = 0.00** (exact CI [0,0]). Even
  when loop-level optimization pulls its lever, the train gain does not transfer and it beats
  prompt-only by nothing. The instruction-following null is **not** an IFEval artifact.
  Reports: `validation-study-bcb.md` (live), `validation-study-code.md` (MBPP+). Math (numeric
  checker) remains a natural next replication.
  - **Code actionability ablation (cross-domain D2′): the one thing that helped transfers.**
    Arm A (no GEPA), bcb, 6 seeds, in-loop visible-test feedback varied lo (verdict) vs hi
    (verdict + visible-test source). **Δ(hi−lo) = +0.20 held-out, 95% CI [+0.10, +0.30]
    (excludes 0)** — a 3× larger lift than IFBench's +0.063, and again a *baseline* gain
    (feedback shape), not Δ(D−B) (=0), so GEPA didn't need to discover it. Reinforces "the
    lever is the fixer + feedback actionability, not loop-level optimization." Reports:
    `validation-study-bcb-actlo.md` / `-acthi.md`; number regenerated by `analyze_deltas.py`.
  - **Actuator experiment (the terminal test): aiming GEPA at the fixer does NOT rescue it.**
    Added a fourth evolvable `actuator` gene — an LLM step that turns the visible-test signal +
    source into the refine step's repair guidance; arm D co-evolves generate+actuator with the
    detector **held fixed**, so Δ(D−B) isolates optimizing the fixer's feedback (the component we
    *know* helps, +0.20). **Small run (2 seeds, 8/10, mmc30): Δ(D−B)=0.00** (GEPA proposed good
    actuator rewrites but accepted none that transferred). **Powered run (3 seeds, 20/20, mmc50):
    Δ(D−B) = −0.10, exact CI [−0.150, −0.050] (excludes zero)** — prompt-only B now *transfers*
    (+0.067 held-out mean) but loop-level D does *worse* (seed 0: 0.55→0.45), the 2-gene search
    space over-fitting the small train split. Across all 3 code configs the ordering is invariant:
    **D ≤ B on held-out, never D > B.** This **closes the budget/power threat** (raising train
    size / held-out resolution / seeds / mmc surfaced no hidden positive — it went negative) and is
    the paper's strongest form: the actionability gain is feedback *design*, not GEPA-discoverable.
    Reports: `validation-study-bcb-actuator.md` / `-actuator-big.md`; `analyze_deltas.py`.
    (Executor note: a memory-hog model draft that OOMs its container is scored a failure, not an
    infra abort — surfaced + fixed during the powered run; genuine host faults still abort loudly.)

**Recommendation: write this up as the arXiv paper now (`docs/paper/main.tex`) and treat D5
as the scoped v2 / follow-up.** Don't spend on D7 or D1′. Optional cheap armor: re-run *only*
run 3 at n=10 (~$1) to preempt the reflexive "n is small" note, though the CI is already tight.

**Update — both levers are being tested (runs 4 and 5).** Rather than stop, both
literature-implied levers were pre-registered and run, one change each:

- **Run 4 — feedback richness** (`.../specs/2026-07-23-feedback-richness-preregistration.md`):
  the critique step is given a deterministic, non-oracle `text_stats` readout (the
  tool/executor signal Huang et al. found necessary). Report: `validation-study-feedback.md`.
  **Result: NOT SUPPORTED** (B = D = 0.80, Δ = 0.00). Two findings, both consistent with
  Huang: (i) the tool *helped the baseline* — seed rose 0.765 → **0.837** vs run 3 — so
  external feedback works; (ii) reflective optimization then *degraded* both arms (−0.04
  mean, −0.10 on one seed), the "self-correction can make it worse" effect. D tracked B
  exactly in all three seeds.
- **Run 5 — complexity** (`.../specs/2026-07-23-complexity-preregistration.md`): a third
  interacting `plan` module (plan→generate→critique→refine), so whole-genome optimization
  has more to co-tune (the DSPy lever). Report: `validation-study-plan.md`.
  **Result: NOT SUPPORTED** (B = D = 0.76, Δ = 0.00). Optimization moved nothing on any arm;
  the third module gave loop-level D no traction. No ceiling/floor flag on either run.

Both are lower-powered exploratory runs (n = 3 seeds); a clear D > B in either would have
warranted a 5-seed confirmation — neither produced one. Full reference list in
`docs/references.bib`.

### Runs 6 & 7 (cross-lineage critic, oracle ceiling)

- **Run 6 — cross-lineage critic** (`validation-study-crossfamily.md`, `.../specs/2026-07-23-cross-family-critic-preregistration.md`): critique on `gpt-5.6-luna`, generation on `gpt-4.1-mini`. **INCONCLUSIVE**: D 0.825 vs B 0.817, Δ = +0.01 — the first run where D didn't lose, but within n=3 noise. The different-model critic also lifted the seed (0.765 → 0.825). *Cross-lineage, not cross-provider* (shared vendor/lineage) — a true cross-family test (Gemini/Claude) is the follow-up. ~$0.36.
- **Run 7 — oracle ceiling** (`validation-study-oracle.md`, `.../specs/2026-07-23-oracle-ceiling-preregistration.md`): the critique gate = the true metric (perfect detector). **Surprise: it did NOT lift the ceiling** — oracle seed 0.748, level with intrinsic (0.765) and *below* the tool (0.837). So detector quality vs performance is **non-monotonic** (F₁ 0.11→0.83→1.0 maps to seed 0.765→0.837→**0.748**). The bottleneck is not detection but the **fix step**: even told exactly what's wrong, `gpt-4.1-mini` can't repair hard constraints in one refine round. ~$0.22.
- **Run 8 — fix budget under oracle** (`validation-study-fixbudget.md`, `.../specs/2026-07-23-fix-budget-preregistration.md`): oracle + refine budget 1→3. **More attempts help but saturate**: oracle seed 0.748 (R1) → **0.813** (R3), still below the 0.85 bar and below the tool's 0.837 — a good tool at R1 ≈ oracle at R3. The fix step is a real but bounded lever; the residual is generator capability. ~$0.33.

### Runs 9 & 10 (feedback-actionability ablation, D2′)

Holding detection fixed at the perfect oracle (R=3) and varying ONLY the actionability of the repair feedback fed to the refine step, across three pre-registered levels (mid = run 8):

| level | repair feedback on failure | seed held-out (arm A) |
|-------|----------------------------|-----------------------|
| **lo** | bare verdict: "not all requirements are satisfied; revise" | 0.810 ± 0.049 |
| **mid** | which constraints failed + their targets (= run 8) | 0.813 ± 0.044 |
| **hi** | mid **+ concrete measured edits** ("draft has 2 words, needs exactly 20: add 18") | **0.873 ± 0.027** |

- **Run 9 — lo** (`validation-study-fbactlo.md`) and **Run 10 — hi** (`validation-study-fbacthi.md`), pre-reg `.../specs/2026-07-23-feedback-actionability-preregistration.md`. ~$0.3–0.5 each (312 / 329 unique calls).
- **Primary result: Δ(hi − lo) = +0.063 ≥ +0.05 → the fix step is partly FEEDBACK-LIMITED.** Concrete measured edits lift the ceiling +6.3 pts over a bare verdict.
- **Two sharp sub-findings:** (1) **lo ≈ mid** (0.810 vs 0.813) — naming *which* constraint failed adds ~nothing; only the *concrete measurement* moves it. The actionable content is the measurement, not the diagnosis. (2) The lever is feedback **shape** (a hand-built measurement fn), **not** loop-level GEPA: Δ(D−B) = 0.00 at every level, and optimization again slightly *degrades* both arms (hi seed 0.873 → opt 0.86).
- **Resolves the run-4 tool>oracle inversion:** hi (0.873) is the first oracle config to *exceed* the tool baseline (0.837). The oracle underperformed at R=1 because its verdicts were terse, not because a perfect detector hurts.
- **Residual remains generator-limited:** even maximal, exact repair guidance saturates at 0.873 (< 0.9, no ceiling flag). So the fix step is *both* feedback-limited (up to ~+0.06) and, beyond that, generator-limited.

## Final verdict across ten runs

The thesis "loop-level GEPA beats prompt-only GEPA" is **NOT SUPPORTED** on IFEval-style
instruction-following at this budget/model — in every regime we could test: single- and
multi-constraint, with genuine headroom (run 3), a real external feedback tool (run 4), an
extra interacting module (run 5), a different-model critic (run 6), an oracle detector
(run 7/8), and an oracle with maximally-actionable feedback (run 10). Δ(D − B) ≤ 0.01
(within noise) in every fair test. Things that moved the *baseline* (the feedback tool, the
different-model critic, actionable feedback) helped B and D equally, so they are not
evidence for the loop-level lever.

**The sharpened conclusion (runs 7–10):** the bottleneck is **not detection** — a perfect
in-loop detector (oracle) did not lift the ceiling (0.748, below the tool's 0.837), so
detector-quality vs performance is *non-monotonic*. The binding constraint is the **fix
step**, which runs 9–10 decompose: it is **partly feedback-limited** (concrete measured
edits beat a bare verdict by +0.063; merely naming the failed constraint adds nothing) and,
beyond that, **generator-limited** (even exact edits saturate at 0.873). The actionable
reframe for making a self-refine loop worth optimizing componentwise is **improve the fixer
and the actionability of feedback (not just its correctness), not the detector** — but note
that the actionability win was feedback *shape*, which GEPA did not need to discover.

### Mechanism (measured, zero cost)

Why is the loop-level lever inert? Because the thing it evolves — the intrinsic
self-critique — is a near-random failure detector. Measured offline from the cassettes
(`scripts/analyze_critique_detector.py`), the seed critique's PASS verdict tracks whether
the draft actually satisfies its constraints with F₁ ≈ 0.11 on run 3 (precision 0.08) and
≈ 0 on run 5. The *same* critique fed the measurement tool (run 4) jumps to precision 1.00,
F₁ 0.83 — which is exactly why the tool lifts the baseline, but that reliable detector sits
in the B and D loops alike, so it opens no D-vs-B gap. This is the empirical version of
Huang et al.'s self-verification-unreliability result, inside our harness.

**Where the residual lives (per-constraint decomposition, $0).** `scripts/analyze_residual_by_constraint.py` replays the cassettes, takes the loop's *final* draft per held-out task, and reports per-constraint satisfaction. Under the oracle detector + maximally-actionable feedback (run 10), the residual is almost entirely **one constraint — `exact_word_count`** (satisfied only ≈0.50, up from 0.16 intrinsic / 0.33 with which-constraint feedback), while every "choice" constraint (no_comma, all_lowercase, num_bullets, include_keyword, end_with, forbidden_word) is at 1.0 and word_max_length at 0.90. So the generator-limited residual is a single mechanical skill (LLMs can't hit exact token counts), not a diffuse ceiling — recoverable by a deterministic actuator, not by prompt/critique optimization.

**The null is tight, not underpowered ($0 stats).** Exact-resample 95% intervals on the paired per-seed Δ(D−B): run 3 (n=5) [−0.033, +0.033]; runs 4 & 10 exactly [0, 0] (B and D optimize to byte-identical genomes every seed). The one positive effect, Δ(hi−lo)=+0.063, has interval [+0.025, +0.085] excluding zero at n=3. So more seeds (D7) would only shrink already-tight intervals — low value.

### Reproducibility (verified)

Every LLM call in every run is recorded to a content-addressed cassette. `scripts/verify_reproducibility.py`
replays **11 runs bit-for-bit** — the 10 IFBench runs plus the MBPP+ cross-domain run — **offline, $0**
via strict `--replay` (no API key; a cassette miss raises), confirming each per-arm seed/optimized
held-out score equals its released JSON exactly (GEPA is deterministic under replay). Run it with the eval venv:

```
scratchpad/evalvenv/bin/python scripts/verify_reproducibility.py
# → [OK] run1 … [OK] D5-code MBPP+ ; ALL 11 RUNS REPRODUCE FROM CASSETTE ($0, offline).
```

**BigCodeBench-Hard is scoped differently, by construction.** It is *not* bit-for-bit replayable:
model solutions may use unseeded randomness, so re-executing them in the Docker sandbox yields
run-to-run-varying test output that shifts the content-addressed prompt key — an intrinsic property
of executable-code benchmarks, not of our harness. What *is* guaranteed: per-execution grading is
deterministic, the string-level non-determinism we can normalize (object ids, addresses, temp paths
in failure text) is scrubbed so identical executions render identical feedback, and the headline
Δ(D−B)=0.00 **reproduced identically across two independent paid runs**. The cassette and scores are
released; strict call-replay is out of scope for it deliberately.

### Toward publication

A pre-print draft consolidating all ten runs, the method, the oracle taxonomy, this
mechanism, threats to validity, and a boundary-mapping agenda (remaining fix-side
interventions, budget/worker sweeps, cross-domain replication) lives in
`docs/paper/main.tex` (references in `docs/references.bib`).
