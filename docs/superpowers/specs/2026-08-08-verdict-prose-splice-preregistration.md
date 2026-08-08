# Pre-registration — E1, the verdict/prose splice

Registered 2026-08-08, before any spend. Configuration below is frozen; deviations get recorded
in a RESULT section rather than edited into this one.

## Why this run exists

Checking Paper II's detector section on 2026-08-06 turned up a comparison that pooled two
different quantities: run 4's `+0.060`, a **seed-genome** between-run gap with no optimizer
anywhere, set beside two Δ(D−B) contrasts between **optimized** arms. Separating them and
recomputing the seed-genome axis properly produced a number the paper had never reported:

| critique | seed-genome held-out gap vs run 3 | in σ(Δ) = 0.029 |
|---|---|---|
| run 4, measurement tool | +0.060 | 2.07σ |
| run 6, cross-lineage `gpt-5.6-luna` | **+0.048** | 1.66σ |
| D3, cross-provider `llama-3.3-70b` | +0.007 | 0.23σ |
| run 7, perfect oracle | −0.028 | −0.98σ |

The `+0.048` reads as a counterexample to "detection is not the binding constraint": an accurate
critic, and the loop improves. It is not necessarily one, because **a model swap is not a
detection-only intervention** — a different model rewrites the critique's prose as well as its
verdict. The released `critic` cassettes show the prose differing sharply in form (median 53
characters for the cross-lineage critic against 210 for the cross-provider one). A proxy for
"does the critique quote a measured quantity" (contains a digit) does *not* separate them, 73%
against 82%, so we cannot settle it by inspection.

Paper II currently says, correctly, that it cannot attribute the `+0.048`. This run is the
attribution. It matters before submission and not after: if the lift is detection, the section's
central claim needs qualifying.

## The intervention

A new `--verdict-model` routes **only** the PASS/FAIL decision to a second model. Both models see
the byte-identical critique prompt; one supplies the gate, the other supplies the repair text fed
to the refine step. Off by default, so every recorded cassette replays unchanged.

**The fallback rule, and why it is not a contaminant.** When the verdict says FAIL but the prose
source said PASS, the prose source has no repair text to give. Handing the refine step the literal
token `PASS` as guidance would be nonsense, so it receives
`"not all requirements are satisfied; revise"` — the exact string the oracle path already uses
when no feedback function is supplied. This is the prose source having nothing to say, made
explicit, and a bare "revise" is already this series' low-actionability baseline (run 9: naming
the failed constraint is worth +0.003 over it).

**Registered prediction of the fallback rate**, from the measured PASS rates (intrinsic 0.26,
cross-lineage 0.13) under independence: V1 ≈ 19%, V2 ≈ 23%, V3 ≈ 10%, V4 ≈ 11%. Recorded here so
the observed rate is a check on this design rather than a fact discovered afterwards. A rate far
above these would mean the two roles disagree more than their marginals imply, which is itself
worth reporting.

## Design: a fully spliced 2×2

verdict ∈ {`gpt-4.1-mini` (intrinsic), `gpt-5.6-luna` (cross-lineage)} × prose ∈ {same two}.

| cell | verdict | prose | reproduces |
|---|---|---|---|
| **V1** | mini | mini | run 3's level, if the splice is inert on the diagonal |
| **V2** | **luna** | mini | isolates the **verdict** |
| **V3** | mini | **luna** | isolates the **prose** |
| **V4** | luna | luna | run 6's level |

All four are spliced, including the diagonal. This is deliberate: run 3 and run 6 are *unspliced*
(one call serving both roles), and a spliced cell issues two calls at the provider's default
temperature, which need not agree even when both route to the same model. Comparing spliced cells
against unspliced runs would mix the mechanism with the effect. The existing runs therefore serve
as an **external validity check** on V1 and V4, not as cells of the factorial.

## Frozen configuration

- `--task-set multi-hard`, `--arms A`, `--seeds 0,1,2`, `--n-train 8`, `--n-heldout 10`
- `--loop-max-retries 1`, `--max-metric-calls 20`, `--reflection-calls 4`
- worker `openai/gpt-4.1-mini`, reflection `openai/gpt-5.6-terra` (inert: `--arms A` optimizes
  nothing; matched to runs 3 and 6 regardless)
- seeds 0,1,2 only. Seeds 3–4 draw *different tasks*, which would put task-draw variance back into
  a comparison whose whole validity is that difficulty cancels.

## Priced 2026-08-08, before any spend

| cell | bound |
|---|---|
| V1 | $0.1190 |
| V2 | $0.3744 |
| V3 | $0.3744 |
| V4 | $0.3744 |
| **total** | **$1.2422** |

Under the $5 per-run default, so no `--max-usd` override. V2's bound was wrong at first draft:
the conservative "price every worker-side call at the priciest model it can route to" rule
included the critique model but not the verdict model, so a cheap-prose / expensive-verdict cell
priced below its true cost. An under-estimating ceiling is not a ceiling. Fixed and covered by
`test_an_expensive_verdict_model_raises_the_estimate_even_with_a_cheap_critic`.

## Hypothesis

**H1 (prose).** The cross-lineage lift is carried by the repair text: V3 − V1 > 0 and accounts
for most of V4 − V1.

**H2 (verdict).** It is carried by the gate: V2 − V1 > 0 and accounts for most of V4 − V1.

H1 and H2 are the two ways this can come out, and Paper II's thesis predicts **H1**. We register
that prediction explicitly: this is a test our own position can fail. If H2 holds, the section's
claim that detection is not the binding constraint is wrong on this axis and gets qualified.

## Primary analysis, registered

**A task-level paired test on the 30 shared held-out tasks**, not a test over 3 seeds. With
n = 3 seeds and σ(Δ) = 0.029, a 0.048 effect is 1.66σ and nothing here could reach significance;
the seed count is capped at 3 by the pairing requirement, so more seeds are not available. The
corrected harness releases the per-task held-out score vector behind every mean (harness item vii,
added for exactly this), so the cells pair task-by-task on identical tasks.

Reported per contrast: mean difference, the paired standard error over 30 tasks, and t. Main
effects are estimated twice each (verdict: V2−V1 and V4−V3; prose: V3−V1 and V4−V2) and both
estimates are reported, never only the larger.

## Verdict rule (decided in advance)

- **H1 SUPPORTED** iff the prose main effect is positive at |t| ≥ 2 on the paired task-level test
  **and** exceeds the verdict main effect.
- **H2 SUPPORTED** iff the same holds with the roles exchanged.
- **INCONCLUSIVE** if neither main effect clears |t| ≥ 2, or if both do and neither dominates.
  Inconclusive is a real and likely outcome at this size and will be reported as one.
- **The splice is INVALID as a mechanism** if V1 departs from run 3's 0.7767, or V4 from run 6's
  0.8250, by more than 2 paired standard errors. Then the two-call mechanism is doing something
  on its own and no cell contrast means anything. This check runs **first**.

## Commitments

- No cell is dropped after seeing its number. All four are bought in one command sequence.
- The fallback rate is reported per cell against the predictions above, whatever it is.
- If the validity check fails, that is the result and it gets written up as one; the contrasts are
  not reported as if the mechanism were sound.
- Cassettes and per-task vectors committed for all four cells, so every number replays at $0.

## The exact command

```bash
cd ~/whystuck
for CELL in "mini mini V1" "mini luna V2" "luna mini V3" "luna luna V4"; do
  set -- $CELL
  PROSE=openai/gpt-4.1-mini; [ "$1" = luna ] && PROSE=openai/gpt-5.6-luna
  VERDICT=openai/gpt-4.1-mini; [ "$2" = luna ] && VERDICT=openai/gpt-5.6-luna
  OPENAI_API_KEY=$(tr -d '\r\n' < scratchpad/openai.key) \
  .venv/bin/python scripts/run_validation_study.py --yes \
    --task-set multi-hard --arms A --seeds 0,1,2 \
    --n-train 8 --n-heldout 10 --max-metric-calls 20 --loop-max-retries 1 \
    --reflection-calls 4 \
    --worker-model openai/gpt-4.1-mini \
    --critique-model "$PROSE" --verdict-model "$VERDICT" \
    --reflection-model openai/gpt-5.6-terra \
    --record tests/fixtures/cassettes/study-splice-$3.json \
    --out docs/validation-study-splice-$3.json \
    --report docs/validation-study-splice-$3.md
done
```

## DEVIATION — run aborted after V1, 2026-08-08

The 2×2 above was started on the OpenAI key and **stopped by the operator after V1**, on the
instruction that this work use the borrowed Groq credential only. V2 was killed a single call in
(1 cassette entry) and V3/V4 never started. Actual spend: V1's bound of **$0.1190**, plus one
`gpt-5.6-luna` call. Nothing from the aborted attempt is in `docs/` or the cassette fixtures; the
files sit in `scratchpad/e1-aborted/` so the release counts stay truthful.

**V1 nevertheless answered the validity check, and it answered it badly.** The registered check
was: *the splice is invalid as a mechanism if V1 departs from run 3's 0.7767 by more than 2 paired
standard errors*, and it was registered to run first precisely so a mechanism artifact could not be
read as an effect.

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| run 3, unspliced | — | — | — | 0.7767 |
| V1, spliced, both roles on `gpt-4.1-mini` | 0.8150 | 0.7700 | 0.8250 | **0.8033** |

**+0.027 from the mechanism alone**, against σ(Δ) = 0.029 and against the +0.048 the design exists
to decompose. The two calls are drawn at the provider's default temperature, so the verdict sample
and the prose sample disagree at some rate even when both route to the same model; when they
disagree in the FAIL/PASS direction the refine step receives the generic `revise` directive
instead of the prose source's complaint. On a near-random intrinsic critique that substitution is
evidently not neutral.

**Consequence for the design, recorded before any re-run.** A mechanism that moves the score by
more than half the effect under study cannot serve as the instrument for measuring it. Whatever
provider the re-run uses, the splice needs either

1. **temperature 0 on both critique calls**, so the diagonal cells are deterministic and the
   mechanism is provably inert where both roles share a model; or
2. **one call, two readings** — issue the critique once and take the verdict from one model's
   reading of that fixed text, which changes what is being asked; or
3. a **paired within-cell design** that differences out the mechanism.

Option 1 is the cheap one and preserves the question. It is also a configuration change, so it
requires a fresh registration rather than an edit to this one.
