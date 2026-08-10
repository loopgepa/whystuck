# Pre-registration — E1-G, the verdict/prose splice on an all-Groq stack

Registered 2026-08-08, before any spend on this configuration. Supersedes the configuration in
`2026-08-08-verdict-prose-splice-preregistration.md`, which was aborted after one cell; that file
keeps its own DEVIATION section and is not edited.

## Two reasons this is a new registration rather than an edit

**1. The credential changed, and that changes the worker.** The operator restricted this work to a
borrowed Groq key. Every run in the released series uses `worker = openai/gpt-4.1-mini`, so an
all-Groq run cannot extend that curve: **this design cannot decompose the `+0.048`**, which is a
`gpt-4.1-mini`-worker result. It answers the same mechanistic question — verdict or prose? — on a
different worker and a different critic pair. That is an independent replication rather than an
explanation of one number, and it is registered as such. Paper II keeps its sentence declining to
attribute the `+0.048`.

**2. The splice mechanism was measured and is not inert.** The aborted attempt's diagonal cell,
both roles on `gpt-4.1-mini` at the provider's default temperature, scored **0.8033** against
run 3's unspliced **0.7767**: the two-call mechanism moved the loop **+0.027** on its own, more
than half the effect under study. Two samples of the same prompt disagree at some rate, and when
they disagree in the FAIL/PASS direction the refine step gets the generic `revise` directive
instead of the prose source's complaint.

**Fix, registered:** `--temperature 0` on every call. The diagonal cells then draw the same text
twice, the fallback cannot fire there, and the mechanism is inert by construction rather than by
hope. G1 and G4 re-check it internally: on the diagonal the observed fallback rate must be **0**.

## Design

worker `groq/llama-3.1-8b-instant`. verdict × prose over
{`groq/llama-3.1-8b-instant` (intrinsic, 8b), `groq/openai/gpt-oss-120b` (cross-lineage, oss)}.

| cell | verdict | prose | role |
|---|---|---|---|
| **G1** | 8b | 8b | intrinsic baseline; diagonal, fallback must be 0 |
| **G2** | **oss** | 8b | isolates the **verdict** |
| **G3** | 8b | **oss** | isolates the **prose** |
| **G4** | oss | oss | both; diagonal, fallback must be 0 |

`gpt-oss-120b` is a genuine cross-lineage critic **here** and a trap elsewhere: against an
`openai/*` worker it rebuilds the shared-lineage confound D3 exists to remove, which is why
`cost.py` labels it. Against a Llama worker that objection does not apply.

## Frozen configuration

- `--task-set multi-hard`, `--arms A`, `--seeds 0,1,2`, `--n-train 8`, `--n-heldout 10`
- `--loop-max-retries 1`, `--max-metric-calls 20`, `--reflection-calls 4`
- `--temperature 0` (the fix above; also makes the whole run deterministic)
- reflection `groq/llama-3.1-8b-instant`, inert under `--arms A` and priced only so the gate passes
- seeds 0,1,2 only: seeds 3–4 draw different tasks and would put task-draw variance back into a
  comparison whose validity is that difficulty cancels

## Priced 2026-08-08, before spend

| cell | bound |
|---|---|
| G1 | $0.0103 |
| G2 | $0.0446 |
| G3 | $0.0446 |
| G4 | $0.0446 |
| **total** | **$0.1441** |

Prior spend on this credential is $0.074, so the running total after this is ≈ $0.22.

## The p₀ gate, checked before the other three cells

Paper I's headline is that the loop-level effect is conditional on p₀, the share of first drafts
that already satisfy every requirement, and that at a saturated or floored p₀ nothing is
measurable. An 8B worker on multi-hard may sit near the floor. **G1 is bought first and read
before G2–G4 are bought.** If G1's seed-genome held-out mean is below 0.15 or above 0.95, the
worker leaves no room for a critic to matter, the remaining cells are not bought, and that is the
recorded result. This is a gate, not an option to re-choose the design after seeing a number.

## Hypothesis

**H1 (prose).** G3 − G1 > 0 and accounts for most of G4 − G1. This is Paper II's prediction.

**H2 (verdict).** G2 − G1 > 0 and accounts for most of G4 − G1. This would be evidence against
Paper II's thesis on a new worker, and would be reported as such.

## Primary analysis, registered

Task-level paired test on the 30 shared held-out tasks, using the per-task score vectors the
corrected harness releases (harness item vii). Three seeds cannot support a seed-level test:
σ(Δ) = 0.029 makes even a 0.048 effect 1.66σ, and the seed count is capped at 3 by the pairing
requirement.

Reported per contrast: mean difference, paired standard error over 30 tasks, t. Each main effect
is estimated twice (verdict: G2−G1 and G4−G3; prose: G3−G1 and G4−G2) and **both** estimates are
reported, never only the larger.

## Verdict rule (decided in advance)

- **INVALID** if either diagonal cell records a non-zero fallback rate at temperature 0. Then the
  calls are not deterministic and the mechanism is back; no contrast is reported. Checked first.
- **NOT MEASURABLE** if the p₀ gate above trips.
- **H1 SUPPORTED** iff the prose main effect is positive at |t| ≥ 2 and exceeds the verdict effect.
- **H2 SUPPORTED** iff the same with the roles exchanged.
- **INCONCLUSIVE** otherwise, including when both clear |t| ≥ 2 and neither dominates. At this size
  inconclusive is a likely outcome and will be reported as one.

## Commitments

- G1 is read only against the p₀ gate and the fallback check before G2–G4 are bought. Its
  contrast values are not used to reselect anything.
- The observed fallback rate is reported per cell whatever it is.
- No cell is dropped after seeing its number.
- Cassettes and per-task vectors committed for every cell bought, so each replays at $0.
- This run does **not** claim to explain the `+0.048`, and no sentence in any paper will say it
  does.

## The exact command

```bash
cd ~/whystuck
W=groq/llama-3.1-8b-instant
O=groq/openai/gpt-oss-120b
for CELL in "$W $W G1" "$W $O G2" "$O $W G3" "$O $O G4"; do
  set -- $CELL
  GROQ_API_KEY=$(tr -d '\r\n' < scratchpad/groq.key) \
  .venv/bin/python scripts/run_validation_study.py --yes \
    --task-set multi-hard --arms A --seeds 0,1,2 \
    --n-train 8 --n-heldout 10 --max-metric-calls 20 --loop-max-retries 1 \
    --reflection-calls 4 --temperature 0 \
    --worker-model $W --reflection-model $W \
    --critique-model "$1" --verdict-model "$2" \
    --record tests/fixtures/cassettes/study-splice-$3.json \
    --out docs/validation-study-splice-$3.json \
    --report docs/validation-study-splice-$3.md
done
```

## RESULT — G1 bought and both gates passed; G2–G4 CANCELLED (2026-08-08)

Recorded late, on 2026-08-09, after an audit found this file had no outcome section while its G1
artifact sat in `docs/`. That is the omission this registration's own preamble warns about — a
registered design whose result is written down somewhere else is a design whose result can be
chosen after the fact. The cancellation was recorded in
`2026-08-08-groq-second-worker-replication-preregistration.md`; it belongs here too.

**G1 bought. G2, G3 and G4 never bought. Spend: G1's bound of $0.0103.**

### The two registered gates, read in the registered order

**Fallback check (INVALID rule) — PASSES, and passes for the wrong reason.** The observed
fallback rate on the diagonal is **0 of 30**, as required. But the rule was written to detect
whether the two calls disagree, and it cannot: the fallback fires only when the verdict says FAIL
while the prose source said PASS, and this critic emits `PASS` on **0 of 30** drafts. With a prose
source that never passes, the fallback rate is 0 whatever the verdict does. **The check is
vacuous on this worker** — it reports the answer the design wanted without having tested the
thing the design cared about. Registering it was right; reading it as evidence the mechanism is
inert would not be.

**What the direct check says instead.** Comparing the recorded completions rather than the
fallback counter: the 60 diagonal completions hold **31 distinct** values, so 29 of the 30 prompts
returned byte-identical text on both surfaces and **one did not**. Against the pre-fix exhibit's
**60 distinct of 60**, `--temperature 0` reaching the critique and verdict surfaces is what moved
0/30 agreement to 29/30. It did not reach 30/30. This registration claimed the diagonal would be
"inert by construction rather than by hope"; at 1/30 that claim is **too strong** and is withdrawn.
Temperature 0 is not a determinism guarantee at the provider — batching and kernel scheduling can
still move a token — so a diagonal cell is inert *in practice and by measurement*, not by
construction. Any future splice must measure it rather than assume it.

**p₀ gate — PASSES, does not trip.** G1's seed-genome held-out mean is **0.7033** (per seed
0.6800, 0.6750, 0.7550), inside the registered `[0.15, 0.95]` band, so the 8B worker leaves room
for a critic to matter. `--arms A` runs no optimizer, so `opt_heldout` equals `seed_heldout`
exactly on all three seeds, as it should.

### Why G2–G4 were still not bought

Both gates passed, so the cancellation is **not** a gate trip. Two independent reasons, both of
which post-date this registration and neither of which is a number from G1:

1. **The cross-lineage critic this design names does not work.** `groq/openai/gpt-oss-120b` is a
   reasoning model: at the 320-token cap it spends the whole allowance on reasoning and returns an
   empty completion, which Paper III's blank-completion guard correctly refuses. Measured 0
   characters at `max_tokens=320` even at `reasoning_effort=low`. G2–G4 all route to it. Recorded
   in the second-worker registration's model table.
2. **There is no lift left to decompose.** The splice exists to attribute a cross-lineage lift.
   On the `llama-3.3-70b` line the cross-lineage critic's *total* effect is $+0.0077$ at
   $t = +0.82$ — nothing — so the 2×2 would be partitioning zero. The oracle `lo`/`mid`/`hi` ladder
   answers the same verdict-versus-prose question better: it holds the verdict perfectly fixed
   while only the prose varies, with no second sampled call, no fallback rule, and no mechanism
   artifact of the kind that moved the OpenAI V1 cell by $+0.027$.

### What G1 is kept for

Not for its contrast values, which the commitments above forbid using. It is kept because it is
the run whose foreign worker enrolled as an eleventh σ replicate — **Paper III's defect 11** — and
the artifact behind a published guard has to be re-derivable or the guard's evidence is a claim
about a file nobody can check. It is registered in `verify_reproducibility.RUNS` as
`splice G1 (diagonal)` and replays bit-for-bit at $0. The numbers in this section are derived
from that artifact and its cassette by
`tests/test_run_validation_study.py::test_the_g1_diagonal_is_inert_by_measurement_not_construction`,
so they cannot drift from what was recorded.

### Standing commitments, discharged

- The observed fallback rate is reported: **0 of 30**, together with the reason it is
  uninformative here.
- No cell was dropped after seeing its number: G2–G4 were dropped on a model-capability fact and
  on a result from a different line, not on G1's contrast.
- No paper sentence claims this run explains the `+0.048`. Paper II keeps its sentence declining
  to attribute it.
