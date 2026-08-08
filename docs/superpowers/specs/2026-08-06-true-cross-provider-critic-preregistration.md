# Pre-registration — D3, the true cross-provider critic

**Registered:** 2026-08-06, *before* the paid run. At registration the run cannot be executed at
all: see "Blocker" below. That is the point of writing this now — the configuration is frozen
before anyone knows whether the credential will arrive, so nothing here can be tuned to a result.

**Follow-up to** run 6 (`docs/validation-study-crossfamily.md`), whose own pre-registration
(`2026-07-23-cross-family-critic-preregistration.md`) flagged its central limitation in a section
headed "⚠ Cross-family caveat": the critic was `openai/gpt-5.6-luna` against an
`openai/gpt-4.1-mini` worker, which is **cross-lineage, not cross-provider**. Same vendor,
tokenizer lineage, and likely training-data and stylistic priors. Paper I lists the true test as
D3 and marks it *blocked on a credential*.

## Why this run matters more than it did a week ago

On 2026-08-05 the series **withdrew** the claim that "cross-family verification outperforms
self-verification" from Papers I and II (commit `db8065e`). Three sites attributed that result to
`valmeekam2023selfcritique` and `huang2023cannot`; neither source contains it. The clause was
withdrawn rather than re-sourced, because no source establishing it had been read.

So the series now asserts a **gap** where it used to assert a borrowed finding. D3 is the
experiment that would fill the gap with our own measurement instead of someone else's. A null is
just as publishable and arguably more interesting: it would say that the self-preference story,
which the literature is confident about and we could not source, does not reach held-out score in
this loop.

## Choice of critic, and one trap

**Primary: `groq/llama-3.3-70b-versatile`** ($0.59 / $0.79 per Mtok, published pricing read
2026-08-06). Different weights, different training lineage, different vendor. Stronger separation
than the two options the earlier registration pre-priced, both of which are proprietary frontier
families that may share more with the worker than their vendor labels suggest.

**Not `groq/openai-gpt-oss-120b`**, at a quarter the price. It is OpenAI's own open-weights
lineage, so pairing it with an `openai/*` worker rebuilds precisely the confound this run exists to
remove. It is priced in `MODEL_PRICES` so that picking it is a decision on the record rather than an
oversight.

**Fallbacks, in order, if Groq stays unreachable:** `anthropic/claude-haiku-4-5` (reachable from
this host, needs a key), then `gemini/gemini-2.5-flash-lite`. Switching to a fallback is a change of
critic and therefore must be recorded as a deviation, with the reason, in the run log.

## Blocker at registration

Groq's API is unreachable from this host and the reason is diagnosed, not guessed:

| endpoint | status |
|---|---|
| `groq.com` | 200 |
| `console.groq.com` | 403 |
| `api.groq.com/openai/v1/models` | 403, `server: cloudflare`, `cf-ray … -MAD` |

The 403 is **identical with and without an `Authorization` header**, so it precedes
authentication: it is not a bad key. Egress from this host is `45.134.213.202`, `AS212238 Datacamp
Limited` (ES) — a datacenter range, which Groq blocks at its edge. **The key's validity and
remaining quota are therefore unknown and unknowable from here**: a 403 before authentication looks
identical for a fresh key and an exhausted one. The fix is on the operator's side: run from a
residential connection. No attempt was made to route around the block.

The credential itself is settled and is *not* the obstacle. It was shared deliberately by a
colleague on 2026-07-31 in response to a direct request, on the stated understanding that the spend
would be small. That understanding is respected by the configuration: only the critique call routes
to Groq, one of every three worker-side calls, so the Groq share of the bound is **$0.69** and the
expected real spend at the measured 23% realization is **$0.16**. The remainder is on this project's
own OpenAI account. The key is stored at `scratchpad/groq.key` (mode 600, gitignored) beside the
OpenAI one.

The OpenAI worker key at `scratchpad/openai.key` is live (HTTP 200, verified 2026-08-06). The stale
copy in `$OPENAI_API_KEY` returns 401 and must not be used.

## Hypothesis

**H1 (directional).** With the critique routed to a different-provider, different-lineage model,
loop-level GEPA (arm D, evolves `generate` + `ensure_correct`) achieves a higher mean held-out
`mc_score` than prompt-only GEPA (arm B, evolves only `generate`).
**H0:** mean(D) ≤ mean(B), or any gap is within seed noise.

**H2 (secondary, observational).** The cross-provider critic is a better failure detector than
run 3's intrinsic self-critique (F₁ ≈ 0.11) and than run 6's cross-lineage critic. Measured
post-hoc from the cassette via `scripts/analyze_critique_detector.py` on the `critic` namespace.
Reported, never used for the verdict.

**H3 (the one the withdrawal created).** If H1 is null while H2 shows the cross-provider critic IS
the better detector, that is direct evidence for Paper II's thesis — detection is not the binding
constraint — obtained on the exact axis the withdrawn literature claim used to cover.

## Frozen configuration (do not change after registration)

- **Task set:** `--task-set multi-hard` (same generator as runs 3 and 6; the metric's do-nothing
  floor is 0.42 here against a 0.77 baseline, whereas on `ifbench` the floor sits at or above the
  reported baselines).
- **Critic:** `--critique-model groq/llama-3.3-70b-versatile`, critique step only; generate and
  refine stay on the worker.
- **Arms:** A / B / C / D, identical seed genome, matched budget.
- **Seeds:** 0–7 (**n = 8**), as Paper I's D3 entry specifies. Run 6 used 3 and was explicitly
  underpowered; σ(Δ) on this family is ≈ 0.029, so 8 seeds is the smallest n that puts the SESOI
  of ±0.05 outside the noise band rather than inside it.
- **Tasks:** n_train = 8, n_heldout = 10 per seed, disjoint.
- **GEPA budget:** max_metric_calls = 20/arm, reflection_calls = 4, loop_max_retries = 1.
- **Models:** worker `openai/gpt-4.1-mini` (320 output cap), critic
  `groq/llama-3.3-70b-versatile` (320), reflection `openai/gpt-5.6-terra` (3000).
- **Ceiling:** see the next section. The dry run is the authority; no bound is typed here by hand.

## Priced, 2026-08-06, before any spend — and the registered config does not fit

Every figure below is `scripts/run_validation_study.py` refusing or reporting, offline, with the
Groq critic priced in. The registered configuration comes to **$6.52**, over the study's standing
$5 key cap. That is a real finding of the pricing step and it is recorded rather than resolved by
quietly shrinking the run.

| configuration | bound |
|---|---|
| **ABCD, 8 seeds, reflection-calls 4, terra — as registered** | **$6.52** |
| ABCD, 8 seeds, reflection-calls 2, terra | $4.76 |
| ABD, 8 seeds, reflection-calls 4, terra (drops control C) | $6.04 |
| ABCD, 5 seeds, reflection-calls 4, terra | $4.07 |
| ABCD, 8 seeds, reflection-calls 4, luna | $4.41 |

A first pricing attempt returned $11.80 because `--reflection-calls` defaults to **10** and the
flag was omitted; the registered value is 4. Noted because it is the same species of error the
series keeps finding: a number that looked like the configuration's cost and was actually the
default's.

**Recommendation: `--max-usd 7` and run as registered**, rather than take any of the cheaper rows.

A standing budget of $20 does not mean passing `--max-usd 20`. The ceiling is a *guard*, and its
job is to stop a run that has gone wrong; set just above the priced bound it does that, and set at
the budget it does nothing until eight times the intended spend has already gone.

- 5 seeds recreates exactly what made run 6 uninformative. Paper I's D3 entry specifies ≥ 8, and
  σ(Δ) ≈ 0.029 on this family means 8 is already the smallest n that puts the ±0.05 SESOI outside
  the noise band.
- reflection-calls 2 halves the optimizer's proposal budget. D4 closed the question of whether
  search depth matters here, so this is defensible on the evidence — but it introduces a second
  difference from run 6 and makes the comparison to it a two-variable one.
- Dropping arm C removes the genome-blind control, which is the arm that distinguishes "the
  treatment did nothing" from "any edit would have done the same".
- `luna` is the cheapest row and the worst choice: it returned content-free completions on the C2
  run, one of which was accepted as an empty gene. That is defect 8 of Paper III, and it was
  caused by this model.

The bound is an upper bound and a loose one. The measured realization on the C1 run was **23%** of
the guard's figure ($0.043 actual against a $0.190 bound), because the guard prices every
worker-side call at the pricier of {worker, critic} and assumes full output caps. Expected real
spend for the registered configuration is therefore of order **$1.50**, not $6.52.

## Expected seed band and flags

Pre-registered band **0.2 ≤ seed ≤ 0.85**. The seed loop uses the cross-provider critic, so its
held-out may differ from run 3's 0.765. A ceiling (≥ 0.9) or floor (≤ 0.1) flag from `report.py`
makes H1 untestable and is reported as such.

## Verdict rule (decided in advance)

The unchanged `eval/report.py` rule. **SUPPORTED** iff mean(D) − mean(B) is positive and exceeds
the summed per-arm stdev. **INCONCLUSIVE** if positive but within noise, if fewer than 2 seeds
complete, or if a ceiling/floor flag fires. **NOT SUPPORTED** iff mean(D) ≤ mean(B). Controls A and
C are expected flat; an arm that accepts nothing on every seed makes its contrast a
D-vs-seed comparison wearing a D-vs-B label, and `analyze_deltas.arms_optimized` must be checked
before the Δ is read — that is defect 1 of Paper III and defect 9 is its mirror.

## Commitments

- Report **as-is**, whatever it is.
- No post-hoc change to arms, budget, seeds, task distribution, critic model, or verdict rule.
  A forced change of critic is a recorded deviation, not an edit.
- Outputs to separate paths: `docs/validation-study-xprovider.{json,md}`, cassette
  `tests/fixtures/cassettes/study-xprovider.json`. Earlier runs untouched.
- The per-seed held-out vector is released, because a seed re-partitions a fixed pool rather than
  drawing a fresh sample on some task sets, so the unit to pair on is the task.
- If the run is never bought, this document stays as a registered-and-unrun record. Paper I's D3
  entry is updated to say the credential is the only remaining obstacle, with the diagnosis above,
  rather than the vaguer "blocked on a credential".

## The exact command, for when the network is open

Nothing below needs editing. The pre-spend preflight refuses at zero token cost if Groq still will
not answer, so running it while blocked is safe and costs nothing.

```bash
cd ~/whystuck
OPENAI_API_KEY=$(tr -d '\r\n' < scratchpad/openai.key) \
GROQ_API_KEY=$(tr -d '\r\n' < scratchpad/groq.key) \
.venv/bin/python scripts/run_validation_study.py --yes \
  --task-set multi-hard --arms ABCD --seeds 0,1,2,3,4,5,6,7 \
  --n-train 8 --n-heldout 10 --max-metric-calls 20 --loop-max-retries 1 \
  --reflection-calls 4 \
  --worker-model openai/gpt-4.1-mini \
  --critique-model groq/llama-3.3-70b-versatile \
  --reflection-model openai/gpt-5.6-terra \
  --max-usd 7 \
  --record tests/fixtures/cassettes/study-xprovider.json \
  --out docs/validation-study-xprovider.json \
  --report docs/validation-study-xprovider.md
```

Read the preflight line before anything else. `HTTP 200` means the run is live. `HTTP 403` means the
host is still blocked and nothing was spent. **`HTTP 429` means the credential works and its quota
is gone** — which is the only way that question can be answered, and it cannot be answered from a
blocked host at all.

---

## RESULT — run bought and completed 2026-08-06

Executed exactly as registered: ABCD, seeds 0–7, `groq/llama-3.3-70b-versatile` critic,
`--max-usd 7`, bound $6.52. Artifacts: `docs/validation-study-xprovider.{json,md}`, cassette
`tests/fixtures/cassettes/study-xprovider.json` (993 calls: 604 worker, 347 critic, 42 reflection).
No deviation from the frozen configuration.

**H1: INCONCLUSIVE** by the registered rule. mean(D) − mean(B) = **+0.0144**, positive but far
inside the noise band: sd(D) + sd(B) = 0.0815. Reported as-is.

**Read the acceptance census before the Δ, per Paper III defect 1.** Arm B accepted on 5 of 8
seeds, arm D on **3 of 8**. Two independent methods agree: the `genes_changed` field, and the
score-based check that an arm accepting nothing reproduces its own seed score.

The Δ decomposes into strata that say something quite different from the headline:

| stratum | seeds | mean Δ(D−B) | share of the headline |
|---|---|---|---|
| D moved — the treatment actually applied | 3 | +0.0100 | 26% |
| **only B moved, D returned the seed** | **2** | **+0.0425** | **74%** |
| neither moved | 3 | +0.0000 | 0% |

**Three quarters of the positive result comes from two seeds where the treatment did not apply
and the comparator damaged itself** (B lost 0.065 and 0.020 against its own seed genome). On those
seeds Δ(D−B) is really Δ(seed−B), a one-sided contrast wearing a D-vs-B label — defect 1's mirror,
arising unprompted in a fresh pre-registered run. Restricted to the seeds where arm D actually
accepted something, Δ = +0.0100 on n = 3.

**H2: the cross-provider critic is the WORST detector of the three, not the best.**

| critic | n | PASS rate | tp | fp | fn | p₀ | F₁ |
|---|---|---|---|---|---|---|---|
| run 3, intrinsic (worker grades itself) | 50 | 0.26 | 1 | 12 | 4 | 0.100 | 0.111 |
| run 6, cross-lineage `gpt-5.6-luna` | 30 | 0.13 | 4 | 0 | 0 | 0.133 | 1.000 |
| **D3, cross-provider `llama-3.3-70b`** | **80** | **0.00** | **0** | **0** | **9** | 0.113 | undefined |

The cross-provider critic **never emits PASS**, missing all nine already-correct drafts. It is
behaviourally "always refine": the degenerate verifier this series keeps meeting, reached from the
direction the self-preference literature predicts should have improved things. Crossing the
provider boundary did not buy a better detector; crossing the *lineage* boundary did, and by a
wide margin on a small sample (30 events, 4 positives — not overclaimed).

**H3 is answered, and it is the run's real contribution.** The withdrawal in `db8065e` removed the
claim that cross-family verification beats self-verification because no source we had read
established it. This run supplies our own measurement, and it does not support the claim either:
the more cross-family the critic, the more degenerate the detector, and held-out score does not
move in any case. Paper II's thesis — detection is not the binding constraint — now has evidence on
the exact axis the borrowed citation used to cover. Run 6 sharpens it: a critic with F₁ = 1.00 on
its sample still produced no held-out movement, and its arm D accepted nothing on any seed.

**A correction to the record for run 6.** `arms_optimized` reports `{'B': 0, 'D': 0}` for it, which
is the absent-field-read-as-zero defect corrected in `524149c`: `validation-study-crossfamily.json`
predates `genes_changed`. The score-based check gives the real figure — arm B accepted on 1 of 3
seeds, arm D on **0 of 3**. Run 6's Δ was a D-vs-seed contrast throughout.

**Tooling gap this exposed, now closed.** `analyze_critique_detector.collect_events` replayed only
the `worker` surface, so a run recorded with `--critique-model` missed every critique call — and a
`CassetteMiss` is *skipped*, so it returned a short list that reads as a smaller sample rather than
a broken replay. All 80 of this run's first-critique events vanished that way on the first attempt.
`separate_critic=True` wires the `critic` namespace. The earlier registration predicted this
extension would be needed and it was.

## Post-hoc, NOT registered — the seed-genome axis, and what it forces

Everything below was computed after the run landed, while checking that the three papers'
neighbouring paragraphs still fitted. It is reported as post-hoc analysis and no verdict rests on
it. It is here because it changed how Paper II states its own result.

The closing argument of Paper II's detector section had been comparing run 4's `+0.060` against
this run's `+0.014` and run 6's `+0.008` as though the three were the same quantity. **They are
not.** `+0.060` is a *seed-genome* between-run gap — no optimizer anywhere — while the other two
are Δ(D−B) between *optimized* arms. Set side by side they read as "the tool bought four times what
a model swap bought", which is not a comparison anyone made deliberately.

Both substituted-critic runs turn out to be eligible for the curve's paired comparison: each
carries all 54 of the byte-identical first-draft worker calls runs 3, 4 and 7–10 share, so their
seed-genome scores pair with run 3's on the same thirty held-out tasks. Measured that way, over the
three seeds the curve shares:

| critique | seed-genome held-out gap vs run 3 | detector |
|---|---|---|
| run 4, measurement tool | **+0.060** | precision 1.00, F₁ 0.83 |
| run 6, cross-lineage `gpt-5.6-luna` | **+0.048** | F₁ 1.000 (n=30) |
| D3, cross-provider `llama-3.3-70b` | **+0.007** | never passes (n=80) |
| run 7, perfect oracle | **−0.028** | perfect by construction |

against σ(Δ) = 0.029. The cross-lineage critic lifts the *unoptimized* loop by nearly as much as
the measurement tool, which is not what the section previously implied, and is a fact the paper now
states.

It does not overturn the thesis, and the reason is a confound the run cannot resolve: **a model
swap is not a detection-only intervention.** A different model rewrites the critique's prose as
well as its verdict. The released `critic` cassettes make that visible — the cross-lineage critic
answers in a median of 53 characters ("add one word; all other rules pass."), the cross-provider one
in 210, usually with a rewritten answer attached. A proxy for "does the critique quote a measured
quantity" (contains a digit) does **not** separate them, 73% against 82%, so the prose difference is
in form rather than in whether a number is present. The only clean verdict-only intervention we have
is the oracle, and it is the worst of the four.

**What this forces: E1, the verdict/prose splice.** Two `--arms A` cells at the curve's three seeds,
carrying one model's verdict with another model's prose:

- (a) cross-lineage **verdict**, intrinsic **prose**
- (b) intrinsic **verdict**, cross-lineage **prose**

If (b) carries the `+0.048` and (a) does not, the lift is feedback and Paper II's thesis gains its
cleanest confirmation on the one data point that currently looks like a counterexample. If (a)
carries it, the thesis needs qualifying, and it needs qualifying *before* submission rather than
after. Cost is small — the c1 conditions show a single-seed `--arms A` cell at ≈ $0.10 — but it is
**not free**: the refine step's prompt changes, so its worker calls are new. It also needs a harness
change, since no flag today separates the verdict from the prose of one critique step.

**Remaining from the registered list, unchanged:** the same comparison with an evolved rather than
a seed critique, and a second cross-provider family to show the degeneracy is not one model's quirk.

## Actual Groq spend, measured 2026-08-08

The `estimated_usd` of `$6.5198` recorded in the run's meta is the pre-flight **ceiling for the
whole run**, dominated by the OpenAI worker and the `gpt-5.6-terra` reflection. It says nothing
about the borrowed Groq credential's share, which is the only figure that matters for repaying it.

Measured by replaying D3 offline with the `critic` wrap instrumented (prompts are stored as
hashes, so input tokens are not readable from the cassette; replay reconstructs each prompt
byte-for-byte, which is why the replay matches at all):

| quantity | value |
|---|---|
| billed critic calls | **347** (unique prompts; equals the cassette's critic-entry count) |
| input tokens | 95 937 |
| output tokens | 22 193 |
| input cost, at $0.59/10⁶ | $0.0566 |
| output cost, at $0.79/10⁶ | $0.0175 |
| **total** | **≈ $0.074** |

1124 critic calls fire during a replay but only 347 are distinct; during the original recording a
repeated prompt is served from the cassette and never billed, so 347 is the billed count.

Counted with `cl100k_base`, which Llama 3.3's 128k tiktoken-style BPE extends but does not equal.
A chars/4 estimate gives $0.084, so the honest range is **$0.07–0.08**. On-demand tier: the probe
response records `service_tier: on_demand`, so neither the batch nor the prompt-caching 50%
discount applies. A failed-and-retried call would have been billed without leaving a cassette
entry, so this is a floor; the run log shows no retries.

An earlier estimate in conversation put this at ≈ $0.26. That was scaled off the run's dollar
bound and is **superseded** by the count above.
