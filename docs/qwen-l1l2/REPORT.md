# Paper I's L1/L2 design on a third worker: result

Pre-registration: `docs/superpowers/specs/2026-10-26-qwen-l1-l2-replication-preregistration.md`,
committed (`a361f38`) before any spend, with the analysis script committed (`1421921`) before any
result was read. Worker `groq/qwen/qwen3.8-27b`, reflection `groq/openai/gpt-oss-120b`, four arms,
10 independent seeds per level, Paper I's p₀-run configuration otherwise. L1 is the hard tier; L2
is the same tier with `exact_word_count` excluded. Analysis: `scripts/analyze_qwen_l1l2.py` →
`docs/qwen-l1l2/analysis.json`. Every number below is in that file or in `replay.json`.

**Reproducible at $0.** `scripts/replay_qwen_l1l2.py` re-runs both levels under `--replay` from the
committed cassettes and compares `seed_heldout`, `opt_heldout`, `genes_changed`, `final_genes` and
`train_score` for every arm and seed with the released JSON: 400 fields, 0 mismatches. A copy of the
L1 JSON with one `opt_heldout` moved by 0.005 fails the same check, naming the field.

**Spend.** A floor of $1.29 (L1) and $0.94 (L2), $2.23 in all, against a pre-flight bound of $9.97
per level. The floor prices one call per distinct prompt, with the caveats in the script's
docstring. With the target-p₀ program (≈ $13, an estimate), the Groq authorisation has spent at
least $15 of $60: the floor leaves out the resampled empty completions and the calls lost in the
three timed-out L2 attempts.

## What ran

Both levels ran as registered. L1 recorded in one attempt: 1533 worker and 96 reflection calls.
L2 needed four. Attempts 1 to 3 each ended when the provider did not answer within litellm's 600 s
timeout (`litellm.Timeout: Connection timed out after 600.0 seconds`). Each attempt resumed from
the cassette, so calls already recorded were replayed, not bought again, and no result was read
between attempts. The fourth attempt finished: 1153 worker and 69 reflection calls in all.

Integrity counts: 0 empty outputs stored in either cassette, 0 `<think>` traces, and 22 (L1) and 25
(L2) empty completions that the runner resampled under `--empty-attempts 6`. Every arm's held-out
score comes from recorded worker outputs. No run stored a score without reaching the model.

## Registered verdict

**H1 is INCONCLUSIVE.** S = mean Δ_L2 − mean Δ_L1 = −0.0100, 95 % bootstrap CI [−0.0575, +0.0355],
Welch t = −0.40 (df 15.2, p = 0.70). The manipulation checks pass, so the levels were built and the
contrast is a test of the design.

| level | realized p₀ (M1) | arm A held-out (M2) | D changed `ensure_correct` (M3) | B | D | Δ(D−B) | 95 % CI | sd | seeds +/−/0 |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 0.04 | 0.721 | 8/10 | 0.7260 | 0.7235 | −0.0025 | [−0.0410, +0.0385] | 0.0676 | 4/6/0 |
| L2 | 0.76 | 0.936 | 7/10 | 0.9570 | 0.9445 | −0.0125 | [−0.0385, +0.0110] | 0.0428 | 3/5/2 |

- M1: realized p₀ is higher at L2 (0.76) than at L1 (0.04). Passes.
- M2: arm A's held-out mean at L2 is 0.936, under the registered 0.95 flag. Passes. See the
  headroom note below: the rule passes, and the ceiling still binds.
- M3: arm D changed `ensure_correct` on 8 and 7 of 10 seeds. The treatment was applied.

**What the interval rules out.** At 95 %, a swing above +0.036 on this worker by the registered
percentile bootstrap; a Welch t interval, [−0.064, +0.044] (scipy, computed in the final review,
not by the registered script), rules out only a swing above +0.044. Paper I's swing was
+0.102 (−0.042 at L1, +0.060 at L2, gpt-4.1-mini, three seeds a level). Most of that exclusion is a
ceiling, not evidence: arm B scores 0.957 at L2, so arm D could beat it by at most 0.043 there, and a
gain of Paper I's L2 size (+0.060) was out of reach at this level on this worker. The registered M2
rule watches arm A and did not fire; the headroom that matters for Δ is arm B's.

**What it does not say.** It says nothing about p₀ as a cause: L1 and L2 are two generators, the
baselines differ (0.73 against 0.96), and this worker puts them at p₀ 0.04 and 0.76, far from
Paper I's 0.067 and 0.267. It is one worker.

## Secondary (registered, descriptive)

The evolved critique's first verdict on held-out tasks, read with the harness rule (the critique
counts as PASS when its stripped, upper-cased text starts with `PASS`):

- L1: arm D's critique emits PASS on 9 of 100 held-out tasks; 5 of 10 seeds emit any PASS. All 9
  are false PASSes.
- L2: PASS rate 0.44, all 10 seeds emit PASS, 10 false PASSes.

## Exploratory (added after the results were read)

The same counts for the seed critique (arm A) and the prompt-only arm (B), whose critique never
changes, separate what evolution did from what the seed critique already did.

| level | arm | realized p₀ | PASS rate | seeds with any PASS | false PASSes |
|---|---|---|---|---|---|
| L1 | A | 0.04 | 0.18 | 8 | 18 |
| L1 | B | 0.05 | 0.19 | 9 | 19 |
| L1 | D | 0.06 | 0.09 | 5 | 9 |
| L2 | A | 0.76 | 0.49 | 10 | 11 |
| L2 | B | 0.78 | 0.51 | 10 | 7 |
| L2 | D | 0.80 | 0.44 | 10 | 10 |

At L1 every PASS from every arm is false: the critique never recognises one of the few compliant
first drafts. Evolution halved the PASS rate there (0.18 and 0.19 for the fixed critiques, 0.09 for
the evolved one) and left half the seeds' evolved critiques with no PASS at all. That is the
direction Paper I reports for its evolved verifier at low p₀ (11 of 12 genes silent), in a weaker
form, and it is not significant: 9 against 18 and 19 PASSes in 100 gives Fisher p = 0.097 and
0.065 (scipy, final review), before allowing for the tasks clustering in ten seeds. At L2 evolution barely moved the rate. On this worker the change in the verifier did not show
up as a difference in held-out score at either level.

## What a reader should take from it

1. On qwen3.8-27b, ten seeds a level, the L1/L2 design gives Δ(D−B) close to zero at both levels
   and no detectable difference between them.
2. The run cannot test a gain as large as Paper I's at L2, because this worker's baseline leaves
   0.043 of headroom there.
3. The evolved critique accepts less often at the low level, an exploratory and not significant
   difference. It costs nothing measurable and buys nothing measurable.
