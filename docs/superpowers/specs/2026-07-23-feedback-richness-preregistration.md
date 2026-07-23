# Pre-registration — feedback-richness loop-level GEPA study (run 4)

**Registered:** 2026-07-23, *before* the paid run (this commit precedes any results).

**Follow-up to** run 3 (`docs/validation-study-multihard.md`), the fair test that came
back a clean null: on hard multi-constraint tasks, loop-level GEPA (arm D) did not beat
prompt-only (arm B), Δ = 0.000. Run 3's loop used a **purely intrinsic** critique — no
external feedback signal. Huang et al. 2024 (*LLMs Cannot Self-Correct Reasoning Yet*,
arXiv:2310.01798) show exactly this: intrinsic self-correction fails to improve and often
degrades, and prior "self-correction works" results relied on **oracle** stopping signals
that vanish when removed ("the improvements in these studies result from using oracles to
guide the self-correction process, and the improvements vanish when oracle labels are not
available"). They find self-correction genuinely helps only with a real external
verification mechanism — a **tool / code executor / unit tests**. Run 4 adds that
ingredient and asks whether it changes the loop-level result.

## The feedback signal — a tool, NOT the oracle

The critique step is given `text_stats(draft)`: a deterministic, one-line readout of
*measured* properties of the draft — word count, character count, sentence count, bullet
count, longest-word length, last words. It takes **only the text** — it never sees the task
or its constraints — so it reports "words=23", never "you failed the exact-word-count rule".
The programmatic scorer (`mc_score`) is still never consulted inside the loop.

Crucially, the pre-spend design review flagged that structural non-oracle-ness (no task
argument) is necessary but not sufficient: a field is *informationally* the oracle when its
verifier has **no per-task parameter**, so the raw measurement alone resolves the verdict.
Two constraints are like this — `all_lowercase` (condition: output == lowercase) and
`no_comma` (condition: commas == 0). Reporting an all-lowercase flag or a comma count would
hand back exactly those verdicts, so both are **deliberately omitted**. Every field the tool
does report requires a task-supplied target to become pass/fail (words vs an exact/min
count, bullet_lines vs a required count, longest_word_letters vs a max length), so the model
must still know each target and do the comparison — the tool/executor regime Huang found
effective, not the oracle. The word/longest-word measures match the scorer's tokenizer
(pinned by test) so the tool is faithful, not misleading.

Both B and D loops receive the stats; only the evolvable gene differs. So Δ = mean(D) −
mean(B) still isolates loop-level optimization — now in a feedback-rich regime.

## Hypotheses

- **H1 (primary, directional):** With the stats tool in the loop, loop-level GEPA (D,
  evolves `generate` + `ensure_correct`) achieves a higher mean held-out `mc_score` than
  prompt-only GEPA (B, evolves only `generate`). Null (H0): mean(D) ≤ mean(B), or any gap
  is within seed noise.
- **H2 (secondary, observational):** The feedback loop's seed baseline (arm A) scores
  higher than run 3's no-feedback seed (0.765), isolating whether external feedback helps
  the loop at all (Huang's tool prediction). Reported, not used for the verdict.

## Frozen configuration (do not change after registration)

- **Task set:** `--task-set multi-hard` (same generator as run 3; exact_word_count + ≥1
  more brutal constraint always present; k ∈ {4,5}; satisfiability guaranteed).
- **Feedback:** `--feedback-stats` (injects `text_stats` into the critique step).
- **Arms:** A / B / C / D (identical seed genome, matched budget).
- **Seeds:** 0, 1, 2 (**n = 3** — this is a lower-powered *exploratory* run; a clear D > B
  warrants a 5-seed confirmation, and a null means "no signal at this power", not a tight
  refutation).
- **Tasks:** n_train = 8, n_heldout = 10 per seed (disjoint).
- **GEPA budget:** max_metric_calls = 20/arm; reflection_calls = 4; loop_max_retries = 1
  (worker makes 1 + 2·1 = 3 calls/eval; no extra call for the stats tool — it is local).
- **Models:** worker `openai/gpt-4.1-mini` (max_tokens 320), reflection
  `openai/gpt-5.6-terra` (max_tokens 1500).
- **Budget:** conservative estimate ≈ $3.03; `--max-usd 5`; the key's **$5 hard cap** is
  the true ceiling. Actual expected far lower (run 3 spent ~0.13× its estimate); resumable
  from the cassette if the cap is hit.

## Expected seed band and flags

Feedback may lift the seed, so the pre-registered band is **0.2 ≤ seed ≤ 0.95**. If the
seed saturates (≥ 0.9 → `report.py` ceiling flag fires), H1 is untestable-by-ceiling but H2
is *confirmed* (feedback erased the difficulty) — an informative outcome either way, to be
reported as-is. Floor (≤ 0.1) → untestable-by-floor.

## Verdict rule (decided in advance)

The existing `eval/report.py` rule, unchanged: **SUPPORTED** iff mean(D) − mean(B) > the
summed per-arm stdev and positive; **INCONCLUSIVE** if positive-but-within-noise, < 2
seeds, or a ceiling/floor flag fires; **NOT SUPPORTED** iff mean(D) ≤ mean(B). Controls A
and C are expected flat.

## Commitments

- Report **as-is**, whatever it is; publish alongside runs 1–3.
- No post-hoc changes to arms, budget, seeds, task distribution, feedback signal, or the
  verdict rule.
- Outputs to separate paths: report `docs/validation-study-feedback.md`, raw
  `docs/validation-study-feedback.json`, cassette
  `tests/fixtures/cassettes/study-feedback.json` — earlier runs' artifacts untouched.
