# Pre-registration — D2′ Feedback-Actionability Ablation (Runs 9 & 10)

**Date:** 2026-07-23
**Status:** COMPLETE (runs 9 & 10, 2026-07-23). Pre-registered BEFORE spending; frozen, no post-hoc changes.

## RESULT (frozen, reported as-is)

Seed held-out (arm A, no GEPA), mid = run 8 reused:

| level | seed held-out A (mean ± std, n=3) |
|-------|-----------------------------------|
| lo (verdict only) | 0.810 ± 0.049 |
| mid (which-constraint = run 8) | 0.813 ± 0.044 |
| hi (concrete measured edits) | **0.873 ± 0.027** |

**Δ(hi − lo) = +0.063 ≥ +0.05 → FEEDBACK-LIMITED** (H supported for the fix step). Sub-findings:
lo ≈ mid (which-constraint naming adds ~0); the actionable content is the measurement, not the
diagnosis. Lever is feedback SHAPE, not loop-level GEPA: Δ(D−B)=0.00 at every level; opt slightly
degrades both arms. hi (0.873) is the first oracle config to exceed the tool baseline (0.837),
resolving the run-4 tool>oracle inversion as an actionability effect. Residual generator-limit
remains (hi saturates < 0.9). Both runs replay bit-for-bit ($0); reproducibility harness now
covers all 10 runs. Reports: `docs/validation-study-fbactlo.{md,json}`,
`docs/validation-study-fbacthi.{md,json}`; cassettes `study-fbactlo.json`, `study-fbacthi.json`.

## Motivation

Run 7 (oracle critique) and Run 8 (oracle + refine-budget 3) established the study's
sharpest mechanistic result: **the bottleneck is the FIX step, not detection.**

- Run 7 — a *perfect* detector (oracle gate) did NOT lift the ceiling: seed held-out
  0.748, BELOW the tool-fed detector's 0.837 and ≈ the intrinsic 0.765. Detector
  quality vs performance is non-monotonic (F1 0.11 → 0.83 → 1.0 gives seed 0.765 →
  0.837 → 0.748).
- Run 8 — oracle + 3 refine rounds pushed seed 0.748 → **0.813**: more attempts help
  but saturate below the 0.85 bar and below the tool baseline.

So the residual gap is one of two things, and Runs 7–8 cannot separate them:

1. **Feedback-limited** — the model *can* fix hard constraints given a good enough
   repair instruction, but the mid-actionability feedback ("failed exact_word_count")
   does not tell it *how*.
2. **Generator-limited** — the model simply cannot execute the edit even when told
   exactly what to change; the residual is raw capability.

D2′ separates them by holding **detection fixed at perfect** (oracle gate, so
detection is not a confound) and varying only **feedback actionability**.

## Hypothesis

H: At fixed perfect detection (oracle critique, R=3), increasing the actionability of
the repair feedback raises the seed held-out score.

- **Feedback-limited world** ⇒ Δ(hi − lo) > 0 materially (say ≳ +0.05). Loop-level
  GEPA *could* matter after all — just via evolving feedback SHAPE, not via the
  genes tried in runs 3–8.
- **Generator-limited world** ⇒ Δ(hi − lo) ≈ 0 within n=3 noise. Confirms the
  loop-level null is fundamental for this model/task: even maximal, exact,
  measured repair guidance cannot push gpt-4.1-mini past ~0.81 on hard constraints.

Either outcome is publishable and closes the mechanism question the paper opens.
This is NOT a D-vs-B test — under a perfect oracle the verify gene is inert (see
`make_self_refine_run_task` docstring). The quantity of interest is Δ across
**feedback levels**, read on the SEED held-out score (arm A, no GEPA), which
isolates the raw loop's fix step from optimization entirely.

## Design — three actionability levels at one fixed config

All three share Run 8's exact config: `--oracle-critique`, `--loop-max-retries 3`,
`--task-set multi-hard`, seeds `0,1,2`, `--n-train 8 --n-heldout 10`,
`--max-metric-calls 20`, `--reflection-calls 4`, worker `gpt-4.1-mini`, reflection
`gpt-5.6-terra`. The ONLY thing that varies is the repair-feedback text fed into the
refine step when the oracle gate reports a failure:

| Level | `--feedback-level` | Critique text on failure | Provenance |
|-------|--------------------|--------------------------|------------|
| **lo**  | `lo`  | verdict only: `"not all requirements are satisfied; revise"` (no which-constraint) | Run 9 (new) |
| **mid** | `mid` (default) | `mc_feedback`: names each failed constraint + its requirement (e.g. "failed exact_word_count (use exactly 20 words)") | **= Run 8** (reuse cassette `study-fixbudget.json`) |
| **hi**  | `hi`  | `mc_feedback` + `mc_edit_directives`: concrete measured edits ("draft has 24 words, remove 4"; "'the' appears 3×, delete all"; "add 2 bullet lines") | Run 10 (new) |

**Budget note:** mid already exists as Run 8, so only lo and hi are spent. Two paid
runs, matched to Run 8 in every other respect.

### `mc_edit_directives` (the "hi" treatment)

For each FAILED constraint, emit a concrete, measured edit computed from the draft +
the task params (this is legitimate under the oracle arm — detection is already
conceded perfect; the treatment variable is repair guidance, not detection):

- `exact_word_count` n: "your draft has {cur} words but needs exactly {n}: {add/remove |cur−n|}"
- `min_words` n: "add at least {n−cur} more words (draft has {cur})"
- `forbidden_word` w: "the word '{w}' appears {c} time(s); remove every one"
- `word_max_length` m: "these words exceed {m} letters: {list}; replace with shorter ones"
- `num_bullets` n: "you have {cur} bullet line(s) but need exactly {n}: {add/remove}"
- `include_keyword` k: "insert the word '{k}' somewhere in the response"
- `all_lowercase`: "lowercase these characters: {uppercase chars present}"
- `end_with` p: "make the response end with the exact phrase '{p}'"
- `no_comma`: "remove all {c} comma(s)"

Only failed constraints get a directive (passing ones are silent).

## Oracle-honesty statement

The "hi" directive uses task parameters to compute exact edits. This is NOT the
oracle-contamination failure mode guarded against in Runs 4–6, because:

1. The arm is EXPLICITLY the oracle condition — the detector gate already uses the
   true metric (`--oracle-critique`). We are past "is detection honest"; detection
   is deliberately perfect here by construction.
2. The treatment variable is **repair-instruction actionability**, and the whole
   point is to see whether maximal guidance moves the fix ceiling. Making the guidance
   maximally concrete is the experiment, not a leak.
3. The programmatic scorer that produces the FINAL reported score is unchanged and is
   still never gamed — the loop's output is graded by `mc_score` exactly as in every
   other run.

Documented so no reader mistakes "hi" for a contaminated non-oracle arm.

## Metrics & analysis (frozen)

- **Primary:** seed held-out score at each level (arm A), lo vs mid vs hi.
  Primary contrast Δ = hi − lo, reported as mean ± std over 3 seeds.
- **Secondary:** optimized held-out (arms B/D) at each level, for completeness; C the
  usual flat control.
- **Pre-registered bands:** if any seed held-out ≥ 0.9 flag ceiling, ≤ 0.1 flag floor
  (uninformative). Runs 7–8 sat at 0.75–0.81, so headroom is expected.
- **Decision rule:** Δ(hi − lo) ≥ +0.05 ⇒ feedback-limited (report H supported for
  the fix step). |Δ| < 0.05 ⇒ generator-limited (report null; the fix ceiling is
  capability, not guidance). No other cut. Report as-is, no p-hacking.

## Reproducibility

Both new runs record cassettes (`study-fbactlo.json`, `study-fbacthi.json`) and are
added to `scripts/verify_reproducibility.py` (EXPECTED_RUNS 8 → 10) so the full
suite still replays bit-for-bit offline ($0). mid needs no new cassette (Run 8's
`study-fixbudget.json` is it).

## Implementation checklist (TDD)

1. `mc_edit_directives(output, task)` in `multi_constraint.py` — failing test first.
2. `oracle_feedback` param on `make_self_refine_run_task` — oracle path uses it when
   set, else falls back to `feedback` (default None ⇒ existing runs unchanged).
3. Thread `oracle_feedback` through `run_study`.
4. `--feedback-level {lo,mid,hi}` on the CLI; builds the right `oracle_feedback`
   (lo/hi) or None (mid); records `feedback_level` in meta. Only meaningful with
   `--oracle-critique` (warn/no-op otherwise, documented).
5. Full offline e2e through real GEPA; dry-run estimate ≤ remaining budget; then spend.
