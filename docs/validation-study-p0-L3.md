# WhyStuck validation study: loop-level vs prompt-only GEPA

*Generated: 2026-07-29T09:15:17.463957+00:00.*

## Method

- **Dataset:** multi-constraint-hard (IFEval-style) — 8 train / 10 held-out tasks, scored by programmatic constraint verifiers (no LLM judge).
- **Worker (task LM):** `openai/gpt-4.1-mini`; **reflection LM:** `openai/gpt-5.6-luna`.
- **Worker loop:** each task eval runs a generate→critique→refine self-refine loop (up to 1 refine round(s)). The `generate` gene drives the draft; the `ensure_correct` gene drives an *intrinsic* critique (the verifier is never consulted mid-loop — it is the scoring metric, so gating on it would be oracle contamination). The final answer is scored programmatically. Evolving the verify gene changes the critique, hence the loop's output — the lever D has and B does not.
- **Budget:** `max_metric_calls`=28 per arm, across 3 seed(s) [0, 1, 2]; estimated spend $1.79.
- **Arms:** A = seed baseline (no optimization); B = prompt-only GEPA (evolves one instruction gene); C = fixed-draft control (a constant draft graded by the real metric, so the genome is genuinely ignored but the landscape is not saturated); D = loop-level GEPA (evolves the whole genome). All arms share an identical seed genome, so held-out gaps isolate what GEPA evolves, not prompt length.

## Results

| Arm | What GEPA evolves | seed held-out | optimized held-out (mean ± std) | Δ |
|-----|-------------------|---------------|---------------------------------|---|
| A | seed baseline (no optimization) | — | Pending | — |
| B | prompt-only GEPA (one gene) | 0.94 | 0.97 ± 0.02 | +0.03 |
| C | fixed-draft control (genome-blind) | — | Pending | — |
| D | loop-level GEPA (whole genome) | 0.94 | 0.97 ± 0.04 | +0.03 |

## Verdict — NOT SUPPORTED

D (0.97) did not beat B (0.97) (Δ=-0.00). On this benchmark and budget, loop-level GEPA showed no advantage over prompt-only.

⚠ **Ceiling effect.** The seed baseline already scores 0.94 (near the 1.0 ceiling), so no arm has **headroom** to improve. This run cannot discriminate loop-level from prompt-only regardless of the verdict above — read it as inconclusive by saturation, not as evidence against the thesis. A fair test needs harder tasks that push the seed well below 1.0.

## Caveats

- Arm C's flatness is the genome-blind control; its optimization can't move the score because the fixed-draft worker ignores the genome.
- The split is small (10 held-out); this claims a *relative* result (loop-level vs prompt-only) under a matched budget, not a leaderboard number.
- Reflection spend is a hard cap: GEPA is told to stop after `reflection_calls` proposals per arm (one reflection LM call each), so the estimated cost is a genuine ceiling, not just a guess.
- Worker completions are hard-capped at 320 tokens (output tokens — the dominant cost — are a genuine ceiling the guard prices exactly). Input tokens are a conservative estimate sized for GEPA-evolved instructions, not an absolute bound, so the guard is enforced pre-flight on that estimate rather than metered mid-run. The instruction-following constraints used here have short outputs, so the cap does not truncate answers; if it ever did, that floor would apply to every arm equally, not bias the loop.
