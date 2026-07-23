# WhyStuck validation study: loop-level vs prompt-only GEPA

*Generated: 2026-07-23T05:26:30.500840+00:00.*

## Method

- **Dataset:** IFBench — 16 train / 16 held-out tasks, scored by programmatic constraint verifiers (no LLM judge).
- **Worker (task LM):** `openai/gpt-4.1-mini`; **reflection LM:** `openai/gpt-5.6-terra`.
- **Worker loop:** each task eval runs a generate→critique→refine self-refine loop (up to 1 refine round(s)). The `generate` gene drives the draft; the `ensure_correct` gene drives an *intrinsic* critique (the verifier is never consulted mid-loop — it is the scoring metric, so gating on it would be oracle contamination). The final answer is scored programmatically. Evolving the verify gene changes the critique, hence the loop's output — the lever D has and B does not.
- **Budget:** `max_metric_calls`=64 per arm, across 2 seed(s) [0, 1]; estimated spend $4.54.
- **Arms:** A = seed baseline (no optimization); B = prompt-only GEPA (evolves one instruction gene); C = command control (genome-blind worker — the flatness control); D = loop-level GEPA (evolves the whole genome). All arms share an identical seed genome, so held-out gaps isolate what GEPA evolves, not prompt length.

## Results

| Arm | What GEPA evolves | seed held-out | optimized held-out (mean ± std) | Δ |
|-----|-------------------|---------------|---------------------------------|---|
| A | seed baseline (no optimization) | 0.47 | 0.47 ± 0.03 | +0.00 |
| B | prompt-only GEPA (one gene) | 0.47 | 0.59 ± 0.03 | +0.12 |
| C | command control (genome-blind) | 0.47 | 0.47 ± 0.03 | +0.00 |
| D | loop-level GEPA (whole genome) | 0.47 | 0.50 ± 0.06 | +0.03 |

## Verdict — NOT SUPPORTED

D (0.50) did not beat B (0.59) (Δ=-0.09). On this benchmark and budget, loop-level GEPA showed no advantage over prompt-only.

## Caveats

- Arm C's flatness is the genome-blind control; its optimization can't move the score because the command worker ignores the genome.
- The split is small (16 held-out); this claims a *relative* result (loop-level vs prompt-only) under a matched budget, not a leaderboard number.
- Reflection spend is a hard cap: GEPA is told to stop after `reflection_calls` proposals per arm (one reflection LM call each), so the estimated cost is a genuine ceiling, not just a guess.
- Worker completions are hard-capped at 320 tokens (output tokens — the dominant cost — are a genuine ceiling the guard prices exactly). Input tokens are a conservative estimate sized for GEPA-evolved instructions, not an absolute bound, so the $5 guard is enforced pre-flight on that estimate rather than metered mid-run. The IFBench subset used here has only short-output constraints, so the cap does not truncate answers; if it ever did, that floor would apply to every arm equally, not bias the loop.
