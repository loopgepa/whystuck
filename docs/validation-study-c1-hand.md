# WhyStuck validation study: loop-level vs prompt-only GEPA

*Generated: 2026-07-28T16:33:42.506543+00:00.*

## Method

- **Dataset:** multi-constraint-hard (IFEval-style) — 32 train / 32 held-out tasks, scored by programmatic constraint verifiers (no LLM judge).
- **Worker (task LM):** `openai/gpt-4.1-mini`; **reflection LM:** `openai/gpt-5.6-terra`.
- **Worker loop:** each task eval runs a generate→critique→refine self-refine loop (up to 1 refine round(s)). The `generate` gene drives the draft; the `ensure_correct` gene drives an *intrinsic* critique (the verifier is never consulted mid-loop — it is the scoring metric, so gating on it would be oracle contamination). The final answer is scored programmatically. Evolving the verify gene changes the critique, hence the loop's output — the lever D has and B does not.
- **Budget:** `max_metric_calls`=64 per arm, across 1 seed(s) [0]; estimated spend $0.10.
- **Arms:** A = seed baseline (no optimization); B = prompt-only GEPA (evolves one instruction gene); C = fixed-draft control (a constant draft graded by the real metric, so the genome is genuinely ignored but the landscape is not saturated); D = loop-level GEPA (evolves the whole genome). All arms share an identical seed genome, so held-out gaps isolate what GEPA evolves, not prompt length.
- **Hand-written seed gene(s):** `ensure_correct` replaced with 'You are checking a draft against the task\'s explicit requirements. Do not rewrite the draft.\n\nWork through the requirements one at a time, and for each one state the measurement before the verdict:\n1. Quote the requirement.\n2. Measure the draft against it, counting explicitly whenever the requirement is numeric. For a word count, number the words and give the total. For a maximum word length, name the longest word and its letter count. For a required bullet count, count the lines that begin with a bullet marker. For a required ending, quote the draft\'s final words. For a forbidden or required word, scan for it and say where it occurs.\n3. Compare the measurement against the target and say pass or fail for that requirement.\n\nJudge only what the task actually demands. Do not invent requirements, and do not fail a draft over style, tone, or length that no requirement mentions: a false alarm costs a rewrite, and a rewrite can break a requirement the draft already satisfied.\n\nWhen a requirement fails, give the exact repair rather than a restatement of the rule: the measured value, the target value, and the smallest edit that closes the gap. For example, "27 words, target exactly 25, so delete two words from the third bullet" rather than "the word count is wrong".'. This measures whether a BETTER gene moves held-out at all — prior to asking whether an optimizer can find one.

## Results

| Arm | What GEPA evolves | seed held-out | optimized held-out (mean ± std) | Δ |
|-----|-------------------|---------------|---------------------------------|---|
| A | seed baseline (no optimization) | 0.76 | 0.76 ± 0.00 | +0.00 |
| B | prompt-only GEPA (one gene) | — | Pending | — |
| C | fixed-draft control (genome-blind) | — | Pending | — |
| D | loop-level GEPA (whole genome) | — | Pending | — |

## Verdict — INCONCLUSIVE

Arms B and D were not both run, so the thesis can't be evaluated.

## Caveats

- Arm C's flatness is the genome-blind control; its optimization can't move the score because the fixed-draft worker ignores the genome.
- The split is small (32 held-out); this claims a *relative* result (loop-level vs prompt-only) under a matched budget, not a leaderboard number.
- Reflection spend is a hard cap: GEPA is told to stop after `reflection_calls` proposals per arm (one reflection LM call each), so the estimated cost is a genuine ceiling, not just a guess.
- Worker completions are hard-capped at 320 tokens (output tokens — the dominant cost — are a genuine ceiling the guard prices exactly). Input tokens are a conservative estimate sized for GEPA-evolved instructions, not an absolute bound, so the guard is enforced pre-flight on that estimate rather than metered mid-run. The instruction-following constraints used here have short outputs, so the cap does not truncate answers; if it ever did, that floor would apply to every arm equally, not bias the loop.
