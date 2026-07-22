# WhyStuck validation study: loop-level vs prompt-only GEPA

*Generated: pending (offline build; step 7 fills this in).*

## Method

- **Dataset:** IFBench — 31 train / 31 held-out tasks, scored by programmatic constraint verifiers (no LLM judge).
- **Worker (task LM):** `openai/gpt-4.1-mini`; **reflection LM:** `openai/gpt-5.6-terra`.
- **Budget:** `max_metric_calls`=150 per arm, across 2 seed(s) [0, 1]; estimated spend $3.43.
- **Arms:** A = seed baseline (no optimization); B = prompt-only GEPA (evolves one instruction gene); C = command control (genome-blind worker — the flatness control); D = loop-level GEPA (evolves the whole genome). All arms share an identical seed genome, so held-out gaps isolate what GEPA evolves, not prompt length.

## Results

**Pending live run (step 7).** The harness, scoring, and budget guard are built and tested offline; this document fills in once the one paid run records its cassettes.

| Arm | What GEPA evolves | seed held-out | optimized held-out (mean ± std) | Δ |
|-----|-------------------|---------------|---------------------------------|---|
| A | seed baseline (no optimization) | — | Pending | — |
| B | prompt-only GEPA (one gene) | — | Pending | — |
| C | command control (genome-blind) | — | Pending | — |
| D | loop-level GEPA (whole genome) | — | Pending | — |

## Verdict

Pending the live run.

## Caveats

- Arm C's flatness is the genome-blind control; its optimization can't move the score because the command worker ignores the genome.
- The split is small (31 held-out); this claims a *relative* result (loop-level vs prompt-only) under a matched budget, not a leaderboard number.
- Reflection spend is a hard cap: GEPA is told to stop after `reflection_calls` proposals per arm (one reflection LM call each), so the estimated cost is a genuine ceiling, not just a guess.
