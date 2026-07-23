# GEPA validation study — results note (all three runs)

*Question:* does **loop-level** GEPA (evolve the self-refine loop's `generate` **and**
`ensure_correct` verify genes) beat **prompt-only** GEPA (evolve `generate` alone), on
instruction-following tasks, under a matched budget?

*Answer across three pre-registered paid runs:* **no measurable advantage.**

## The three runs

All runs share one design: 4 arms on an identical seed genome — A = no optimization,
B = prompt-only (evolve `generate`), C = genome-blind command control (flatness check),
D = loop-level (evolve `generate` + `ensure_correct`). The worker is a
generate→critique→refine loop; the critique is *intrinsic* (the constraint verifier is
the scoring metric and is never consulted mid-loop — gating on it would be oracle
contamination). Only what GEPA is allowed to evolve differs between B and D, so the
held-out gap Δ = mean(D) − mean(B) isolates the loop-level lever, not prompt length.
Worker `gpt-4.1-mini`, reflection `gpt-5.6-terra`.

| Run | Task regime | seed held-out | B (prompt-only) | D (loop-level) | Δ = D − B | Verdict | Why |
|-----|-------------|---------------|-----------------|----------------|-----------|---------|-----|
| 1 | single-constraint (IFBench slice) | 0.47 | 0.59 | 0.50 | **−0.09** | NOT SUPPORTED | **uninformative** — one constraint ⇒ nothing to decompose ⇒ verify gene inert |
| 2 | easy multi-constraint (k = 3–4) | **0.97** | 0.97 | 0.97 | **0.00** | NOT SUPPORTED | **uninformative** — seed saturated, no headroom (ceiling flag) |
| 3 | **hard multi-constraint** (k = 4–5, exact-count + brutal) | **0.765** | 0.770 | 0.770 | **0.00** | NOT SUPPORTED | **the fair test** — real headroom, no ceiling/floor flag |

Run 3 is the one that counts. Seed held-out ≈ 0.765 sits inside the pre-registered band
0.2 ≤ seed ≤ 0.85, so neither the ceiling nor floor flag fires — the loop had genuine
room to help and the verify gene had multiple independent constraints to decompose.

The run-3 tie is not a dead line: per-seed, D and B trade wins (seed 1: D +0.05, B −0.005;
seed 4: B +0.055, D flat) and net to Δ = 0.000. Control C is flat on every seed, which
confirms the harness is sound — a real signal *could* have shown up and didn't.

## How this fits the literature

- The result is exactly what the self-correction literature predicts. Huang et al.,
  [*LLMs Cannot Self-Correct Reasoning Yet*](https://arxiv.org/abs/2310.01798) (ICLR 2024),
  find that **intrinsic** self-correction — refining with no external feedback signal —
  fails to reliably improve outputs and can degrade them. Arm D's verify gene is precisely
  an intrinsic critique (no oracle). A null is the expected outcome, not a surprise.
- [Self-Refine](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)
  (Madaan et al., NeurIPS 2023) reports ~20 % gains from self-feedback, but on open-ended
  generation quality, not on tasks with an *external verifiable* metric where the model
  cannot see its own score. Our tasks are the hard case for self-refinement.
- [GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026 oral) claims reflective prompt
  evolution beats RL — a claim about **prompt** evolution vs policy gradients, *not* about
  loop-level vs prompt-level. Our null does not contradict GEPA; it answers a question GEPA
  does not.
- DSPy pipeline-optimization work reports that optimizing multiple interacting modules pays
  off **as pipeline complexity grows**. Our loop is minimal: 2 genes, 1 refine round. That
  is the low-complexity end where the literature would least expect a loop-level edge.

## Bottom line

Within scope — one worker model, one small budget (max_metric_calls = 28, 1 refine round),
one task family — **loop-level GEPA gives no edge over prompt-only GEPA on verifiable
instruction-following.** This is a clean, publishable negative result, and it agrees with
the strongest prior on intrinsic self-correction. The burden now shifts: the interesting
question is no longer "does the loop help?" (default: no) but "under what conditions
*would* it?"

## Next step

Two honest options, in order of recommendation:

1. **Ship the null (recommended).** Three pre-registered runs converging on "no", the
   third with genuine headroom and aligned with Huang et al., is a complete finding. Publish
   this note + the three reports; stop spending. Cost to do this: $0.

2. **One more pre-registered run testing the two literature-implied levers, if the question
   is worth more money.** Change exactly one thing at a time:
   - **Feedback richness (Huang lever):** give the critique a *non-oracle* external signal it
     is allowed to see — e.g. a deterministic partial-credit hint ("your draft has 23 words,
     the rule wants an exact count") that is not the final metric. This is the single change
     most likely to flip the result, because it converts intrinsic self-correction (which
     fails) into feedback-guided correction (which works).
   - **Pipeline complexity (DSPy lever):** raise loop depth (≥ 2–3 refine rounds) and/or add a
     third gene, so there is more interacting structure for loop-level evolution to exploit.

   Do **not** vary budget, model, and task family all at once — that is p-hacking a win.
   Pre-register the single lever, the expected seed band, and the verdict rule before spending.

*Recommendation: option 1.* The fair test has been run and answered; further spend should
require a specific hypothesis about *why* this run would differ, and the feedback-richness
lever is the only one with strong prior support.
