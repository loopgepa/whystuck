# Hand-written genes

Each `.txt` here is a complete gene body, injected with
`--seed-gene GENE=FILE` (see `scripts/run_validation_study.py`). The file content
becomes the prompt **verbatim**, so it carries no comments and no header. Rationale
lives here instead.

## Why hand-written genes at all

The study's loop-level arm asks whether an *optimizer* can find a better verify gene.
That question is only meaningful if a better verify gene exists to be found. A
hand-written gene establishes the ceiling for the thing GEPA can actually do, which is
rewrite prose, and it costs a fraction of an optimization run. If a gene written with
full knowledge of the study's own findings does not move held-out score, then a null
from the optimizer says nothing about the optimizer.

## `verify_measure_v2.txt` (the C1 ablation)

Replaces `ensure_correct`. The seed gene is:

> Before answering, re-read the task and check each stated constraint is satisfied; if
> any is not, revise the answer until it is.

Note what that is: **generator-shaped prose in a critique slot**. It says "before
answering" and "revise the answer" to a step whose only job is to return a verdict on a
draft that already exists. The harness appends `Reply exactly PASS if every requirement
is satisfied; otherwise list what to fix.`, which is doing most of the work of making it
a judge at all.

`verify_measure_v2.txt` is written against the three things the study measured about
this detector:

1. **It is a near-random detector (F₁ = 0.11).** So the rubric forces an explicit
   measurement *before* a verdict, per requirement, rather than a global impression.
2. **Both error directions matter and call for opposite fixes.** A false pass ships a
   violation; a false alarm triggers a rewrite that can break an already-satisfied
   requirement. The rubric addresses each separately.
3. **Feedback actionability beat detection quality** (+0.063 text, and the oracle
   detector with terse output *underperformed* a worse detector with instructive
   output). So the rubric requires measured-value / target-value / smallest-edit rather
   than a restatement of the rule.

Deliberately **prose only**. It asks the model to do its own counting; it injects no
measurement tool and never consults the scorer. This is the distinction that makes C1
worth running even though run 4 already showed a tool-fed critique helping (+0.072 on
the baseline): per the paper's revised oracle taxonomy, 165 of 288 constraint instances
are *verdict-equivalent* given `text_stats`, so most of run 4's lift was measurement
substituting for judgement. GEPA cannot add a tool. It can only rewrite prose. C1
therefore measures the ceiling of the search space GEPA actually operates in.

A reasonable objection is that this rubric is built from the study's conclusions and so
is tuned to help. That is the intent: it is a **ceiling**, not a fair sample of what a
naive author would write. The inference it supports is one-directional. If it moves
held-out, an optimizer has something to find and the optimization run is worth paying
for. If it does not, prose-only detector improvement is inert here and no optimizer
result can rescue it.
