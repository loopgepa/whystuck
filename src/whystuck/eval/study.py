"""The 4-arm GEPA validation study (see docs/superpowers/specs/2026-07-22-...).

Tests one claim: **loop-level GEPA (optimizing the whole loop genome) beats
prompt-only GEPA (optimizing a single instruction) under a matched budget.**

Four arms share an *identical* 2-gene seed genome — a `generate` skill gene and
an `ensure_correct` verifier-rubric gene (mirroring GEPA's 2-module IFBench
agent). They differ only in what GEPA may evolve, so any held-out gap isolates
componentized optimization from a "more prompt text" confound:

- **A** — evolves nothing (seed baseline; no optimization).
- **B** — evolves the single `generate` gene (prompt-only GEPA).
- **C** — evolves the whole genome but scores each candidate with a genome-blind
  constant draft graded by the real metric (the control). Its landscape is flat because
  the worker ignores the genome, yet it is NOT saturated, so GEPA still searches and a
  spurious gain would be visible. An earlier command-exit-status control scored a perfect
  1.0, which made the arm incapable of failing.
- **D** — evolves the whole genome with the LLM worker (loop-level GEPA).

Each arm's winning genome is scored on the held-out split with `ifbench_score`
(graded fraction) via the LLM worker — for *every* arm, including C. B/D train on
that *same* fractional `ifbench_score` (aligning the training objective with the
reported metric) plus `ifbench_feedback` (per-constraint prose GEPA reflects on);
arm C's training signal is the command exit status (genome-blind), independent of
IFBench.

The study drives the low-level `optimize_loop` seam directly (not the MCP
wrappers) so it can pass explicit train tasks and inject `_optimize` for offline
tests. Requires `whystuck[eval]` + real models only for the live run; with
`_optimize` injected it runs fully offline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from whystuck.core.models import (
    GeneType,
    LoopGene,
    LoopGenome,
    LoopMetrics,
    LoopStepTrace,
    LoopTrace,
    outcome_for,
)
from whystuck.core.storage import WhyStuckStore
from whystuck.core.util import mean, utc_now_iso
from whystuck.eval.adapter import RunTask, VerifyTruth
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.ifbench import ifbench_feedback, ifbench_score
from whystuck.eval.llm_worker import Complete, Feedback, Score, make_self_refine_run_task
from whystuck.eval.optimize import optimize_loop

GENERATE_GENE = "generate"
VERIFY_GENE = "ensure_correct"
PLAN_GENE = "plan"  # optional third module (complexity lever); off unless use_plan_gene
ACTUATOR_GENE = "actuator"  # optional fixer-feedback gene (actuator experiment); off by default

SEED_GENERATE = "Follow every instruction in the task exactly."
SEED_VERIFY = (
    "Before answering, re-read the task and check each stated constraint is "
    "satisfied; if any is not, revise the answer until it is."
)
SEED_PLAN = (
    "Before answering, list each explicit constraint in the task as a short "
    "checklist the answer must satisfy."
)
SEED_ACTUATOR = (
    "The draft failed a check. Using the test results, write concrete instructions "
    "for how to fix the draft so it passes."
)
ARMS = ("A", "B", "C", "D")


def arm_evolvable(arm: str, *, include_plan: bool = False, actuator: bool = False) -> set[str]:
    """Which genes GEPA may evolve in each arm.

    A evolves nothing; B one gene (prompt-only — stays `generate` even when a third
    gene is present, so the B/D contrast still isolates loop-level optimization); C and
    D evolve the whole genome (C via the genome-blind constant draft, D via the LLM
    worker). `include_plan` widens the whole genome to the optional `plan` gene.

    `actuator` selects the actuator experiment's genome: the whole genome is
    {generate, actuator} and the detector (`ensure_correct`) is deliberately held
    fixed — the experiment asks whether GEPA helps when pointed at the fixer's
    feedback rather than the detector. `actuator` and `include_plan` are mutually
    exclusive (distinct experiments)."""
    if actuator:
        whole = {GENERATE_GENE, ACTUATOR_GENE}
    else:
        whole = {GENERATE_GENE, VERIFY_GENE} | ({PLAN_GENE} if include_plan else set())
    return {"A": set(), "B": {GENERATE_GENE}, "C": set(whole), "D": set(whole)}[arm]


# Back-compat: the two-gene evolvable map (no plan gene), keyed by arm.
ARM_EVOLVABLE: dict[str, set[str]] = {a: arm_evolvable(a) for a in ARMS}


@dataclass
class ArmResult:
    """One arm's outcome, including the provenance needed to interpret it.

    The last three fields exist because held-out scores alone cannot distinguish
    "the extra evolvable gene was optimized and did not help" from "the optimizer
    never accepted a change to it, so the treatment was never applied". Reporting
    `genes_changed` makes that checkable by a reader; `train_score` makes the
    train-split over-fitting explanation falsifiable (it was previously computed and
    discarded); `final_genes` releases the text that actually produced the score.
    """

    name: str
    evolvable_genes: tuple[str, ...]
    seed_heldout: float
    opt_heldout: float
    delta: float
    variant_id: str | None
    train_score: float | None = None
    genes_changed: tuple[str, ...] = ()
    final_genes: dict[str, str] | None = None


def seed_genome(
    loop_id: str,
    *,
    evolvable: set[str],
    include_plan: bool = False,
    include_actuator: bool = False,
) -> LoopGenome:
    """The shared seed genome; `evolvable` marks which genes GEPA may change.

    Content is identical across arms — only the evolvable flags differ — so the
    seed prompt (and thus the seed held-out score) is the same for every arm.
    Two genes by default; `include_plan` adds the optional third `plan` gene so the
    loop runs plan→generate→critique→refine (the complexity lever). `include_actuator`
    instead adds the `actuator` gene, an LLM step that turns the test signal into the
    repair guidance fed to refine (the actuator experiment).
    """
    genes = [
        LoopGene(
            id=GENERATE_GENE,
            type=GeneType.SKILL,
            name="Generate",
            content=SEED_GENERATE,
            evolvable=GENERATE_GENE in evolvable,
        ),
        LoopGene(
            id=VERIFY_GENE,
            type=GeneType.VERIFIER_RUBRIC,
            name="Ensure correct",
            content=SEED_VERIFY,
            evolvable=VERIFY_GENE in evolvable,
        ),
    ]
    if include_plan:
        genes.append(
            LoopGene(
                id=PLAN_GENE,
                type=GeneType.PROMPT,
                name="Plan",
                content=SEED_PLAN,
                evolvable=PLAN_GENE in evolvable,
            )
        )
    if include_actuator:
        genes.append(
            LoopGene(
                id=ACTUATOR_GENE,
                type=GeneType.PROMPT,
                name="Actuator",
                content=SEED_ACTUATOR,
                evolvable=ACTUATOR_GENE in evolvable,
            )
        )
    return LoopGenome(id=loop_id, name="IFBench study loop", genes=genes)


def _heldout_score(genome: LoopGenome, tasks: list[BenchmarkTask], run_task: RunTask) -> float:
    """Mean graded IFBench score of the genome on `tasks`, via the SAME loop worker
    used for training — so held-out and training measure the identical thing (the
    loop's final programmatic score), with no single-shot/loop mismatch.

    Raises if a trace lacks ``metadata['eval_score']`` OR if ``tasks`` is empty: a
    silent 0.0 in either case would report a misconfigured run (e.g. n_heldout=0, or
    a run_task that doesn't record the score) as a catastrophic all-fail genome and
    bury it in the paid report, rather than surfacing the misconfiguration."""
    if not tasks:
        raise ValueError(
            f"_heldout_score called with an empty task list for genome {genome.id!r}; "
            "returning 0.0 would masquerade as a genuine all-fail — check n_heldout / "
            "pool sizing upstream"
        )
    scores: list[float] = []
    for t in tasks:
        trace = run_task(genome, t)
        raw = trace.metadata.get("eval_score")
        if raw is None:
            raise ValueError(
                f"trace {trace.id!r} has no metadata['eval_score']; run_task is misconfigured "
                "(the self-refine worker must record the programmatic score there)"
            )
        scores.append(float(raw))
    return mean(scores)


#: The constant answer the fixed-draft control emits. Deliberately plausible-but-wrong:
#: it satisfies no counting or formatting constraint, so the real metric scores it well
#: below 1.0 and the control's landscape is flat WITHOUT being saturated.
CONTROL_FIXED_DRAFT = "placeholder answer"


def _fixed_draft_run_task(
    loop_id: str, score_fn: Score, feedback_fn: Feedback, draft: str = CONTROL_FIXED_DRAFT
) -> Callable[[LoopGenome, BenchmarkTask], LoopTrace]:
    """A genome-blind run_task scored by the REAL metric (the honest arm-C control).

    The command control it replaces always exits 0, so `scalarize` hands GEPA a perfect
    1.0 and the search terminates immediately with no proposals — flat because it is
    maxed out, which cannot detect a harness that credits a gain to an ignored genome.
    Emitting a fixed imperfect draft and scoring it programmatically keeps the landscape
    flat (the genome is genuinely ignored) while leaving real headroom, so a spurious
    gain would be visible. Costs nothing: no model call, no container.
    """

    def run_task(_genome: LoopGenome, task: BenchmarkTask) -> LoopTrace:
        eval_score = float(score_fn(draft, task))
        ok = eval_score >= 1.0
        fb = feedback_fn(draft, task)
        return LoopTrace(
            id=f"control-{task.id}",
            loop_id=loop_id,
            created_at=utc_now_iso(),
            task_id=task.id,
            task_summary=str(task.payload.get("goal", task.id)),
            final_outcome=outcome_for(ok),
            failure_summary=None if ok else fb,
            steps=[
                LoopStepTrace(step_id="generate-0", role="worker", output_summary=draft),
            ],
            metrics=LoopMetrics(success=ok),
            metadata={"eval_score": eval_score, "feedback": fb},
        )

    return run_task


def run_study(
    store: WhyStuckStore,
    *,
    complete: Complete,
    reflection_lm: Any,
    train: list[BenchmarkTask],
    heldout: list[BenchmarkTask],
    arms: tuple[str, ...] = ARMS,
    loop_max_retries: int = 2,
    max_metric_calls: int = 150,
    max_candidate_proposals: int | None = None,
    module_selector: str = "round_robin",
    verify_truth: VerifyTruth | None = None,
    score_fn: Score = ifbench_score,
    feedback_fn: Feedback = ifbench_feedback,
    use_plan_gene: bool = False,
    use_actuator: bool = False,
    stats_fn: Callable[[str], str] | None = None,
    exec_feedback: Feedback | None = None,
    critique_complete: Complete | None = None,
    oracle_critique: bool = False,
    oracle_feedback: Feedback | None = None,
    actuator_context: Feedback | None = None,
    _optimize: Any | None = None,
) -> dict[str, ArmResult]:
    """Run the requested arms and return per-arm held-out results.

    `complete` is the (cassette-wrapped, for the live run) LLM worker, driven as a
    generate→critique→refine loop (`loop_max_retries` rounds); `reflection_lm` is
    passed through to GEPA. Arm C is the genome-blind control (a constant draft
    scored by the real metric).
    commands for arm C. `max_candidate_proposals` caps GEPA proposals per arm (one
    reflection LM call each), making the reflection budget a hard bound rather than
    an estimate. `score_fn`/`feedback_fn` select the task set's scorer/feedback
    (ifbench single-constraint by default; pass the multi_constraint pair for the
    composable multi-constraint study). Inject `_optimize` to bypass GEPA offline.
    """
    # One loop worker for BOTH training (B/D) and held-out scoring (all arms), so
    # the two measure the identical thing. The critique step is driven by the
    # evolvable verify gene, which is what lets arm D differ from arm B.
    worker_run_task = make_self_refine_run_task(
        complete,
        score_fn,
        feedback=feedback_fn,
        max_retries=loop_max_retries,
        plan_gene=PLAN_GENE if use_plan_gene else None,
        stats_fn=stats_fn,
        exec_feedback=exec_feedback,
        critique_complete=critique_complete,
        oracle_critique=oracle_critique,
        oracle_feedback=oracle_feedback,
        actuator_gene=ACTUATOR_GENE if use_actuator else None,
        actuator_context=actuator_context,
    )
    results: dict[str, ArmResult] = {}
    for arm in arms:
        evolvable = arm_evolvable(arm, include_plan=use_plan_gene, actuator=use_actuator)
        genome = seed_genome(
            f"study-{arm}",
            evolvable=evolvable,
            include_plan=use_plan_gene,
            include_actuator=use_actuator,
        )
        store.save_loop(genome)
        seed_h = _heldout_score(genome, heldout, worker_run_task)
        evolvable_genes = tuple(g.id for g in genome.genes if g.evolvable)

        if arm == "A":
            # Arm A scores held-out a SECOND time rather than copying its first result.
            # Copying hardcoded delta = 0.0, which made "a non-flat A invalidates the run"
            # unfalsifiable. The two measurements agree whenever the path is deterministic
            # (under cassette replay, by construction), so this is a determinism check —
            # it is NOT the flatness control the earlier write-up claimed.
            opt_a = _heldout_score(genome, heldout, worker_run_task)
            results[arm] = ArmResult(
                arm, evolvable_genes, seed_h, opt_a, round(opt_a - seed_h, 4), None
            )
            continue

        run_task = (
            _fixed_draft_run_task(genome.id, score_fn, feedback_fn)
            if arm == "C"
            else worker_run_task
        )
        out = optimize_loop(
            store,
            genome.id,
            train,
            run_task,
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            max_candidate_proposals=max_candidate_proposals,
            module_selector=module_selector,
            # Component-specific reflection for the critique gene. Only meaningful for
            # arms that actually evolve it, and only when a truth signal is supplied;
            # otherwise the adapter keeps its original component-agnostic records.
            verify_gene=VERIFY_GENE if verify_truth is not None else None,
            verify_truth=verify_truth,
            _optimize=_optimize,
        )
        winner = store.load_variant(genome.id, out["variant_id"])
        opt_h = _heldout_score(winner.genome, heldout, worker_run_task)
        # Provenance: which genes the optimizer actually changed, and their final text.
        # Compared against THIS arm's own seed genome (identical content across arms),
        # so an empty tuple means "the optimizer accepted nothing" — the difference
        # between a treatment that failed and a treatment that never ran.
        seed_text = {g.id: g.content for g in genome.genes}
        changed = tuple(
            g.id
            for g in winner.genome.genes
            if isinstance(g.content, str) and g.content != seed_text.get(g.id)
        )
        final_genes = {
            g.id: g.content
            for g in winner.genome.genes
            if g.evolvable and isinstance(g.content, str)
        }
        results[arm] = ArmResult(
            arm,
            evolvable_genes,
            seed_h,
            opt_h,
            round(opt_h - seed_h, 4),
            out["variant_id"],
            train_score=out.get("best_score"),
            genes_changed=changed,
            final_genes=final_genes,
        )
    return results
