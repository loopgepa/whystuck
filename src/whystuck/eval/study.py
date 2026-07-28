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
  command worker (the control; its landscape is flat, so GEPA can't improve it).
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

from whystuck.core.models import GeneType, LoopGene, LoopGenome, LoopTrace
from whystuck.core.runner import run_command_loop
from whystuck.core.storage import WhyStuckStore
from whystuck.core.util import mean, utc_now_iso
from whystuck.eval.adapter import RunTask
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
    D evolve the whole genome (C via the genome-blind command worker, D via the LLM
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
    name: str
    evolvable_genes: tuple[str, ...]
    seed_heldout: float
    opt_heldout: float
    delta: float
    variant_id: str | None


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


def _command_run_task(
    loop_id: str, worker: tuple[str, ...], verifier: tuple[str, ...], max_retries: int
) -> Callable[[LoopGenome, BenchmarkTask], LoopTrace]:
    """A genome-blind run_task (arm C): the command output ignores the genome."""

    def run_task(_genome: LoopGenome, task: BenchmarkTask) -> LoopTrace:
        return run_command_loop(
            loop_id=loop_id,
            trace_id=f"opt-{task.id}",
            created_at=utc_now_iso(),
            worker=list(worker),
            verifier=list(verifier),
            max_retries=max_retries,
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
    worker_cmd: tuple[str, ...] = ("true",),
    verifier_cmd: tuple[str, ...] = ("true",),
    max_retries: int = 1,
    loop_max_retries: int = 2,
    max_metric_calls: int = 150,
    max_candidate_proposals: int | None = None,
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
    passed through to GEPA. `worker_cmd`/`verifier_cmd` are the genome-blind
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
            results[arm] = ArmResult(arm, evolvable_genes, seed_h, seed_h, 0.0, None)
            continue

        run_task = (
            _command_run_task(genome.id, worker_cmd, verifier_cmd, max_retries)
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
            _optimize=_optimize,
        )
        winner = store.load_variant(genome.id, out["variant_id"])
        opt_h = _heldout_score(winner.genome, heldout, worker_run_task)
        results[arm] = ArmResult(
            arm, evolvable_genes, seed_h, opt_h, round(opt_h - seed_h, 4), out["variant_id"]
        )
    return results
