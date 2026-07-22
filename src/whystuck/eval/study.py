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
(graded fraction) via the LLM worker — for *every* arm, including C. The GEPA
*training* signal differs by arm: B/D use `ifbench_pass` (binary, LLM worker);
arm C's is the command exit status (genome-blind), independent of IFBench.

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
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.ifbench import ifbench_pass, ifbench_score
from whystuck.eval.llm_worker import Complete, default_build_prompt, make_run_task
from whystuck.eval.optimize import optimize_loop

GENERATE_GENE = "generate"
VERIFY_GENE = "ensure_correct"

SEED_GENERATE = "Follow every instruction in the task exactly."
SEED_VERIFY = (
    "Before answering, re-read the task and check each stated constraint is "
    "satisfied; if any is not, revise the answer until it is."
)

# What each arm may evolve. A evolves nothing; B one gene; C and D the whole
# genome (C via the genome-blind command worker, D via the LLM worker).
ARM_EVOLVABLE: dict[str, set[str]] = {
    "A": set(),
    "B": {GENERATE_GENE},
    "C": {GENERATE_GENE, VERIFY_GENE},
    "D": {GENERATE_GENE, VERIFY_GENE},
}
ARMS = ("A", "B", "C", "D")


@dataclass
class ArmResult:
    name: str
    evolvable_genes: tuple[str, ...]
    seed_heldout: float
    opt_heldout: float
    delta: float
    variant_id: str | None


def seed_genome(loop_id: str, *, evolvable: set[str]) -> LoopGenome:
    """The shared 2-gene seed; `evolvable` marks which genes GEPA may change.

    Content is identical across arms — only the evolvable flags differ — so the
    seed prompt (and thus the seed held-out score) is the same for every arm.
    """
    return LoopGenome(
        id=loop_id,
        name="IFBench study loop",
        genes=[
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
        ],
    )


def _heldout_score(genome: LoopGenome, tasks: list[BenchmarkTask], complete: Complete) -> float:
    """Mean graded IFBench score of the genome's LLM-worker output on `tasks`."""
    if not tasks:
        return 0.0
    return mean([ifbench_score(complete(default_build_prompt(genome, t)), t) for t in tasks])


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
    max_metric_calls: int = 150,
    max_candidate_proposals: int | None = None,
    _optimize: Any | None = None,
) -> dict[str, ArmResult]:
    """Run the requested arms and return per-arm held-out results.

    `complete` is the (cassette-wrapped, for the live run) LLM worker;
    `reflection_lm` is passed through to GEPA. `worker_cmd`/`verifier_cmd` are the
    genome-blind commands for arm C. `max_candidate_proposals` caps GEPA proposals
    per arm (one reflection LM call each), making the reflection budget a hard
    bound rather than an estimate. Inject `_optimize` to bypass GEPA offline.
    """
    results: dict[str, ArmResult] = {}
    for arm in arms:
        evolvable = ARM_EVOLVABLE[arm]
        genome = seed_genome(f"study-{arm}", evolvable=evolvable)
        store.save_loop(genome)
        seed_h = _heldout_score(genome, heldout, complete)
        evolvable_genes = tuple(g.id for g in genome.genes if g.evolvable)

        if arm == "A":
            results[arm] = ArmResult(arm, evolvable_genes, seed_h, seed_h, 0.0, None)
            continue

        run_task = (
            _command_run_task(genome.id, worker_cmd, verifier_cmd, max_retries)
            if arm == "C"
            else make_run_task(complete, ifbench_pass)
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
        opt_h = _heldout_score(winner.genome, heldout, complete)
        results[arm] = ArmResult(
            arm, evolvable_genes, seed_h, opt_h, round(opt_h - seed_h, 4), out["variant_id"]
        )
    return results
