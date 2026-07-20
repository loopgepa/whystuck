"""Bridge a WhyStuck loop genome onto GEPA's candidate/evaluation interface.

GEPA evolves a `candidate: dict[str, str]` of named text components. Here those
are the loop's evolvable **text** genes (skill/agent/prompt/verifier rubric). The
runner-backed dict genes (retry/stop/budget) are not text-optimized in this alpha.

`evaluate` runs each benchmark task through an injected `run_task` (the loop
executor) and scores the resulting trace — so this module is fully offline-testable
and never imports `gepa`. `optimize.py` adapts it to the real GEPA optimizer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from whystuck.core.models import LoopGenome, LoopTrace
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.objectives import objective_vector, scalarize

RunTask = Callable[[LoopGenome, BenchmarkTask], LoopTrace]


@dataclass
class EvalBatch:
    outputs: list[str]
    scores: list[float]
    trajectories: list[LoopTrace] | None = None


def candidate_from_genome(genome: LoopGenome) -> dict[str, str]:
    """The evolvable text genes as GEPA's candidate."""
    return {g.id: g.content for g in genome.genes if g.evolvable and isinstance(g.content, str)}


def genome_from_candidate(base_genome: LoopGenome, candidate: dict[str, str]) -> LoopGenome:
    """A copy of the genome with its text genes replaced from the candidate."""
    new = base_genome.model_copy(deep=True)
    for gene in new.genes:
        if gene.id in candidate and isinstance(gene.content, str):
            gene.content = candidate[gene.id]
    return new


class LoopGenomeAdapter:
    def __init__(self, base_genome: LoopGenome, run_task: RunTask) -> None:
        self.base_genome = base_genome
        self.run_task = run_task

    def evaluate(
        self, batch: list[BenchmarkTask], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvalBatch:
        genome = genome_from_candidate(self.base_genome, candidate)
        outputs: list[str] = []
        scores: list[float] = []
        traces: list[LoopTrace] = []
        for task in batch:
            trace = self.run_task(genome, task)
            outputs.append(trace.final_outcome)
            scores.append(scalarize(trace.metrics))
            traces.append(trace)
        return EvalBatch(
            outputs=outputs, scores=scores, trajectories=traces if capture_traces else None
        )

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvalBatch, components_to_update: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Turn each task's outcome into GEPA reflection records, per component."""
        records: list[dict[str, Any]] = []
        for i, trace in enumerate(eval_batch.trajectories or []):
            feedback = f"score={eval_batch.scores[i]:.2f}; " + (
                trace.failure_summary or "succeeded"
            )
            records.append(
                {
                    "Inputs": {"outcome": trace.final_outcome},
                    "Generated Outputs": objective_vector(trace.metrics),
                    "Feedback": feedback,
                }
            )
        return {component: records for component in components_to_update}
