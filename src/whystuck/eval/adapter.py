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
from whystuck.eval.objectives import scalarize

RunTask = Callable[[LoopGenome, BenchmarkTask], LoopTrace]


def _score_of(trace: LoopTrace) -> float:
    """The score GEPA maximizes. Prefer a fractional ``metadata['eval_score']``
    (the LLM worker records constraint-satisfaction fraction there); fall back to
    the success-dominant ``scalarize`` for command-loop traces that don't set it,
    so v0.1 scoring is unchanged."""
    raw = trace.metadata.get("eval_score")
    return float(raw) if raw is not None else scalarize(trace.metrics)


def _worker_output(trace: LoopTrace) -> str:
    """The worker's *final* output text, for reflection.

    For a self-refine loop this is the last generate/refine step (role "worker"),
    NOT the initial draft and NOT a "verifier" critique step; for the single-shot
    worker there is exactly one such step. Falls back to any last step with output
    text, then to the outcome string (command-loop traces keep output on ToolCall,
    not the step, so they land on the outcome).

    An empty string is a *legitimate* worker answer (a content-filter refusal or an
    empty completion), so we return it as-is rather than masking it with the outcome
    word — a fabricated "success"/"failure" fed to GEPA's reflector would be worse
    than an honest empty output. The `is not None` filter (not truthiness) is the
    same discipline `_score_of` uses to keep a real 0.0 from looking 'unset'."""
    for step in reversed(trace.steps):
        if step.role == "worker" and step.output_summary is not None:
            return step.output_summary  # narrowed to str; "" is kept, not masked
    for step in reversed(trace.steps):
        if step.output_summary is not None:
            return step.output_summary
    return trace.final_outcome


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
            outputs.append(_worker_output(trace))
            scores.append(_score_of(trace))
            traces.append(trace)
        return EvalBatch(
            outputs=outputs, scores=scores, trajectories=traces if capture_traces else None
        )

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvalBatch, components_to_update: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Build GEPA reflection records with the real task signal.

        Each record surfaces what the reflector needs to propose a better prompt:
        the task instruction (Inputs), the worker's actual output (Generated
        Outputs), and per-task Feedback (which requirements passed/failed). GEPA cannot
        improve a component it can't see the effect of, so these must be the real
        task/output, not internal status codes."""
        records: list[dict[str, Any]] = []
        traj = eval_batch.trajectories or []
        for i, trace in enumerate(traj):
            score = eval_batch.scores[i] if i < len(eval_batch.scores) else _score_of(trace)
            # Presence check, not truthiness: an explicit "" feedback from the callable
            # is a real (if empty) signal and must not be overwritten by the synth string.
            raw_fb = trace.metadata.get("feedback")
            feedback = (
                raw_fb
                if raw_fb is not None
                else f"score={score:.2f}; " + (trace.failure_summary or "succeeded")
            )
            records.append(
                {
                    "Inputs": trace.task_summary or trace.task_id or "",
                    "Generated Outputs": _worker_output(trace),
                    "Feedback": feedback,
                }
            )
        return {component: records for component in components_to_update}
