"""LLM-backed worker — the piece that makes text-gene optimization actually bite.

The command+verifier runner's outcome doesn't depend on the loop's prose genes,
so GEPA can't optimize them through it. This worker builds a prompt *from* the
text genes, calls an LLM, and scores the output — so changing a skill/verifier
rubric changes the result, and GEPA has a gradient to climb.

`complete` (the LLM call) and `score` (the task's success check) are injected, so
this is fully offline-testable. The production `complete` is litellm-backed and
lives behind the `eval` extra (network).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from whystuck.core.models import LoopGenome, LoopMetrics, LoopStepTrace, LoopTrace, StepStatus
from whystuck.eval.adapter import RunTask
from whystuck.eval.benchmark import BenchmarkTask

Complete = Callable[[str], str]
Score = Callable[[str, BenchmarkTask], bool]
BuildPrompt = Callable[[LoopGenome, BenchmarkTask], str]


def default_build_prompt(genome: LoopGenome, task: BenchmarkTask) -> str:
    """Assemble a prompt from the genome's text genes + the task goal."""
    parts = [
        f"# {gene.type}: {gene.name}\n{gene.content}"
        for gene in genome.genes
        if isinstance(gene.content, str)
    ]
    payload = task.payload or {}
    goal = payload.get("goal") or payload.get("task_summary") or task.id
    parts.append(f"# Task\n{goal}")
    return "\n\n".join(parts)


def make_run_task(
    complete: Complete, score: Score, *, build_prompt: BuildPrompt = default_build_prompt
) -> RunTask:
    """Build a run_task(genome, task) → LoopTrace that runs the loop via an LLM."""

    def run_task(genome: LoopGenome, task: BenchmarkTask) -> LoopTrace:
        prompt = build_prompt(genome, task)
        output = complete(prompt)
        ok = bool(score(output, task))
        status = StepStatus.SUCCESS if ok else StepStatus.FAILURE
        return LoopTrace(
            id=f"llm-{task.id}",
            loop_id=genome.id,
            created_at=datetime.now(UTC).isoformat(),
            final_outcome="success" if ok else "failure",
            failure_summary=None if ok else "verifier rejected the LLM worker output",
            steps=[LoopStepTrace(step_id="worker", role="worker", status=status)],
            metrics=LoopMetrics(success=ok),
        )

    return run_task


def litellm_complete(model: str) -> Complete:
    """Production `complete` backed by litellm (network; needs the eval extra)."""

    def complete(prompt: str) -> str:
        try:
            import litellm  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "the LLM worker needs the eval extra: pip install whystuck[eval]"
            ) from exc
        resp = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content or ""

    return complete
