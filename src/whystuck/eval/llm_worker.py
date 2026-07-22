"""LLM-backed worker — the piece that makes text-gene optimization actually bite.

The command+verifier runner's outcome doesn't depend on the loop's prose genes,
so GEPA can't optimize them through it. This worker builds a prompt *from* the
text genes, calls an LLM, and scores the output — so changing a skill/verifier
rubric changes the result, and GEPA has a gradient to climb.

`complete` (the LLM call) and `score` (the task's success check) are injected, so
this is fully offline-testable. The production `complete` is litellm-backed
(network); litellm ships with the `whystuck[eval]` extra, not the base install.
"""

from __future__ import annotations

from collections.abc import Callable

from whystuck.core.models import (
    LoopGenome,
    LoopMetrics,
    LoopStepTrace,
    LoopTrace,
    outcome_for,
    step_status_for,
)
from whystuck.core.util import utc_now_iso
from whystuck.eval.adapter import RunTask
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.prompts import render_text_genes

Complete = Callable[[str], str]
# A task scorer: fraction of requirements satisfied ∈ [0, 1]. A bool scorer still
# works (True/False coerce to 1.0/0.0) — the fractional form just gives GEPA a
# smoother gradient on partially-satisfied tasks.
Score = Callable[[str, BenchmarkTask], float]
# Optional per-task feedback prose for GEPA's reflective dataset (e.g. which
# constraints passed/failed). Without it, a generic success/failure message is used.
Feedback = Callable[[str, BenchmarkTask], str]
BuildPrompt = Callable[[LoopGenome, BenchmarkTask], str]


def _task_text(task: BenchmarkTask) -> str:
    """The task's instruction, as shown to the worker and to GEPA's reflector."""
    payload = task.payload or {}
    return str(payload.get("goal") or payload.get("task_summary") or task.id)


def default_build_prompt(genome: LoopGenome, task: BenchmarkTask) -> str:
    """Assemble a prompt from the genome's text genes + the task goal."""
    parts = render_text_genes(genome)
    parts.append(f"# Task\n{_task_text(task)}")
    return "\n\n".join(parts)


def keyword_score(output: str, task: BenchmarkTask) -> bool:
    """Default ``Score``: the output must contain the task's declared expectation.

    ``payload.expect`` is a substring, or ``payload.expect_any`` a list of them;
    matching is case-insensitive. With no expectation declared nothing can be
    verified, so the task scores ``False`` — an unscored task must not count as a
    free win, or it would flatten GEPA's gradient for that task."""
    payload = task.payload or {}
    expect = payload.get("expect")
    candidates = payload.get("expect_any") or ([expect] if expect else [])
    low = output.lower()
    return any(isinstance(e, str) and e.lower() in low for e in candidates)


def make_run_task(
    complete: Complete,
    score: Score,
    *,
    feedback: Feedback | None = None,
    build_prompt: BuildPrompt = default_build_prompt,
) -> RunTask:
    """Build a run_task(genome, task) → LoopTrace that runs the loop via an LLM.

    The trace carries the real reflective signal GEPA needs to optimize: the task
    instruction (`task_summary` + the worker step's `input_summary`), the worker's
    actual output text (`output_summary`), the fractional score
    (`metadata['eval_score']`, which the adapter maximizes), and per-task feedback
    (`metadata['feedback']`). Without these, GEPA's reflector is blind to the task
    and cannot propose targeted improvements.
    """

    def run_task(genome: LoopGenome, task: BenchmarkTask) -> LoopTrace:
        prompt = build_prompt(genome, task)
        output = complete(prompt)
        eval_score = float(score(output, task))
        ok = eval_score >= 1.0  # "success" == every requirement satisfied
        task_text = _task_text(task)
        fb = (
            feedback(output, task)
            if feedback is not None
            else ("all requirements satisfied" if ok else "verifier rejected the worker output")
        )
        return LoopTrace(
            id=f"llm-{task.id}",
            loop_id=genome.id,
            created_at=utc_now_iso(),
            task_id=task.id,
            task_summary=task_text,
            final_outcome=outcome_for(ok),
            failure_summary=None if ok else fb,
            steps=[
                LoopStepTrace(
                    step_id="worker",
                    role="worker",
                    status=step_status_for(ok),
                    input_summary=task_text,
                    output_summary=output,
                )
            ],
            metrics=LoopMetrics(success=ok),
            metadata={"eval_score": eval_score, "feedback": fb},
        )

    return run_task


def litellm_complete(model: str) -> Complete:
    """Production `complete` backed by litellm (network; ships with `whystuck[eval]`)."""

    def complete(prompt: str) -> str:
        try:
            import litellm  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "the LLM worker needs litellm: pip install 'whystuck[eval]'"
            ) from exc
        resp = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content or ""

    return complete
