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
Score = Callable[[str, BenchmarkTask], bool]
BuildPrompt = Callable[[LoopGenome, BenchmarkTask], str]


def default_build_prompt(genome: LoopGenome, task: BenchmarkTask) -> str:
    """Assemble a prompt from the genome's text genes + the task goal."""
    parts = render_text_genes(genome)
    payload = task.payload or {}
    goal = payload.get("goal") or payload.get("task_summary") or task.id
    parts.append(f"# Task\n{goal}")
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
    complete: Complete, score: Score, *, build_prompt: BuildPrompt = default_build_prompt
) -> RunTask:
    """Build a run_task(genome, task) → LoopTrace that runs the loop via an LLM."""

    def run_task(genome: LoopGenome, task: BenchmarkTask) -> LoopTrace:
        prompt = build_prompt(genome, task)
        output = complete(prompt)
        ok = bool(score(output, task))
        return LoopTrace(
            id=f"llm-{task.id}",
            loop_id=genome.id,
            created_at=utc_now_iso(),
            final_outcome=outcome_for(ok),
            failure_summary=None if ok else "verifier rejected the LLM worker output",
            steps=[LoopStepTrace(step_id="worker", role="worker", status=step_status_for(ok))],
            metrics=LoopMetrics(success=ok),
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
