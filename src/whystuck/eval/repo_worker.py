"""Repo worker — the output→apply→verify wiring that makes optimization bite.

For each task: copy the declared repo state into an isolated temp dir, build a
prompt from the loop's text genes + the goal + the target file, ask the LLM for
the new file content, write it, run the declared verifier, and score the result.
Everything happens in the temp dir — the user's real working tree is never touched.

`complete` (the LLM) and `execute` (the verifier runner) are injected, so this is
testable with a fake LLM + a real local verifier. Production `complete` is
`whystuck.eval.llm_worker.litellm_complete` (network).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from whystuck.core.models import (
    LoopGenome,
    LoopMetrics,
    LoopStepTrace,
    LoopTrace,
    StepStatus,
    outcome_for,
    step_status_for,
)
from whystuck.core.subproc import run_capture, summarize
from whystuck.core.util import utc_now_iso
from whystuck.eval.adapter import RunTask
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.llm_worker import Complete
from whystuck.eval.prompts import render_text_genes

# (command, cwd) -> (exit_code, combined_output)
Execute = Callable[[list[str], Path], tuple[int, str]]


def _default_execute(cmd: list[str], cwd: Path) -> tuple[int, str]:
    return run_capture(cmd, cwd=cwd)


def _default_build_prompt(genome: LoopGenome, goal: str, target: str, before: str) -> str:
    parts = render_text_genes(genome)
    parts.append(f"# Goal\n{goal}")
    if target:
        parts.append(f"# File to rewrite: {target}\nCurrent content:\n{before}")
        parts.append(f"Return ONLY the full new content of {target}.")
    return "\n\n".join(parts)


def make_repo_run_task(
    complete: Complete,
    *,
    execute: Execute = _default_execute,
    build_prompt: Callable[[LoopGenome, str, str, str], str] = _default_build_prompt,
) -> RunTask:
    """Build a run_task(genome, task) that applies an LLM patch to an isolated repo
    copy and scores it with the task's verifier.

    Task payload: {files: {relpath: content}, target: relpath, verifier: [cmd], goal: str}.
    """

    def run_task(genome: LoopGenome, task: BenchmarkTask) -> LoopTrace:
        payload = task.payload or {}
        files: dict[str, str] = payload.get("files", {})
        target: str = payload.get("target", "")
        verifier: list[str] = payload.get("verifier", [])
        goal: str = payload.get("goal", "")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel, content in files.items():
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")

            target_path = root / target if target else None
            before = (
                target_path.read_text(encoding="utf-8")
                if target_path and target_path.exists()
                else ""
            )
            new_content = complete(build_prompt(genome, goal, target, before))
            if target_path is not None:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text(new_content, encoding="utf-8")

            code, output = execute(verifier, root) if verifier else (0, "")

        ok = code == 0
        diff_lines = abs(new_content.count("\n") - before.count("\n"))
        return LoopTrace(
            id=f"repo-{task.id}",
            loop_id=genome.id,
            created_at=utc_now_iso(),
            final_outcome=outcome_for(ok),
            failure_summary=None if ok else summarize(output),
            steps=[
                LoopStepTrace(step_id="worker", role="worker", status=StepStatus.SUCCESS),
                LoopStepTrace(step_id="verify", role="verify", status=step_status_for(ok)),
            ],
            metrics=LoopMetrics(success=ok, diff_size_lines=diff_lines),
        )

    return run_task
