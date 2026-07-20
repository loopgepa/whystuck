"""Tests for whystuck.eval.repo_worker — the output→apply→verify wiring.

Proves end-to-end: the text genes shape the LLM patch, the patch is applied to an
isolated repo copy, a REAL verifier subprocess runs, and the score reflects it.
Uses a fake `complete` (deterministic) + a real python verifier (offline).
"""

from pathlib import Path

from whystuck.core.models import GeneType, LoopGene, LoopGenome
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.repo_worker import make_repo_run_task

_VERIFIER = [
    "python3",
    "-c",
    "import sys; sys.exit(0 if open('answer.txt').read().strip()=='42' else 1)",
]


def _genome(skill: str) -> LoopGenome:
    return LoopGenome(
        id="fix",
        name="Fix",
        genes=[LoopGene(id="skill", type=GeneType.SKILL, name="S", content=skill)],
    )


def _task() -> BenchmarkTask:
    return BenchmarkTask(
        id="t1",
        payload={
            "files": {"answer.txt": "0"},
            "target": "answer.txt",
            "verifier": _VERIFIER,
            "goal": "make answer.txt say 42",
        },
    )


# Fake LLM: writes the correct value only when the prompt asks to be correct.
def _complete(prompt: str) -> str:
    return "42" if "correct" in prompt.lower() else "0"


def test_good_genes_apply_a_fix_that_passes_the_verifier():
    run_task = make_repo_run_task(_complete)
    trace = run_task(_genome("Give the correct value."), _task())
    assert trace.final_outcome == "success"
    assert trace.metrics.success is True


def test_bad_genes_fail_the_verifier():
    run_task = make_repo_run_task(_complete)
    trace = run_task(_genome("Just guess."), _task())
    assert trace.final_outcome == "failure"
    assert trace.failure_summary is not None


def test_does_not_touch_the_real_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_task = make_repo_run_task(_complete)
    run_task(_genome("Give the correct value."), _task())
    assert list(tmp_path.iterdir()) == []  # optimization ran in an isolated temp repo


def test_missing_verifier_scores_on_apply_only():
    run_task = make_repo_run_task(_complete)
    task = BenchmarkTask(id="t2", payload={"files": {}, "target": "answer.txt", "goal": "x"})
    trace = run_task(_genome("Give the correct value."), task)
    assert trace.final_outcome == "success"  # no verifier → apply succeeded
