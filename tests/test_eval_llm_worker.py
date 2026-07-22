"""Tests for whystuck.eval.llm_worker — proves text genes move the score."""

import pytest

from whystuck.core.models import GeneType, LoopGene, LoopGenome
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.llm_worker import (
    default_build_prompt,
    keyword_score,
    litellm_complete,
    make_run_task,
)


def _genome(skill: str) -> LoopGenome:
    return LoopGenome(
        id="bugfix",
        name="Bugfix",
        genes=[LoopGene(id="skill", type=GeneType.SKILL, name="S", content=skill)],
    )


# Fake LLM: writes a "minimal" patch only when the prompt asks for one.
def _complete(prompt: str) -> str:
    return "a minimal patch" if "minimal" in prompt.lower() else "a sprawling patch"


def _score(output: str, task: BenchmarkTask) -> bool:
    return "minimal" in output.lower()


def test_better_text_gene_yields_success():
    run_task = make_run_task(_complete, _score)
    good = run_task(_genome("Fix with the minimal patch."), BenchmarkTask(id="t1"))
    bad = run_task(_genome("Fix the bug."), BenchmarkTask(id="t1"))
    assert good.final_outcome == "success"
    assert bad.final_outcome == "failure"


def test_prompt_includes_text_genes_and_goal():
    prompt = default_build_prompt(
        _genome("Prefer minimal diffs."), BenchmarkTask(id="t1", payload={"goal": "fix login"})
    )
    assert "Prefer minimal diffs." in prompt
    assert "fix login" in prompt


def test_keyword_score_matches_declared_expectation():
    task = BenchmarkTask(id="t1", payload={"expect": "Minimal"})
    assert keyword_score("here is a minimal patch", task) is True  # case-insensitive
    assert keyword_score("here is a sprawling patch", task) is False


def test_keyword_score_accepts_any_of_a_list():
    task = BenchmarkTask(id="t1", payload={"expect_any": ["fixed", "resolved"]})
    assert keyword_score("the bug is resolved", task) is True
    assert keyword_score("the bug remains", task) is False


def test_keyword_score_without_expectation_is_not_a_free_win():
    # No declared criterion means nothing was verified — it must not score as a pass.
    assert keyword_score("anything at all", BenchmarkTask(id="t1")) is False


def test_litellm_complete_without_litellm_is_actionable():
    complete = litellm_complete("openai/gpt-4o-mini")
    # The error must name the real fix. litellm is now part of the `whystuck[eval]`
    # extra (added so the one paid run reproduces from a single install), so the
    # extra IS the fix — superseding the earlier state where litellm was separate.
    with pytest.raises(RuntimeError, match=r"whystuck\[eval\]"):
        complete("hi")


# --- reflective-signal capture (fixes GEPA reflection starvation) -------------


def test_run_task_captures_task_input_and_worker_output():
    # The trace must carry the task instruction and the worker's real output so
    # the adapter can build a GEPA reflective dataset with actual content.
    run_task = make_run_task(lambda p: "solid", lambda o, t: 1.0)
    trace = run_task(_genome("skill"), BenchmarkTask(id="t1", payload={"goal": "do the thing"}))
    assert trace.task_summary == "do the thing"
    assert trace.task_id == "t1"
    step = next(s for s in trace.steps if s.role == "worker")
    assert step.output_summary == "solid"
    assert step.input_summary == "do the thing"


def test_run_task_stores_fractional_eval_score_and_partial_is_not_success():
    # A fractional scorer (0.5) must be stored verbatim in metadata['eval_score'],
    # and a partial result is NOT a full success.
    run_task = make_run_task(lambda p: "out", lambda o, t: 0.5)
    trace = run_task(_genome("skill"), BenchmarkTask(id="t"))
    assert trace.metadata["eval_score"] == 0.5
    assert trace.metrics.success is False
    assert trace.final_outcome == "failure"


def test_run_task_full_score_is_success():
    run_task = make_run_task(lambda p: "out", lambda o, t: 1.0)
    trace = run_task(_genome("skill"), BenchmarkTask(id="t"))
    assert trace.metrics.success is True
    assert trace.metadata["eval_score"] == 1.0


def test_run_task_feedback_hook_lands_in_metadata():
    run_task = make_run_task(
        lambda p: "out", lambda o, t: 0.0, feedback=lambda o, t: "constraint X failed"
    )
    trace = run_task(_genome("skill"), BenchmarkTask(id="t"))
    assert trace.metadata["feedback"] == "constraint X failed"
