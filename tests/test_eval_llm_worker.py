"""Tests for whystuck.eval.llm_worker — proves text genes move the score."""

import pytest

from whystuck.core.models import GeneType, LoopGene, LoopGenome
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.llm_worker import default_build_prompt, litellm_complete, make_run_task


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


def test_litellm_complete_without_litellm_is_actionable():
    complete = litellm_complete("openai/gpt-4o-mini")
    # The error must name the real fix — `pip install litellm`, not the eval extra,
    # which never included litellm (review finding #5).
    with pytest.raises(RuntimeError, match="pip install litellm"):
        complete("hi")
