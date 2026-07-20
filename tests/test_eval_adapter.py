"""Tests for whystuck.eval.adapter — the loop-genome ↔ GEPA-candidate bridge.

A fake run_task (outcome depends on the candidate text) keeps this offline and
lets us prove a better candidate actually scores higher.
"""

from conftest import FIXED_TS

from whystuck.core.models import GeneType, LoopGene, LoopGenome, LoopMetrics, LoopTrace
from whystuck.eval.adapter import (
    LoopGenomeAdapter,
    candidate_from_genome,
    genome_from_candidate,
)
from whystuck.eval.benchmark import BenchmarkTask


def _genome() -> LoopGenome:
    return LoopGenome(
        id="bugfix",
        name="Bugfix",
        genes=[
            LoopGene(id="skill", type=GeneType.SKILL, name="S", content="Fix the bug."),
            LoopGene(id="retry", type=GeneType.RETRY_POLICY, name="R", content={"max_retries": 1}),
        ],
    )


def _run_task(genome: LoopGenome, task: BenchmarkTask) -> LoopTrace:
    # Outcome depends on the candidate's skill text — success iff it says "minimal".
    skill = next(g for g in genome.genes if g.id == "skill").content
    ok = isinstance(skill, str) and "minimal" in skill
    return LoopTrace(
        id=f"run-{task.id}",
        loop_id="bugfix",
        created_at=FIXED_TS,
        final_outcome="success" if ok else "failure",
        failure_summary=None if ok else "patch was not minimal",
        metrics=LoopMetrics(success=ok),
    )


def test_candidate_extracts_only_text_genes():
    cand = candidate_from_genome(_genome())
    assert cand == {"skill": "Fix the bug."}  # retry (dict) excluded


def test_genome_from_candidate_replaces_without_mutating_base():
    base = _genome()
    new = genome_from_candidate(base, {"skill": "Fix with the smallest/minimal patch."})
    assert next(g for g in new.genes if g.id == "skill").content.endswith("patch.")
    assert next(g for g in base.genes if g.id == "skill").content == "Fix the bug."  # base intact


def test_evaluate_scores_better_candidate_higher():
    adapter = LoopGenomeAdapter(base_genome=_genome(), run_task=_run_task)
    tasks = [BenchmarkTask(id="t1"), BenchmarkTask(id="t2")]
    bad = adapter.evaluate(tasks, {"skill": "Fix the bug."})
    good = adapter.evaluate(tasks, {"skill": "Fix with the minimal patch."})
    assert sum(good.scores) > sum(bad.scores)
    assert len(good.scores) == 2


def test_make_reflective_dataset_uses_failing_feedback():
    adapter = LoopGenomeAdapter(base_genome=_genome(), run_task=_run_task)
    tasks = [BenchmarkTask(id="t1")]
    batch = adapter.evaluate(tasks, {"skill": "Fix the bug."}, capture_traces=True)
    ds = adapter.make_reflective_dataset({"skill": "Fix the bug."}, batch, ["skill"])
    assert "skill" in ds
    assert "minimal" in ds["skill"][0]["Feedback"] or "not minimal" in ds["skill"][0]["Feedback"]
