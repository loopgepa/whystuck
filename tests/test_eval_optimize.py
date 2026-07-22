"""Tests for whystuck.eval.optimize — GEPA wrapper (real gepa injected/faked)."""

import pytest
from conftest import FIXED_TS

from whystuck.core.models import GeneType, LoopGene, LoopGenome, LoopMetrics, LoopTrace
from whystuck.core.storage import WhyStuckStore
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.optimize import _gepa_optimize, optimize_loop


def _seed_loop(store: WhyStuckStore) -> None:
    store.save_loop(
        LoopGenome(
            id="bugfix",
            name="Bugfix",
            genes=[LoopGene(id="skill", type=GeneType.SKILL, name="S", content="Fix the bug.")],
        )
    )


def _run_task(genome: LoopGenome, task: BenchmarkTask) -> LoopTrace:
    skill = next(g for g in genome.genes if g.id == "skill").content
    ok = isinstance(skill, str) and "minimal" in skill
    return LoopTrace(
        id=f"run-{task.id}",
        loop_id="bugfix",
        created_at=FIXED_TS,
        final_outcome="success" if ok else "failure",
        metrics=LoopMetrics(success=ok),
    )


def test_optimize_saves_best_variant_and_frontier(tmp_path):
    store = WhyStuckStore(tmp_path)
    _seed_loop(store)

    def fake_optimize(**kw):
        # GEPA would explore; here we just hand back the better candidate.
        return {"best_candidate": {"skill": "Fix with the minimal patch."}}

    out = optimize_loop(
        store,
        "bugfix",
        [BenchmarkTask(id="t1"), BenchmarkTask(id="t2")],
        _run_task,
        reflection_lm="fake",
        _optimize=fake_optimize,
    )
    assert out["best_score"] == 1.0  # winning candidate succeeds on both tasks
    variant = store.load_variant("bugfix", out["variant_id"])
    assert "minimal" in next(g for g in variant.genome.genes if g.id == "skill").content
    assert "optimized" in variant.tags
    frontier = store.load_frontier("bugfix")
    assert frontier and frontier[0].variant_id == out["variant_id"]


def test_llm_optimize_landscape_responds_to_genome(tmp_path):
    # BUG 2 guard: the LLM-backed optimize path must score each candidate by
    # running a worker whose prompt is built *from* the genome — so the winning
    # genes actually move the score. (The command path ignored the genome, so
    # every candidate scored identically and GEPA had no gradient.)
    from whystuck.tools import optimize_loop_via_llm

    store = WhyStuckStore(tmp_path)
    _seed_loop(store)
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "t1.yaml").write_text("id: t1\npayload:\n  goal: fix the bug\n")

    # Fake LLM: writes a "minimal" patch only when the genome asked for one.
    def complete(prompt):
        return "a minimal patch" if "minimal" in prompt.lower() else "a sprawling patch"

    def score(output, task):
        return "minimal" in output.lower()

    def pick(candidate):
        return lambda **kw: {"best_candidate": candidate}

    def optimize(winning_skill):
        return optimize_loop_via_llm(
            store,
            "bugfix",
            complete=complete,
            score=score,
            reflection_lm="fake",
            tasks_dir=str(tasks_dir),
            _optimize=pick({"skill": winning_skill}),
        )

    good = optimize("Fix with the minimal patch.")
    bad = optimize("Fix the bug.")
    assert good["best_score"] == 1.0  # genome asked for minimal → worker succeeds
    assert bad["best_score"] == 0.0  # genome did not → same tasks now fail (not flat)


def test_real_optimize_without_gepa_gives_actionable_error():
    # gepa is not in the base install; the real path must fail with guidance.
    with pytest.raises(RuntimeError, match="whystuck\\[eval\\]"):
        _gepa_optimize(
            seed_candidate={}, trainset=[], adapter=None, reflection_lm="x", max_metric_calls=1
        )
