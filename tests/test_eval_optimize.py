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


def test_optimize_loop_forwards_max_candidate_proposals(tmp_path):
    # The reflection-call count is only a *hard* budget bound if GEPA is told to
    # stop after N proposals (one reflection LM call each). optimize_loop must
    # forward that cap to the optimizer so the cost guard's reflection term is
    # enforced, not merely estimated.
    store = WhyStuckStore(tmp_path)
    _seed_loop(store)
    captured: dict = {}

    def fake_optimize(**kw):
        captured.update(kw)
        return {"best_candidate": {"skill": "Fix with the minimal patch."}}

    optimize_loop(
        store,
        "bugfix",
        [BenchmarkTask(id="t1")],
        _run_task,
        reflection_lm="fake",
        max_metric_calls=50,
        max_candidate_proposals=15,
        _optimize=fake_optimize,
    )
    assert captured["max_candidate_proposals"] == 15


def test_optimize_loop_omits_proposal_cap_when_none(tmp_path):
    # Default (None) must not smuggle a spurious cap into the optimizer kwargs.
    store = WhyStuckStore(tmp_path)
    _seed_loop(store)
    captured: dict = {}

    def fake_optimize(**kw):
        captured.update(kw)
        return {"best_candidate": {"skill": "Fix with the minimal patch."}}

    optimize_loop(
        store,
        "bugfix",
        [BenchmarkTask(id="t1")],
        _run_task,
        reflection_lm="fake",
        _optimize=fake_optimize,
    )
    assert "max_candidate_proposals" not in captured


def _fake_gepa(monkeypatch):
    """Install a minimal fake `gepa` (+ the submodules `_gepa_optimize` imports)
    into sys.modules so the REAL `_gepa_optimize` runs with gepa absent from the
    venv. Returns the dict that the fake `gepa.optimize` records its kwargs into,
    so a test can assert on how the proposal cap was wired.
    """
    import sys
    import types

    recorded: dict = {}

    class _Result:
        best_candidate = {"skill": "x"}

    def _optimize(**kw):
        recorded.update(kw)
        return _Result()

    class GEPAAdapter:  # subclassed by _gepa_optimize's _Bridge
        pass

    class EvaluationBatch:
        def __init__(self, outputs=None, scores=None, trajectories=None):
            self.outputs, self.scores, self.trajectories = outputs, scores, trajectories

    class MaxCandidateProposalsStopper:
        def __init__(self, cap):
            self.cap = cap

    gepa = types.ModuleType("gepa")
    gepa.optimize = _optimize  # type: ignore[attr-defined]
    adapter_mod = types.ModuleType("gepa.core.adapter")
    adapter_mod.GEPAAdapter = GEPAAdapter  # type: ignore[attr-defined]
    adapter_mod.EvaluationBatch = EvaluationBatch  # type: ignore[attr-defined]
    stop_mod = types.ModuleType("gepa.utils.stop_condition")
    stop_mod.MaxCandidateProposalsStopper = MaxCandidateProposalsStopper  # type: ignore[attr-defined]

    for name, mod in [
        ("gepa", gepa),
        ("gepa.core", types.ModuleType("gepa.core")),
        ("gepa.core.adapter", adapter_mod),
        ("gepa.utils", types.ModuleType("gepa.utils")),
        ("gepa.utils.stop_condition", stop_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return recorded


class _NullAdapter:
    def evaluate(self, *a, **k): ...
    def make_reflective_dataset(self, *a, **k): ...


def test_gepa_optimize_wires_proposal_cap_as_stopper(monkeypatch):
    # The PR's whole point: an int cap must become a MaxCandidateProposalsStopper
    # under gepa.optimize(stop_callbacks=[...]). Exercise the REAL _gepa_optimize
    # against a faked gepa so a kwarg typo / wrong wrapping / gepa rename fails CI.
    recorded = _fake_gepa(monkeypatch)
    _gepa_optimize(
        seed_candidate={"skill": "s"},
        trainset=[],
        adapter=_NullAdapter(),
        reflection_lm="fake",
        max_metric_calls=5,
        max_candidate_proposals=7,
    )
    stoppers = recorded["stop_callbacks"]
    assert len(stoppers) == 1
    assert stoppers[0].cap == 7  # the cap reached the stopper intact


def test_gepa_optimize_omits_stopper_when_no_cap(monkeypatch):
    # No cap → no stop_callbacks kwarg, so GEPA's own default stopping applies.
    recorded = _fake_gepa(monkeypatch)
    _gepa_optimize(
        seed_candidate={"skill": "s"},
        trainset=[],
        adapter=_NullAdapter(),
        reflection_lm="fake",
        max_metric_calls=5,
    )
    assert "stop_callbacks" not in recorded
