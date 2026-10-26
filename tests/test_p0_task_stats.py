"""Per-task calibration statistics, read back from recorded arm-A cassettes.

`select_levels` builds a p0 level out of two numbers per task: the share of first drafts that
were already correct (`p_first`) and the mean final score (`final`). Both come from replaying
the seed-genome loop over the calibration cassettes. The tests pin three things a wrong
extractor would get wrong without failing: the FIRST draft is the one scored for p_first (not
the refined one), every salted replicate is replayed under its own salt (an unsalted replay
misses every call), and a task whose calls are absent raises instead of dropping out of the
pool, which would shrink the sample and bias it toward whatever did get recorded.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

from whystuck.eval.cassette import Cassette, CassetteMiss
from whystuck.eval.llm_worker import make_self_refine_run_task
from whystuck.eval.multi_constraint import _pairs, build_witness, mc_feedback, mc_score
from whystuck.eval.study import seed_genome
from whystuck.eval.task_manifest import (
    build_manifest,
    calibration_levels,
    calibration_pool,
    load_manifest,
    save_manifest,
)

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "p0_task_stats.py"


def _mod():
    spec = importlib.util.spec_from_file_location("p0_task_stats", _SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _record(cassette_path, tasks, sample, behaviour):
    """Record the seed-genome loop over `tasks` with a fake worker.

    behaviour(task, sample) -> (first_draft, critique, refined)."""
    by_goal = {t.payload["goal"]: t for t in tasks}

    def fake(prompt: str, **kw) -> str:
        assert kw.get("sample") == sample
        task = next(t for g, t in by_goal.items() if g in prompt)
        first, critique, refined = behaviour(task, sample)
        if prompt.startswith(seed_genome("x", evolvable=set()).genes[1].content):
            return critique
        if "# Your previous answer" in prompt:
            return refined
        return first

    with Cassette(cassette_path, mode="record") as cas:
        wrapped = cas.wrap(fake, name="worker")
        rt = make_self_refine_run_task(
            lambda p: wrapped(p, sample=sample), mc_score, feedback=mc_feedback, max_retries=1
        )
        g = seed_genome("study-A", evolvable=set())
        for t in tasks:
            rt(g, t)


def _setup(tmp_path, n=6):
    tasks = calibration_pool(n, seed=1000)
    m = build_manifest(
        {"n": n, "seed": 1000}, tasks, calibration_levels([t.id for t in tasks], chunk=3), stats={}
    )
    save_manifest(tmp_path / "m.json", m)
    return tasks, load_manifest(tmp_path / "m.json")


def test_first_draft_and_final_are_read_per_task_and_per_salt(tmp_path):
    tasks, manifest = _setup(tmp_path)
    good = {t.id: build_witness(_pairs(t)) for t in tasks}
    right_first = {tasks[0].id, tasks[1].id}

    def behaviour(task, sample):
        # tasks[0]: right first time on both salts. tasks[1]: right first only on salt 1.
        # Everyone else: wrong first, repaired to the witness on refine.
        first_ok = task.id == tasks[0].id or (task.id == tasks[1].id and sample == 1)
        first = good[task.id] if first_ok else "nope"
        return first, ("PASS" if first_ok else "fix it"), good[task.id]

    for k in (0, 1):
        _record(tmp_path / f"c{k}.json", tasks, k, behaviour)
    stats = _mod().task_stats(
        manifest, "calib", seeds=[0, 1], n_heldout=3,
        runs=[(str(tmp_path / "c0.json"), 0), (str(tmp_path / "c1.json"), 1)],
    )  # fmt: skip
    assert set(stats) == {t.id for t in tasks}
    assert stats[tasks[0].id]["p_first"] == 1.0
    assert stats[tasks[1].id]["p_first"] == 0.5
    for t in tasks[2:]:
        assert stats[t.id]["p_first"] == 0.0
        assert stats[t.id]["final"] == 1.0  # repaired
    assert all(s["k"] == 2 for s in stats.values())
    assert right_first <= {i for i, s in stats.items() if s["p_first"] > 0}


def test_final_is_the_score_after_refinement_not_the_first_draft(tmp_path):
    tasks, manifest = _setup(tmp_path, n=3)

    def behaviour(task, sample):
        return (
            build_witness(_pairs(task)),
            "fix it anyway",
            "broken",
        )  # critique rejects a good draft

    _record(tmp_path / "c.json", tasks, 0, behaviour)
    stats = _mod().task_stats(
        manifest, "calib", seeds=[0], n_heldout=3, runs=[(str(tmp_path / "c.json"), 0)]
    )
    for t in tasks:
        assert stats[t.id]["p_first"] == 1.0
        assert stats[t.id]["final"] == pytest.approx(mc_score("broken", t))
        assert stats[t.id]["final"] < 1.0


def test_a_missing_task_raises_rather_than_shrinking_the_pool(tmp_path):
    tasks, manifest = _setup(tmp_path)
    _record(
        tmp_path / "c.json", tasks[:3], 0, lambda t, s: ("x", "fix", "y")
    )  # chunk 1 never recorded
    with pytest.raises(CassetteMiss):
        _mod().task_stats(
            manifest, "calib", seeds=[0, 1], n_heldout=3, runs=[(str(tmp_path / "c.json"), 0)]
        )


def test_replaying_under_the_wrong_salt_misses(tmp_path):
    tasks, manifest = _setup(tmp_path, n=3)
    _record(tmp_path / "c.json", tasks, 0, lambda t, s: ("x", "fix", "y"))
    with pytest.raises(CassetteMiss):
        _mod().task_stats(
            manifest, "calib", seeds=[0], n_heldout=3, runs=[(str(tmp_path / "c.json"), 1)]
        )
