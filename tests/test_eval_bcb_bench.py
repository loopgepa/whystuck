"""Tests for whystuck.eval.bcb_bench — the BigCodeBench-Hard loader + Docker scorers.

Skip cleanly without Docker + the whystuck-bcb image + the vendored data file (so CI
without Docker is green). Where available, they execute real solutions in the sandbox.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from whystuck.eval.bcb_bench import _DATA
from whystuck.eval.benchmark import BenchmarkTask


def _docker_ok() -> bool:
    env = {
        **os.environ,
        "DOCKER_HOST": os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock"),
    }
    try:
        for args in (["docker", "info"], ["docker", "image", "inspect", "whystuck-bcb:latest"]):
            if subprocess.run(
                args, capture_output=True, timeout=15, env=env, check=False
            ).returncode:
                return False
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not (_docker_ok() and _DATA.exists()),
    reason="needs Docker + whystuck-bcb image + vendored bigcodebench_hard.json",
)


def _first():
    from whystuck.eval.bcb_bench import load_bcb_tasks

    return load_bcb_tasks("train", n=1, seed=0)[0]


def test_load_split_deterministic_and_disjoint():
    from whystuck.eval.bcb_bench import load_bcb_tasks

    tr1 = load_bcb_tasks("train", n=4, seed=0)
    tr2 = load_bcb_tasks("train", n=4, seed=0)
    ho = load_bcb_tasks("heldout", n=4, seed=0)
    assert [t.id for t in tr1] == [t.id for t in tr2]
    assert not (set(t.id for t in tr1) & set(t.id for t in ho))


def test_goal_shows_spec_but_not_hidden_tests_or_reference():
    task = _first()
    goal = task.payload["goal"]
    assert task.payload["entry_point"] in goal
    # the unittest source (visible AND hidden) and the reference must never reach the prompt
    assert task.payload["test"] not in goal
    assert task.payload["reference"] not in goal
    for m in task.payload["hidden_methods"]:
        # the hidden methods' source (their `def`) must never leak into the worker prompt
        assert "def " + m not in goal


def test_reference_scores_one_on_hidden_methods():
    from whystuck.eval.bcb_bench import bcb_score

    task = _first()
    assert bcb_score(task.payload["reference"], task) == 1.0


def test_reference_wrapped_in_markdown_scores_one():
    from whystuck.eval.bcb_bench import bcb_score

    task = _first()
    assert bcb_score(f"```python\n{task.payload['reference']}\n```", task) == 1.0


def test_empty_solution_scores_zero():
    from whystuck.eval.bcb_bench import bcb_score

    assert bcb_score("", _first()) == 0.0


def test_feedback_is_visible_based_and_never_leaks_hidden_source():
    from whystuck.eval.bcb_bench import bcb_feedback

    task = _first()
    fb = bcb_feedback("def task_func(*a, **k):\n    return None\n", task)
    assert fb
    assert task.payload["test"] not in fb  # never dumps the whole suite


def test_score_raises_on_sandbox_fault_not_silent_zero(monkeypatch):
    import whystuck.eval.bcb_bench as bb
    from whystuck.eval.code_exec import ExecResult

    monkeypatch.setattr(
        bb,
        "run_unittest",
        lambda *a, **k: ExecResult(False, "sandbox error: boom", "sandbox_error"),
    )
    with pytest.raises(RuntimeError, match="did not run|sandbox"):
        bb.bcb_score("def task_func(): return 1", _first())


def test_score_zero_on_genuine_model_failure(monkeypatch):
    import whystuck.eval.bcb_bench as bb
    from whystuck.eval.code_exec import ExecResult

    for verdict in ("timeout", "incomplete", "failed"):
        monkeypatch.setattr(bb, "run_unittest", lambda *a, v=verdict, **k: ExecResult(False, v, v))
        assert bb.bcb_score("def task_func(): ...", _first()) == 0.0


def test_preflight_passes_on_vendored_data_and_flags_corruption():
    from whystuck.eval.bcb_bench import load_bcb_tasks, preflight_bcb_tasks

    preflight_bcb_tasks(load_bcb_tasks("all", seed=0)[:3])  # sample keeps it quick; must not raise

    bad = BenchmarkTask(
        id="synthetic",
        payload={
            "goal": "g",
            "entry_point": "task_func",
            "test": (
                "import unittest\nclass TestCases(unittest.TestCase):\n"
                "    def test_a(self):\n        self.assertEqual(task_func(), 1)\n"
                "    def test_b(self):\n        self.assertEqual(task_func(), 1)\n"
                "    def test_c(self):\n        self.assertEqual(task_func(), 1)\n"
                "    def test_d(self):\n        self.assertEqual(task_func(), 1)\n"
            ),
            "visible_methods": ["test_a", "test_b"],
            "hidden_methods": ["test_c", "test_d"],
            "reference": "def task_func():\n    return 999\n",  # fails its own tests
        },
    )
    with pytest.raises(RuntimeError, match="preflight|reference"):
        preflight_bcb_tasks([bad])
