"""Tests for whystuck.eval.code_bench — the vendored MBPP+ loader + scorers.

These execute real code through the sandbox, so they need numpy in the interpreter
(the dev group installs it). The vendored data is committed, so no network is used.
"""

from __future__ import annotations

import pytest

from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.code_bench import (
    code_feedback,
    code_score,
    extract_code,
    load_code_tasks,
    vendored_tasks,
)

# --- extract_code ---------------------------------------------------------------


def test_extract_code_pulls_a_fenced_python_block():
    text = "Here is my solution:\n```python\ndef f():\n    return 1\n```\nHope it helps!"
    code = extract_code(text)
    assert "def f():" in code
    assert "Hope it helps" not in code
    assert "```" not in code


def test_extract_code_returns_raw_text_when_unfenced():
    text = "def f():\n    return 1\n"
    assert extract_code(text).strip() == text.strip()


def test_extract_code_joins_multiple_blocks():
    # A model that puts imports in one block and the function in another: both survive.
    text = "```python\nimport math\n```\nand\n```python\ndef f():\n    return math.pi\n```"
    code = extract_code(text)
    assert "import math" in code and "def f():" in code


# --- loader ---------------------------------------------------------------------


def test_vendored_tasks_nonempty_and_shaped():
    tasks = vendored_tasks()
    assert len(tasks) >= 32
    t = tasks[0]
    for key in ("prompt", "entry_point", "base_tests", "plus_test", "reference"):
        assert key in t and t[key], key


def test_load_split_is_deterministic_and_disjoint():
    tr1 = load_code_tasks("train", n=8, seed=0)
    tr2 = load_code_tasks("train", n=8, seed=0)
    ho = load_code_tasks("heldout", n=8, seed=0)
    assert [t.id for t in tr1] == [t.id for t in tr2]  # deterministic
    assert not (set(t.id for t in tr1) & set(t.id for t in ho))  # disjoint splits


def test_load_raises_rather_than_silently_truncating():
    with pytest.raises(ValueError, match="only"):
        load_code_tasks("train", n=10_000, seed=0)


def test_goal_shows_entry_point_and_base_tests_but_not_plus_or_reference():
    task = load_code_tasks("train", n=1, seed=0)[0]
    goal = task.payload["goal"]
    assert task.payload["entry_point"] in goal  # worker is told the function name
    assert task.payload["base_tests"][0] in goal  # base tests are VISIBLE (in-loop)
    # the hidden scorer and canonical solution must never reach the worker prompt
    assert task.payload["plus_test"] not in goal
    assert task.payload["reference"] not in goal


# --- scoring / feedback (against real vendored tasks) ---------------------------


def _first_task() -> BenchmarkTask:
    return load_code_tasks("train", n=1, seed=0)[0]


def test_reference_solution_scores_one_on_plus_tests():
    task = _first_task()
    assert code_score(task.payload["reference"], task) == 1.0


def test_reference_wrapped_in_markdown_still_scores_one():
    task = _first_task()
    wrapped = f"Sure!\n```python\n{task.payload['reference']}\n```"
    assert code_score(wrapped, task) == 1.0


def test_empty_or_wrong_solution_scores_zero():
    task = _first_task()
    assert code_score("", task) == 0.0
    # a solution that defines the right name but returns a constant fails the plus tests
    wrong = f"def {task.payload['entry_point']}(*a, **k):\n    return None\n"
    assert code_score(wrong, task) == 0.0


def test_code_feedback_is_base_based_and_never_leaks_plus():
    task = _first_task()
    wrong = f"def {task.payload['entry_point']}(*a, **k):\n    return None\n"
    fb = code_feedback(wrong, task)
    assert fb  # non-empty actionable prose
    assert task.payload["plus_test"] not in fb  # feedback is the VISIBLE base signal only


def test_code_feedback_passes_for_the_reference():
    task = _first_task()
    fb = code_feedback(task.payload["reference"], task)
    assert "pass" in fb.lower()
