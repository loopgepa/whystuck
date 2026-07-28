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


# --- infrastructure faults must NOT masquerade as a genuine 0.0 (paid-run safety) ---


def _code_task(base_tests, plus_test):
    return BenchmarkTask(
        id="synthetic",
        payload={
            "goal": "g",
            "entry_point": "f",
            "base_tests": base_tests,
            "plus_test": plus_test,
            "reference": "def f():\n    return 1\n",
        },
    )


def test_code_score_raises_on_a_data_fault_not_a_silent_zero():
    # A held-out (plus) suite with no assertions is CORRUPT DATA, not a wrong answer.
    # Scoring it 0.0 would silently lower the reported score of a paid run; it must raise.
    from whystuck.eval.code_bench import code_score as cs

    task = _code_task(["assert f() == 1"], "x = 1\n")  # plus block has no assert
    with pytest.raises(RuntimeError, match="did not run|no_assertions"):
        cs("def f():\n    return 1\n", task)


def test_code_score_raises_on_a_sandbox_fault(monkeypatch):
    # A host fault (fork/setrlimit failure) means we never got a real verdict — it must
    # abort loudly, never be folded into "the model scored 0".
    import whystuck.eval.code_bench as cb
    from whystuck.eval.code_exec import ExecResult

    monkeypatch.setattr(
        cb, "run_tests", lambda *a, **k: ExecResult(False, "sandbox error: boom", "sandbox_error")
    )
    with pytest.raises(RuntimeError, match="did not run|sandbox"):
        cb.code_score("def f(): return 1", _first_task())


def test_code_score_still_zero_on_a_genuine_model_failure(monkeypatch):
    # A model that writes an infinite loop (timeout) or exits early (incomplete) is a real
    # failing answer → a legitimate 0.0, NOT an infra fault. These must not raise.
    import whystuck.eval.code_bench as cb
    from whystuck.eval.code_exec import ExecResult

    for verdict in ("timeout", "incomplete", "failed"):
        monkeypatch.setattr(cb, "run_tests", lambda *a, v=verdict, **k: ExecResult(False, v, v))
        assert cb.code_score("def f(): ...", _first_task()) == 0.0


def test_code_feedback_raises_on_a_sandbox_fault(monkeypatch):
    # The in-loop feedback must not tell the critique "a base test failed" when the
    # sandbox itself broke — that would pollute exactly the signal arm D optimizes.
    import whystuck.eval.code_bench as cb
    from whystuck.eval.code_exec import ExecResult

    monkeypatch.setattr(
        cb, "run_tests", lambda *a, **k: ExecResult(False, "sandbox error: boom", "sandbox_error")
    )
    with pytest.raises(RuntimeError, match="did not run|sandbox"):
        cb.code_feedback("def f(): return 1", _first_task())


def test_preflight_passes_on_vendored_data_and_flags_corruption():
    # The pre-spend guard: every vendored reference must pass BOTH suites through the
    # sandbox on THIS host, or the paid run would produce all-zeros. Also proves the
    # executor works here before any spend.
    from whystuck.eval.code_bench import preflight_code_tasks

    good = load_code_tasks("all", seed=0)
    try:
        preflight_code_tasks(good)  # does not raise
    except RuntimeError as exc:
        # `preflight_code_tasks` deliberately distinguishes a slow HOST from bad data, and
        # this assertion is about the DATA. On a loaded machine a vendored reference (in
        # practice mbpp-271) can exceed the wall-clock cap through no fault of the dataset,
        # and failing here would make the suite report a data-integrity problem that does
        # not exist. Skipping on exactly that signal keeps the test deterministic without
        # weakening the guard: the cap still refuses a live run, which is where it matters.
        if "HOST capacity signal" not in str(exc):
            raise
        pytest.skip(f"host too loaded to time the vendored references: {exc}")

    bad = _code_task(["assert f() == 999"], "assert f() == 999\n")  # ref returns 1, fails
    with pytest.raises(RuntimeError, match="preflight|reference"):
        preflight_code_tasks([bad])


def test_preflight_reports_a_reference_timeout_as_a_host_fault_not_corrupt_data(monkeypatch):
    """A reference solution that times out means the HOST was too slow, not bad data.

    One vendored MBPP+ reference legitimately needs several seconds to clear its own plus
    suite, so on a loaded machine it can exceed the wall-clock cap. Reporting that as ``the
    data file is corrupt — re-vendor it'' sends the reader to fix a file that is fine, and
    makes the reproducibility check fail for a reason unrelated to reproducibility. The
    timeout verdict therefore needs its own branch and its own message.
    """
    from whystuck.eval import code_bench
    from whystuck.eval.code_exec import ExecResult

    monkeypatch.setattr(
        code_bench, "run_tests", lambda *a, **k: ExecResult(False, "timeout", "timeout")
    )
    task = BenchmarkTask(
        id="slow-1",
        payload={
            "prompt": "p",
            "entry_point": "f",
            "base_tests": ["assert f() == 1"],
            "plus_test": "assert f() == 1",
            "reference": "def f():\n    return 1",
        },
    )
    with pytest.raises(RuntimeError) as exc:
        code_bench.preflight_code_tasks([task])
    msg = str(exc.value)
    assert "timed out" in msg
    assert "corrupt" not in msg  # must not blame the data file
