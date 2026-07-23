"""Tests for whystuck.eval.code_exec — the sandboxed unit-test runner for v2-code.

This runs model-generated code, so correctness of the pass/fail verdict and the
safety bounds (timeout, memory) are load-bearing: a harness bug would either
mis-score a paid run or hang it. All cases are hand-written (no dataset needed).
"""

from __future__ import annotations

from whystuck.eval.code_exec import run_tests

_GOOD = "def add(a, b):\n    return a + b\n"
_BAD = "def add(a, b):\n    return a - b\n"
_TESTS = "assert add(2, 3) == 5\nassert add(0, 0) == 0\n"


def test_correct_solution_passes():
    r = run_tests(_GOOD, _TESTS, timeout=5.0)
    assert r.passed is True
    assert r.detail == "ok"


def test_wrong_solution_fails_with_a_nonempty_detail():
    r = run_tests(_BAD, _TESTS, timeout=5.0)
    assert r.passed is False
    assert r.detail  # a real failure message, not empty
    assert r.detail != "ok"


def test_timeout_is_a_failure_not_a_hang():
    # An infinite loop must be killed at the timeout and scored as failure, quickly.
    import time

    start = time.monotonic()
    r = run_tests("def f():\n    pass\n", "while True:\n    pass\n", timeout=2.0)
    elapsed = time.monotonic() - start
    assert r.passed is False
    assert "timeout" in r.detail.lower()
    assert elapsed < 10.0  # returned near the timeout, did not hang


def test_syntax_error_is_a_graceful_failure():
    r = run_tests("def f(:\n bad", "assert True\n", timeout=5.0)
    assert r.passed is False
    assert r.detail


def test_runtime_exception_in_tests_is_a_failure():
    r = run_tests(_GOOD, "assert add(2, 2) == 5\n", timeout=5.0)  # 4 != 5
    assert r.passed is False


def test_memory_bomb_is_bounded_not_oom_kill_of_host():
    # A huge allocation must fail inside the sandbox (rlimit), not take down the host.
    r = run_tests("x = [0] * (10**10)\n", "assert True\n", timeout=5.0, mem_mb=256)
    assert r.passed is False


def test_execution_is_deterministic():
    # Same inputs -> same verdict/detail twice (the property --replay relies on).
    a = run_tests(_GOOD, _TESTS, timeout=5.0)
    b = run_tests(_GOOD, _TESTS, timeout=5.0)
    assert (a.passed, a.detail) == (b.passed, b.detail)


def test_solution_and_tests_share_namespace():
    # The realistic MBPP shape: the solution defines a function the tests then call.
    sol = "def is_even(n):\n    return n % 2 == 0\n"
    tests = "assert is_even(4)\nassert not is_even(3)\n"
    assert run_tests(sol, tests, timeout=5.0).passed is True
