"""Tests for whystuck.eval.code_exec — the sandboxed unit-test runner for v2-code.

This runs model-generated code, so correctness of the pass/fail verdict and the
safety bounds (timeout, memory) are load-bearing: a harness bug would either
mis-score a paid run or hang it. All cases are hand-written (no dataset needed).
"""

from __future__ import annotations

import contextlib
import os
import tempfile

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
    # loop lives in the SOLUTION (the test block must carry a real assert, else it is
    # short-circuited as "no assertions" before anything runs).
    r = run_tests("while True:\n    pass\n", "assert True\n", timeout=2.0)
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


# --- false-PASS defenses (a solution under optimization pressure must not reward-hack) ---


def test_sys_exit_zero_before_tests_is_not_a_pass():
    # A stub that exits 0 before the (failing) tests run must NOT score as passed.
    r = run_tests("import sys\nsys.exit(0)\n", "assert False\n", timeout=5.0)
    assert r.passed is False


def test_os_exit_zero_before_tests_is_not_a_pass():
    r = run_tests("import os\nos._exit(0)\n", "assert 1 == 2\n", timeout=5.0)
    assert r.passed is False


def test_raise_systemexit_zero_is_not_a_pass():
    r = run_tests("raise SystemExit(0)\n", "assert 1 == 2\n", timeout=5.0)
    assert r.passed is False


def test_unterminated_string_cannot_swallow_the_test_block():
    # Solution opens a triple-quoted string; because tests are a SEPARATE compiled unit,
    # they still run (and here fail), rather than being swallowed as a string literal.
    sol = 'def add(a, b):\n    return a - b\n\n_x = """\n'
    tests = "assert add(2, 3) == 5\n"
    assert run_tests(sol, tests, timeout=5.0).passed is False


def test_empty_test_block_is_not_a_vacuous_pass():
    r = run_tests("def add(a, b):\n    return a - b\n", "", timeout=5.0)
    assert r.passed is False
    assert "assert" in r.detail.lower()


def test_comment_only_test_block_is_not_a_vacuous_pass():
    r = run_tests("def add(a, b):\n    return a - b\n", "# TODO real tests\n", timeout=5.0)
    assert r.passed is False


def test_numpy_imports_under_the_sandbox():
    # The MBPP+ plus-test harness imports numpy unconditionally; the sandbox must let
    # it load (thread-pinned so OpenBLAS's virtual reservation stays under RLIMIT_AS)
    # or every plus-test would fail with an import error, not a real verdict.
    sol = "import numpy as np\n\ndef f():\n    return np.array([1, 2, 3]).sum()\n"
    tests = "assert f() == 6\n"
    r = run_tests(sol, tests, timeout=15.0)
    assert r.passed is True, r.detail


def test_blas_threads_are_pinned_to_one_for_determinism():
    # Single-thread BLAS is what makes numpy reductions bit-for-bit reproducible under
    # --replay (and keeps its virtual memory small). Assert the pins reach the child.
    sol = "import os\n"
    tests = (
        'assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"\n'
        'assert os.environ.get("OMP_NUM_THREADS") == "1"\n'
        'assert os.environ.get("MKL_NUM_THREADS") == "1"\n'
    )
    r = run_tests(sol, tests, timeout=5.0)
    assert r.passed is True, r.detail


def test_memory_bomb_still_bounded_with_the_raised_limit():
    # Raising the default RLIMIT_AS to admit numpy must NOT defang the bomb guard: a
    # 10**10-element list (~80 GB) still dwarfs any sane limit and must fail, not OOM.
    r = run_tests("x = [0] * (10**10)\n", "assert True\n", timeout=5.0)
    assert r.passed is False


def test_timeout_kills_spawned_descendants_no_leak():
    # A solution that spawns a detached grandchild then hangs: after the timeout, the
    # grandchild's process group must be killed (not leaked). We verify by PID liveness.
    import time

    marker = os.path.join(tempfile.gettempdir(), f"ws_orphan_{os.getpid()}.pid")
    if os.path.exists(marker):
        os.remove(marker)
    # Build the grandchild source via %r reprs (no manual quote nesting): it records its
    # PID to `marker`, then sleeps well past the timeout.
    grandchild = f"import os, time\nopen({marker!r}, 'w').write(str(os.getpid()))\ntime.sleep(30)\n"
    sol = (
        f"import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        f"while True:\n    time.sleep(0.1)\n"
    )
    r = run_tests(sol, "assert True\n", timeout=2.0)
    assert r.passed is False and "timeout" in r.detail.lower()
    # give the kill a moment, then confirm the grandchild is gone
    deadline = time.monotonic() + 4.0
    child_pid = None
    while time.monotonic() < deadline:
        if os.path.exists(marker):
            with open(marker) as fh:
                child_pid = int(fh.read() or "0")
            break
        time.sleep(0.05)
    if child_pid:
        alive = True
        for _ in range(40):
            try:
                os.kill(child_pid, 0)
                time.sleep(0.1)
            except ProcessLookupError:
                alive = False
                break
        with contextlib.suppress(OSError):
            os.remove(marker)
        assert not alive, f"grandchild {child_pid} leaked past the timeout"
