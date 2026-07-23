"""Tests for whystuck.eval.bcb_exec — the Docker-sandboxed unittest runner (v2 code).

These require a working Docker daemon + the built ``whystuck-bcb`` image; they skip
cleanly where neither is available (CI without Docker). Correctness of the pass/fail
verdict and the safety bounds (offline, memory, timeout) is load-bearing: this scores
untrusted model code and gates a paid run.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from whystuck.eval.bcb_exec import run_unittest


def _docker_ok() -> bool:
    env = {
        **os.environ,
        "DOCKER_HOST": os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock"),
    }
    try:
        info = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=15, env=env, check=False
        )
        if info.returncode != 0:
            return False
        img = subprocess.run(
            ["docker", "image", "inspect", "whystuck-bcb:latest"],
            capture_output=True,
            timeout=15,
            env=env,
            check=False,
        )
        return img.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_ok(), reason="needs a working Docker daemon + the whystuck-bcb image"
)

_SOL = "import numpy as np\ndef task_func(x):\n    return int(np.sum(x))\n"
_TEST = (
    "import unittest\n"
    "class TestCases(unittest.TestCase):\n"
    "    def test_a(self):\n        self.assertEqual(task_func([1, 2, 3]), 6)\n"
    "    def test_b(self):\n        self.assertEqual(task_func([]), 0)\n"
    "    def test_c(self):\n        self.assertEqual(task_func([5]), 5)\n"
    "    def test_d(self):\n        self.assertEqual(task_func([-1, 1]), 0)\n"
)


def test_correct_solution_passes_all_methods():
    r = run_unittest(_SOL, _TEST, timeout=40)
    assert r.passed is True and r.verdict == "pass"


def test_method_subset_is_honored():
    # The anti-oracle split runs only a chosen subset; a subset the solution satisfies passes.
    r = run_unittest(_SOL, _TEST, ["test_a", "test_b"], timeout=40)
    assert r.passed is True


def test_wrong_solution_fails_with_detail():
    bad = "def task_func(x):\n    return 999\n"
    r = run_unittest(bad, _TEST, timeout=40)
    assert r.passed is False and r.verdict == "failed"
    assert "999" in r.detail  # a real assertion message, not empty


def test_failing_subset_is_a_failure_not_a_pass():
    bad = "def task_func(x):\n    return 999\n"
    r = run_unittest(bad, _TEST, ["test_c"], timeout=40)  # 999 != 5
    assert r.passed is False and r.verdict == "failed"


def test_no_runnable_methods_is_no_assertions_not_a_vacuous_pass():
    empty = "import unittest\nclass TestCases(unittest.TestCase):\n    pass\n"
    r = run_unittest(_SOL, empty, timeout=40)
    assert r.passed is False and r.verdict == "no_assertions"


def test_selecting_a_nonexistent_method_is_not_a_vacuous_pass():
    # If the split names a method the suite doesn't have, we must not report a pass on
    # zero tests run — it collapses to no_assertions, never a silent success.
    r = run_unittest(_SOL, _TEST, ["test_does_not_exist"], timeout=40)
    assert r.passed is False and r.verdict == "no_assertions"


def test_import_error_in_solution_is_a_graceful_failure():
    r = run_unittest("import a_module_that_does_not_exist\n", _TEST, timeout=40)
    assert r.passed is False and r.verdict == "failed"


def test_timeout_is_a_failure_not_a_hang():
    import time

    start = time.monotonic()
    loop = "def task_func(x):\n    while True:\n        pass\n"
    r = run_unittest(loop, _TEST, ["test_a"], timeout=8)
    assert r.passed is False and r.verdict == "timeout"
    assert time.monotonic() - start < 40  # returned near the timeout, container killed


def test_network_is_blocked_offline_enforced():
    # --network none must make a real outbound connection fail (proves offline is ENFORCED,
    # not merely unused). A task_func that opens a socket errors → the test errors → failed.
    net_sol = (
        "import socket\n"
        "def task_func(x):\n"
        "    socket.create_connection(('8.8.8.8', 53), timeout=3)\n"
        "    return int(sum(x))\n"
    )
    r = run_unittest(net_sol, _TEST, ["test_a"], timeout=40)
    assert r.passed is False  # cannot reach the network inside the sandbox


def test_execution_is_deterministic():
    a = run_unittest(_SOL, _TEST, timeout=40)
    b = run_unittest(_SOL, _TEST, timeout=40)
    assert (a.passed, a.verdict) == (b.passed, b.verdict)


def test_solution_cannot_monkeypatch_unittest_into_a_fake_pass():
    # A solution under optimization pressure must not grade itself: patching
    # unittest.TestResult.wasSuccessful (or assertEqual) must NOT yield a pass, because the
    # harness reloads unittest after the solution runs and scores from the result lists.
    cheat = (
        "import unittest\n"
        "unittest.TestResult.wasSuccessful = lambda self: True\n"
        "def task_func(x):\n    raise RuntimeError('broken')\n"
    )
    r = run_unittest(cheat, _TEST, ["test_a", "test_b"], timeout=40)
    assert r.passed is False and r.verdict == "failed"


def test_skipped_tests_do_not_count_as_a_pass():
    # A @skip'd test verified nothing; a suite of only-skips must not score a pass.
    skipping = (
        "import unittest\n"
        "class TestCases(unittest.TestCase):\n"
        "    @unittest.skip('x')\n"
        "    def test_a(self):\n        self.assertEqual(task_func([2]), 4)\n"
        "    @unittest.skip('x')\n"
        "    def test_b(self):\n        self.assertEqual(task_func([3]), 9)\n"
    )
    r = run_unittest("def task_func(x):\n    return None\n", skipping, timeout=40)
    assert r.passed is False


def test_daemon_connection_error_is_sandbox_error_not_zero(monkeypatch):
    # A mid-run daemon blip: docker CLI exits 1 with a connection message. That must be a
    # sandbox_error (never scored 0.0), even though a real test failure also exits 1.
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/ws-no-such-docker.sock")
    r = run_unittest(_SOL, _TEST, ["test_a"], timeout=40)
    assert r.passed is False and r.verdict == "sandbox_error"


def test_missing_image_is_a_sandbox_error_not_a_zero(monkeypatch):
    # A wrong/absent image is an infrastructure fault (docker run exit 125) → sandbox_error,
    # which the scorer must refuse to record as a genuine 0.0.
    import whystuck.eval.bcb_exec as be

    monkeypatch.setattr(be, "_IMAGE", "whystuck-bcb-nonexistent:latest")
    r = run_unittest(_SOL, _TEST, ["test_a"], timeout=40)
    assert r.passed is False and r.verdict == "sandbox_error"
