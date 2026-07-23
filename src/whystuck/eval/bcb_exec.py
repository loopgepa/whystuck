"""Container-sandboxed runner for BigCodeBench-style unittest suites (v2 code, Docker).

BigCodeBench-Hard tasks use scientific libraries (pandas, matplotlib, sklearn, ...) and
``unittest.TestCase`` suites whose references legitimately touch the filesystem and mock
network I/O. A bare subprocess is neither safe nor reproducible for that, so this runs each
solution + selected tests inside a PINNED Docker image (``whystuck-bcb``; see
``docker/bcb/Dockerfile``) with ``--network none`` (offline ENFORCED, not merely unused), a
memory cap, a PID cap, a fresh writable temp cwd, and a wall-clock timeout that ``docker
kill``s the container. The fixed image + single-thread BLAS + fixed hash seed make execution
a deterministic function of the code — which is what lets held-out scores replay bit-for-bit
even though execution is not cassette-recorded (deterministic-local, like the arm-C runner).

This is the reproducibility-and-safety UPGRADE over ``code_exec``'s subprocess sandbox
(which is "not container-grade" by its own docstring); it reuses that module's
``ExecResult`` / ``Verdict`` / ``INFRA_VERDICTS`` so ``bcb_bench`` can branch on the verdict
exactly like ``code_bench`` — raising on an infrastructure fault (a Docker/host error) rather
than mis-scoring it as a genuine 0.0 on a paid run.

The anti-oracle split runs a CHOSEN SUBSET of the suite's ``test_*`` methods: the visible
subset feeds the in-loop critique, the hidden subset is the held-out scorer, and the loop
never sees the hidden methods. A PASS is reported only when every SELECTED method provably
ran to success — computed from the result's failure/error/skipped LISTS (never the patchable
``wasSuccessful()``) AND ``testsRun == len(selected)`` — and confirmed by an unforgeable
per-run nonce (see the harness below), so a solution cannot score a vacuous pass by exiting
early, running zero tests, monkeypatching unittest, or forging the sentinel.
"""

from __future__ import annotations

import itertools
import os
import subprocess
from collections.abc import Sequence

from whystuck.eval.code_exec import (
    ExecResult,
    _tail,  # shared traceback-tail formatter
)

_SENTINEL = "<<WS_BCB_COMPLETED_7d1e4c>>"
_NO_METHODS = "<<WS_BCB_NO_METHODS>>"
_IMAGE = os.environ.get("WHYSTUCK_BCB_IMAGE", "whystuck-bcb:latest")
_name_counter = itertools.count()

# Runs INSIDE the container. Reads a per-run NONCE from stdin (into a function local — never
# a module global, never env/argv/a file, so a solution cannot read it back via
# /proc/self/{environ,cmdline} or ``import __main__``), then execs the solution, re-imports a
# fresh unittest, execs the tests, runs ONLY the selected test_* methods, and writes
# ``SENTINEL+NONCE`` iff every selected method genuinely passed. Exit codes: 0 = ran (check
# sentinel+nonce), 1 = solution/test raised or a test failed, 3 = no runnable methods, 5 =
# environment fault reading the mounted files (mount).
#
# Anti-cheat measures for an OPTIMIZATION loop (a solution under reward pressure must not
# grade itself pass):
#   (a) UNFORGEABLE PASS SIGNAL — the pass token is SENTINEL + a host-chosen random nonce the
#       solution cannot know, so a solution that forges the sentinel and ``os._exit(0)``s
#       before the tests run cannot produce the matching nonce; the host rejects it.
#   (b) FRESH FRAMEWORK — unittest and its submodules are purged from sys.modules and
#       re-imported AFTER the solution runs, defeating a monkeypatch of TestCase.assertEqual
#       (in unittest.case) or TestResult.wasSuccessful (in unittest.result); importlib.reload
#       would NOT help (it re-runs only the package __init__, not those submodules).
#   (c) RESULT-LIST SUCCESS — pass is computed from failures/errors/SKIPPED lists, never the
#       patchable wasSuccessful(); a skip / expected-failure counts as NOT a pass.
# Residual: a solution using ``sys._getframe`` to walk to the harness frame and steal the
# nonce — much higher effort; the container (--network none, resource caps) is the ultimate
# boundary, and the worker here is a chat-API model with no repo/file access.
_HARNESS = r"""
import sys, io, traceback

def _run():
    nonce = sys.stdin.readline().strip()  # host secret; kept LOCAL, read before any solution code
    sentinel = "<<WS_BCB_COMPLETED_7d1e4c>>"
    no_methods = "<<WS_BCB_NO_METHODS>>"
    selected = sys.argv[1:]

    try:
        with open("solution.py", encoding="utf-8") as f:
            sol_src = f.read()
        with open("tests.py", encoding="utf-8") as f:
            tst_src = f.read()
    except OSError:  # cannot read the mounted files => host/mount fault, NOT a model failure
        traceback.print_exc()
        sys.exit(5)

    g = {"__name__": "__main__"}
    try:
        exec(compile(sol_src, "solution.py", "exec"), g)
    except BaseException:
        traceback.print_exc()
        sys.exit(1)

    for _m in [k for k in list(sys.modules) if k == "unittest" or k.startswith("unittest.")]:
        del sys.modules[_m]
    import unittest

    try:
        exec(compile(tst_src, "tests.py", "exec"), g)
    except BaseException:
        traceback.print_exc()
        sys.exit(1)

    cls = g.get("TestCases")
    if not (isinstance(cls, type) and issubclass(cls, unittest.TestCase)):
        cands = [v for v in g.values()
                 if isinstance(v, type) and issubclass(v, unittest.TestCase)]
        cls = cands[0] if cands else None
    if cls is None:
        sys.stdout.write(no_methods)
        sys.exit(3)

    all_methods = [m for m in dir(cls) if m.startswith("test")]
    methods = [m for m in (selected or all_methods) if m in all_methods]
    if not methods:
        sys.stdout.write(no_methods)
        sys.exit(3)

    suite = unittest.TestSuite([cls(m) for m in methods])
    res = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    passed = (
        res.testsRun == len(methods)
        and not res.failures
        and not res.errors
        and not res.skipped
        and not getattr(res, "expectedFailures", [])
        and not getattr(res, "unexpectedSuccesses", [])
    )
    if passed:
        sys.stdout.write(sentinel + nonce)  # nonce proves the tests genuinely ran to pass
        sys.stdout.flush()
        sys.exit(0)

    lines = []
    for kind, items in (("FAIL", res.failures), ("ERROR", res.errors)):
        for test, tb in items:
            tail = tb.strip().splitlines()[-1] if tb.strip() else ""
            lines.append(f"{kind} {test}: {tail}")
    if res.skipped:
        lines.append(f"SKIPPED {len(res.skipped)} test(s) — a skip is not a pass")
    sys.stderr.write("\n".join(lines[:5]) or "tests did not all pass")
    sys.exit(1)

_run()
"""

# Docker-CLI messages that mean the daemon/connection failed (never appear in a unittest
# traceback), so a mid-run daemon blip is classified as an infra fault regardless of code.
_DOCKER_CONN_ERRORS = (
    "Cannot connect to the Docker daemon",
    "Is the docker daemon running",
    "error during connect",
    "failed to connect to the docker API",
)


def _docker_cmd(name: str, workdir: str, mem_mb: int) -> list[str]:
    """The locked-down ``docker run`` argv: no network, capped memory/PIDs/CPU, non-root,
    a writable HOME/MPLCONFIGDIR in the container's tmpfs, the workdir mounted read-write."""
    uid, gid = os.getuid(), os.getgid()
    return [
        "docker", "run", "--rm", "--name", name,
        "-i",  # attach stdin so the host can feed the per-run nonce (never via env/argv/file)
        "--network", "none",
        "--memory", f"{mem_mb}m", "--memory-swap", f"{mem_mb}m",
        "--pids-limit", "512", "--cpus", "2",
        "--user", f"{uid}:{gid}",
        "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp",
        "-v", f"{workdir}:/w", "-w", "/w",
        _IMAGE, "python", "harness.py",
    ]  # fmt: skip


def _kill(name: str) -> None:
    """Force the container down on timeout. ``docker kill`` then ``docker rm -f`` as a
    backstop so a wedged container can't linger and starve later tasks of the CPU/mem budget."""
    for cmd in (["docker", "kill", name], ["docker", "rm", "-f", name]):
        with subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as p:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()


def run_unittest(
    solution: str,
    test_code: str,
    method_names: Sequence[str] | None = None,
    *,
    timeout: float = 30.0,
    mem_mb: int = 2048,
) -> ExecResult:
    """Run ``solution`` + ``test_code`` in a Docker sandbox, executing the given unittest
    methods (``None`` => all ``test_*``). Returns ``passed=True`` (verdict ``"pass"``) iff
    every selected method ran to success; otherwise a structured verdict:
    ``"failed"`` (a test failed / the code raised), ``"timeout"``, ``"incomplete"`` (clean
    exit, no sentinel), ``"no_assertions"`` (no runnable test methods — a corrupt split), or
    ``"sandbox_error"`` (Docker/host fault — the caller must NOT score this as 0.0)."""
    import tempfile

    name = f"wsbcb_{os.getpid()}_{next(_name_counter)}"
    # Per-run unforgeable pass token: the harness must echo SENTINEL+nonce, and it only
    # learns the nonce by reading stdin (kept host-side, never in env/argv/a file). A
    # solution that forges the sentinel + os._exit(0) cannot produce the matching nonce.
    nonce = os.urandom(16).hex()
    try:
        with tempfile.TemporaryDirectory() as workdir:
            for fname, content in (
                ("solution.py", solution),
                ("tests.py", test_code),
                ("harness.py", _HARNESS),
            ):
                with open(os.path.join(workdir, fname), "w", encoding="utf-8") as fh:
                    fh.write(content)
            os.chmod(workdir, 0o777)  # the container's mapped user must write temp files here
            args = _docker_cmd(name, workdir, mem_mb)
            if method_names:
                args += list(method_names)
            try:
                proc = subprocess.run(
                    args, input=f"{nonce}\n", capture_output=True, text=True,
                    timeout=timeout, check=False,
                )  # fmt: skip
            except subprocess.TimeoutExpired:
                _kill(name)
                return ExecResult(False, "timeout", verdict="timeout")
    except OSError as exc:  # cannot even launch docker (missing binary, etc.)
        return ExecResult(False, f"sandbox error: {exc}", verdict="sandbox_error")

    out, err, rc = proc.stdout, proc.stderr, proc.returncode
    # Host/sandbox faults (daemon down, image/exec broken, OOM-kill, mount fault) must NOT be
    # scored as a genuine 0.0 — the caller (_raise_if_infra) aborts on these. A daemon
    # connection blip exits 1 (overloaded with a real test failure), so match its message.
    if any(msg in err for msg in _DOCKER_CONN_ERRORS):
        return ExecResult(False, f"sandbox error: docker daemon: {_tail(err)}", "sandbox_error")
    if rc in (125, 126, 127):  # container/exec could not start (daemon, image, PATH)
        return ExecResult(False, f"sandbox error: docker run failed: {_tail(err)}", "sandbox_error")
    if rc == 137:  # 128+SIGKILL: OOM-killed or externally killed
        return ExecResult(False, "sandbox error: container killed (137/OOM)", "sandbox_error")
    if rc == 5:  # harness could not read the mounted files => mount/permission fault
        return ExecResult(False, f"sandbox error: mount/read fault: {_tail(err)}", "sandbox_error")
    if out.rstrip().endswith(_NO_METHODS):
        return ExecResult(False, "no runnable test methods (empty/corrupt split)", "no_assertions")
    if rc == 0 and out.rstrip().endswith(_SENTINEL + nonce):
        return ExecResult(True, "ok", verdict="pass")
    if rc == 0:  # clean exit, no valid sentinel+nonce => didn't run to a genuine pass
        return ExecResult(False, "tests did not run to completion", verdict="incomplete")
    detail = _tail(err) or _tail(out) or f"exit {rc}"
    return ExecResult(False, detail, verdict="failed")
