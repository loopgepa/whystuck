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
ran to success (``wasSuccessful()`` AND ``testsRun == len(selected)``), so a solution that
exits early or a suite that silently runs zero tests cannot score a vacuous pass.
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

# Runs INSIDE the container. Execs the solution then the tests in one namespace, locates the
# TestCases class, runs ONLY the method names passed as argv (empty => all test_* methods),
# and prints the sentinel LAST iff every selected method succeeded. Exit codes: 0 = ran
# (check sentinel), 1 = solution/test raised or a test failed, 3 = no runnable methods.
_HARNESS = r"""
import sys, io, unittest, traceback

SENTINEL = "<<WS_BCB_COMPLETED_7d1e4c>>"
selected = sys.argv[1:]
g = {"__name__": "__main__"}
for src in ("solution.py", "tests.py"):
    try:
        with open(src, encoding="utf-8") as f:
            code = f.read()
        exec(compile(code, src, "exec"), g)
    except BaseException:
        traceback.print_exc()
        sys.exit(1)

cls = g.get("TestCases")
if not (isinstance(cls, type) and issubclass(cls, unittest.TestCase)):
    cands = [v for v in g.values()
             if isinstance(v, type) and issubclass(v, unittest.TestCase)]
    cls = cands[0] if cands else None
if cls is None:
    sys.stdout.write("<<WS_BCB_NO_METHODS>>")
    sys.exit(3)

all_methods = [m for m in dir(cls) if m.startswith("test")]
methods = [m for m in (selected or all_methods) if m in all_methods]
if not methods:
    sys.stdout.write("<<WS_BCB_NO_METHODS>>")
    sys.exit(3)

suite = unittest.TestSuite([cls(m) for m in methods])
buf = io.StringIO()
res = unittest.TextTestRunner(stream=buf, verbosity=0).run(suite)
if res.wasSuccessful() and res.testsRun == len(methods):
    sys.stdout.write(SENTINEL)
    sys.stdout.flush()
    sys.exit(0)

lines = []
for kind, items in (("FAIL", res.failures), ("ERROR", res.errors)):
    for test, tb in items:
        tail = tb.strip().splitlines()[-1] if tb.strip() else ""
        lines.append(f"{kind} {test}: {tail}")
sys.stderr.write("\n".join(lines[:5]) or "tests did not all pass")
sys.exit(1)
"""


def _docker_cmd(name: str, workdir: str, mem_mb: int) -> list[str]:
    """The locked-down ``docker run`` argv: no network, capped memory/PIDs/CPU, non-root,
    a writable HOME/MPLCONFIGDIR in the container's tmpfs, the workdir mounted read-write."""
    uid, gid = os.getuid(), os.getgid()
    return [
        "docker", "run", "--rm", "--name", name,
        "--network", "none",
        "--memory", f"{mem_mb}m", "--memory-swap", f"{mem_mb}m",
        "--pids-limit", "512", "--cpus", "2",
        "--user", f"{uid}:{gid}",
        "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp",
        "-v", f"{workdir}:/w", "-w", "/w",
        _IMAGE, "python", "harness.py",
    ]  # fmt: skip


def _kill(name: str) -> None:
    with subprocess.Popen(
        ["docker", "kill", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ) as p:
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
                    args, capture_output=True, text=True, timeout=timeout, check=False
                )
            except subprocess.TimeoutExpired:
                _kill(name)
                return ExecResult(False, "timeout", verdict="timeout")
    except OSError as exc:  # cannot even launch docker (missing binary, etc.)
        return ExecResult(False, f"sandbox error: {exc}", verdict="sandbox_error")

    out, err, rc = proc.stdout, proc.stderr, proc.returncode
    # Docker-level faults (daemon/image/OOM-kill) are host faults, not model failures.
    if rc == 125:  # docker could not run the container at all
        return ExecResult(False, f"sandbox error: docker run failed: {_tail(err)}", "sandbox_error")
    if rc == 137:  # 128+SIGKILL: OOM-killed or externally killed
        return ExecResult(False, "sandbox error: container killed (137/OOM)", "sandbox_error")
    if out.rstrip().endswith(_NO_METHODS):
        return ExecResult(False, "no runnable test methods (empty/corrupt split)", "no_assertions")
    if rc == 0 and out.rstrip().endswith(_SENTINEL):
        return ExecResult(True, "ok", verdict="pass")
    if rc == 0:  # clean exit, no sentinel => did not run to completion
        return ExecResult(False, "tests did not run to completion", verdict="incomplete")
    detail = _tail(err) or _tail(out) or f"exit {rc}"
    return ExecResult(False, detail, verdict="failed")
