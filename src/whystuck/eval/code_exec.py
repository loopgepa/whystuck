"""A minimal sandboxed runner for model-generated Python + unit tests (v2-code study).

The v2 code study needs a *reliable, executable* verifier: run a candidate solution
against a set of `assert`-style tests and return a deterministic pass/fail plus a short
failure message (the actionable feedback the refine step consumes). This is the tool the
critique in v1 lacked.

Isolation is a subprocess with POSIX resource limits (CPU + address space), a wall-clock
timeout, a fresh temp cwd, and captured stdout/stderr. **This is not a container-grade
sandbox** — it bounds runaway CPU/memory/time and isolates the filesystem cwd, but does
not block network or all syscalls. It is acceptable here because the executed code is
non-adversarial model output over MBPP+-style pure-function tasks (no I/O/network needed);
run inside a container/VM if that assumption does not hold.

Determinism: for non-timing-sensitive tasks the verdict is a pure function of the code
(the timeout is set well above true runtime and never arbitrates a correct solution), so
held-out scores reproduce bit-for-bit under `--replay` even though execution itself is not
cassette-recorded (it is deterministic-local, like the arm-C command runner).
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

_MB = 1024 * 1024


@dataclass(frozen=True)
class ExecResult:
    """Verdict of running a solution against a test block."""

    passed: bool
    detail: str  # "ok" on success; a short failure reason otherwise (never empty)


def _limits(cpu_seconds: int, mem_bytes: int) -> Callable[[], None]:
    """preexec_fn: cap CPU seconds and address space in the child before exec (POSIX)."""

    def _apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

    return _apply


def _tail(text: str, lines: int = 3, cap: int = 300) -> str:
    """Last few non-empty lines of a traceback/stderr, length-capped for prompts."""
    kept = [ln for ln in text.strip().splitlines() if ln.strip()]
    return "\n".join(kept[-lines:])[:cap]


def run_tests(
    solution: str, test_code: str, *, timeout: float = 10.0, mem_mb: int = 512
) -> ExecResult:
    """Run ``solution`` then ``test_code`` in one sandboxed subprocess.

    The solution and tests share a module namespace (the MBPP shape: the solution
    defines a function the tests call). Returns ``passed=True`` iff the process exits 0
    (all asserts held); otherwise ``passed=False`` with a short ``detail`` — an assertion/
    traceback tail, ``"timeout"``, or an error message. Never raises on ordinary failure;
    the timeout is enforced by killing the child's process group.
    """
    script = f"{solution}\n\n{test_code}\n"
    # CPU rlimit is a coarse backstop; the wall-clock timeout is the primary guard. Give
    # CPU a little more than wall time so the timeout (which yields a clean 'timeout'
    # detail) fires first on a busy loop, rather than a SIGXCPU with a confusing message.
    cpu_seconds = max(1, int(timeout) + 2)
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "candidate.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(script)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_limits(cpu_seconds, mem_mb * _MB),  # noqa: PLW1509 - POSIX sandbox
                start_new_session=True,  # own process group, so a timeout kills descendants
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return ExecResult(False, "timeout")
        if proc.returncode == 0:
            return ExecResult(True, "ok")
        detail = _tail(proc.stderr) or _tail(proc.stdout) or f"exit {proc.returncode}"
        return ExecResult(False, detail)
