"""A minimal sandboxed runner for model-generated Python + unit tests (v2-code study).

The v2 code study needs a *reliable, executable* verifier: run a candidate solution
against a set of `assert`-style tests and return a deterministic pass/fail plus a short
failure message (the actionable feedback the refine step consumes). This is the tool the
critique in v1 lacked.

Correctness is load-bearing (a wrong verdict corrupts a paid run), so a PASS is only
reported when the trusted tests provably ran to completion — never from the exit code
alone. Three defenses, because the executed code is under optimization pressure and could
otherwise reward-hack a pass:

1. Solution and tests are written to SEPARATE files and ``exec``'d as separately-compiled
   units in a harness, so an unterminated string / syntax anomaly in the solution cannot
   swallow the test block (and a syntax error in either surfaces as a failure).
2. The harness prints a completion SENTINEL only after the last test line; PASS requires
   exit 0 AND that sentinel as the final stdout line. Any early ``sys.exit(0)`` /
   ``os._exit(0)`` / ``raise SystemExit`` before the tests finish yields no sentinel → fail.
3. A test block with no ``assert`` is rejected up front (``no assertions``), so an empty or
   truncated hidden-test split can't score a wrong solution as a vacuous pass.

Isolation is a subprocess with POSIX resource limits (CPU + address space), a wall-clock
timeout that kills the whole process group (so descendants don't leak), a fresh temp cwd,
and captured stdout/stderr. **This is not a container-grade sandbox** — it bounds runaway
CPU/memory/time and isolates the filesystem cwd, but does not block network or all
syscalls, and the sentinel defense assumes non-adversarial output (a solution that both
prints the exact sentinel and exits 0 could still spoof a pass). It is acceptable here
because the executed code is `gpt-4.1-mini` output over MBPP+-style pure-function tasks
(no I/O/network needed); run inside a container/VM if that assumption does not hold.

Two further bounds to know before scaling this up: (a) ``preexec_fn`` runs after ``fork``
and is not thread-safe, so call ``run_tests`` from ONE thread (the study runs tasks
sequentially) — parallelize with processes, not threads, or switch to ``posix_spawn``-based
limits first; (b) ``RLIMIT_AS`` bounds only the CHILD's memory — captured stdout/stderr
still flow into this process via ``communicate()``, so a runaway ``print`` loop is bounded
only by the wall-clock timeout window, not by size. Fine for the timeout/task profile here.

The default address-space limit (``mem_mb=2048``) is generous on purpose: MBPP+ plus-test
harnesses import numpy, whose OpenBLAS reserves a large virtual-memory arena on import.
The child env pins ``OPENBLAS/OMP/MKL_NUM_THREADS=1``, which shrinks that reservation to
well under the limit AND makes numpy reductions deterministic; a real memory bomb (e.g. a
10**10-element list, ~80 GB) still dwarfs the limit and is caught.

Determinism: the verdict is a pure function of the code for tasks that are not sensitive
to time, cwd path, or locale (MBPP+ pure functions qualify); the timeout is set well above
true runtime and never arbitrates a correct solution, and hash seed / locale / bytecode
writing / BLAS thread count are pinned. This is why held-out scores reproduce bit-for-bit
under `--replay` even though execution is not cassette-recorded (it is deterministic-local,
like the arm-C command runner).
"""

from __future__ import annotations

import ast
import contextlib
import os
import resource
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

_MB = 1024 * 1024
_SENTINEL = "<<WS_TESTS_COMPLETED_9f3a2b>>"

# Harness: read the two units from disk, exec them in one shared namespace (the MBPP
# shape: solution defines a function the tests call), then print the sentinel LAST.
_HARNESS = (
    "import sys\n"
    'with open("solution.py", encoding="utf-8") as _f:\n    _sol = _f.read()\n'
    'with open("tests.py", encoding="utf-8") as _f:\n    _tst = _f.read()\n'
    '_g = {"__name__": "__main__"}\n'
    'exec(compile(_sol, "solution.py", "exec"), _g)\n'
    'exec(compile(_tst, "tests.py", "exec"), _g)\n'
    f"sys.stdout.write({_SENTINEL!r})\nsys.stdout.flush()\n"
)


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


def _has_assertion(test_code: str) -> bool:
    """True if the test block contains at least one `assert` (so a PASS is meaningful).

    On a syntax error we return True and let the sandbox surface it as a real failure,
    rather than mislabeling malformed tests as 'no assertions'."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return True
    return any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def run_tests(
    solution: str, test_code: str, *, timeout: float = 10.0, mem_mb: int = 2048
) -> ExecResult:
    """Run ``solution`` then ``test_code`` in one sandboxed subprocess.

    Returns ``passed=True`` iff the tests provably ran to completion with every assert
    holding (exit 0 AND the completion sentinel); otherwise ``passed=False`` with a short
    ``detail`` — an assertion/traceback tail, ``"timeout"``, ``"no assertions in test
    block"``, ``"tests did not run to completion"`` (early exit / swallowed), or a
    ``"sandbox error"``. Never raises on ordinary failure or on a transient host error
    (the latter fails just this item, distinguishably, rather than aborting a batch)."""
    if not _has_assertion(test_code):
        return ExecResult(False, "no assertions in test block")

    cpu_seconds = max(1, int(timeout) + 2)  # coarse backstop; wall-clock timeout wins first
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LC_ALL": "C",
        "TZ": "UTC",
        # Pin BLAS/OpenMP to one thread: MBPP+ plus-test harnesses import numpy, and
        # multi-threaded OpenBLAS both (a) reserves >0.5 GB of virtual memory on import
        # — which trips RLIMIT_AS and makes numpy fail to load — and (b) makes reductions
        # non-deterministic across replays. One thread fixes both, which the determinism
        # contract (module docstring) and the plus-test scorer depend on.
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    try:
        with tempfile.TemporaryDirectory() as workdir:
            for name, content in (
                ("solution.py", solution),
                ("tests.py", test_code),
                ("harness.py", _HARNESS),
            ):
                with open(os.path.join(workdir, name), "w", encoding="utf-8") as fh:
                    fh.write(content)
            proc = subprocess.Popen(
                [sys.executable, "harness.py"],
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=_limits(cpu_seconds, mem_mb * _MB),  # noqa: PLW1509 - POSIX sandbox
                start_new_session=True,  # own process group, so a timeout kills descendants
                env=env,
            )
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_group(proc.pid)
                proc.communicate()  # reap the killed group
                return ExecResult(False, "timeout")
    except OSError as exc:  # transient host failure (fork/setrlimit) — fail THIS item, loudly
        return ExecResult(False, f"sandbox error: {exc}")

    if proc.returncode == 0 and out.rstrip().endswith(_SENTINEL):
        return ExecResult(True, "ok")
    if proc.returncode == 0:  # clean exit but no sentinel => tests never completed
        return ExecResult(False, "tests did not run to completion (early exit or swallowed)")
    detail = _tail(err) or _tail(out.replace(_SENTINEL, "")) or f"exit {proc.returncode}"
    return ExecResult(False, detail)


def _kill_group(pid: int) -> None:
    """SIGKILL the child's whole process group (set up via start_new_session), so a
    solution that spawned descendants does not leak them past the timeout."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(pid), signal.SIGKILL)
