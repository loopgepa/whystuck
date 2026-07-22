"""The v0.1 command+verifier ingest runner (IMPLEMENTATION §7).

Offline, free, deterministic: run a declared worker command then a declared
verifier command, retrying on verifier failure, and emit a genuine LoopTrace.
Command execution is injected so callers (and tests) control the environment;
the default executor uses subprocess (via core.subproc). This runner opens no
network connections (the optional `claude -p` worker of §7.2 is out of scope here).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from whystuck.core.models import (
    LoopMetrics,
    LoopStepTrace,
    LoopTrace,
    ToolCall,
    outcome_for,
    step_status_for,
)
from whystuck.core.subproc import run_capture, summarize

Executor = Callable[[list[str]], tuple[int, str]]


def _default_execute(cmd: list[str]) -> tuple[int, str]:
    return run_capture(cmd)


def _safe_execute(execute: Executor, cmd: list[str]) -> tuple[int, str]:
    """Run a command, converting failure-to-launch into a recorded failure step
    instead of an exception that discards the whole trace."""
    if not cmd:
        # subprocess.Popen([]) raises an uncaught IndexError (not caught below)
        # — an empty declared command is a config error, not a crash.
        return (127, "empty command")
    try:
        return execute(cmd)
    except FileNotFoundError as exc:
        # cmd is non-empty here: the `if not cmd` guard above already returned.
        return (127, f"command not found: {cmd[0]} ({exc})")
    except subprocess.TimeoutExpired as exc:
        return (124, f"timed out: {' '.join(cmd)} ({exc})")
    except OSError as exc:
        return (1, f"failed to run {cmd[0]}: {exc}")


def _step(attempt: int, role: str, cmd: list[str], code: int, output: str) -> LoopStepTrace:
    return LoopStepTrace(
        step_id=f"{role}-{attempt}",
        role=role,
        status=step_status_for(code == 0),
        tool_calls=[
            ToolCall(name=cmd[0] if cmd else "", exit_code=code, output_summary=summarize(output))
        ],
    )


def run_command_loop(
    *,
    loop_id: str,
    trace_id: str,
    created_at: str,
    worker: list[str],
    verifier: list[str],
    max_retries: int = 1,
    variant_id: str | None = None,
    execute: Executor | None = None,
) -> LoopTrace:
    """Run worker+verifier up to max_retries+1 times; return the resulting trace."""
    run = execute or _default_execute
    steps: list[LoopStepTrace] = []
    passed = False
    attempts_used = 0
    last_verify_output = ""

    for attempt in range(max_retries + 1):
        attempts_used = attempt + 1

        worker_code, worker_out = _safe_execute(run, worker)
        steps.append(_step(attempt, "worker", worker, worker_code, worker_out))

        verify_code, verify_out = _safe_execute(run, verifier)
        steps.append(_step(attempt, "verify", verifier, verify_code, verify_out))
        last_verify_output = verify_out

        if verify_code == 0:
            passed = True
            break

    return LoopTrace(
        id=trace_id,
        loop_id=loop_id,
        variant_id=variant_id,
        created_at=created_at,
        steps=steps,
        metrics=LoopMetrics(success=passed, retry_count=attempts_used - 1),
        final_outcome=outcome_for(passed),
        failure_summary=None if passed else summarize(last_verify_output),
    )
