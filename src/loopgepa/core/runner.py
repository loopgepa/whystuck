"""The v0.1 command+verifier ingest runner (IMPLEMENTATION §7).

Offline, free, deterministic: run a declared worker command then a declared
verifier command, retrying on verifier failure, and emit a genuine LoopTrace.
Command execution is injected so callers (and tests) control the environment;
the default executor uses subprocess. This runner opens no network connections
(the optional `claude -p` worker of §7.2 is out of scope here).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

from loopgepa.core.models import (
    LoopMetrics,
    LoopStepTrace,
    LoopTrace,
    StepStatus,
    ToolCall,
)

# Cap stored command output so we keep a useful summary, not raw logs (ADR-004.1).
_OUTPUT_SUMMARY_CAP = 500

Executor = Callable[[list[str]], tuple[int, str]]


def _default_execute(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    return proc.returncode, (proc.stdout + proc.stderr)


def _summarize(output: str) -> str:
    output = output.strip()
    if len(output) <= _OUTPUT_SUMMARY_CAP:
        return output
    return output[:_OUTPUT_SUMMARY_CAP] + "…"


def _step(attempt: int, role: str, cmd: list[str], code: int, output: str) -> LoopStepTrace:
    status = StepStatus.SUCCESS if code == 0 else StepStatus.FAILURE
    return LoopStepTrace(
        step_id=f"{role}-{attempt}",
        role=role,
        status=status,
        tool_calls=[ToolCall(name=cmd[0], exit_code=code, output_summary=_summarize(output))],
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
    execute: Executor = _default_execute,
) -> LoopTrace:
    """Run worker+verifier up to max_retries+1 times; return the resulting trace."""
    steps: list[LoopStepTrace] = []
    passed = False
    attempts_used = 0
    last_verify_output = ""

    for attempt in range(max_retries + 1):
        attempts_used = attempt + 1

        worker_code, worker_out = execute(worker)
        steps.append(_step(attempt, "worker", worker, worker_code, worker_out))

        verify_code, verify_out = execute(verifier)
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
        final_outcome="success" if passed else "failure",
        failure_summary=None if passed else _summarize(last_verify_output),
    )
