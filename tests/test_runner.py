"""Tests for whystuck.core.runner — the offline command+verifier ingest runner.

Command execution is injected so these are deterministic and never touch the
real shell or network (IMPLEMENTATION §7).
"""

from collections.abc import Callable

from conftest import FIXED_TS

from whystuck.core.models import StepStatus
from whystuck.core.runner import run_command_loop

WORKER = ["run-worker"]
VERIFIER = ["run-verifier"]


def _scripted(
    verify_codes: list[int], worker_code: int = 0
) -> Callable[[list[str]], tuple[int, str]]:
    seq = iter(verify_codes)

    def execute(cmd: list[str]) -> tuple[int, str]:
        if cmd == VERIFIER:
            return (next(seq), "verifier said no")
        return (worker_code, "worker ran")

    return execute


def _run(verify_codes: list[int], max_retries: int = 1, **kw):
    return run_command_loop(
        loop_id="bugfix",
        trace_id="t1",
        created_at=FIXED_TS,
        worker=WORKER,
        verifier=VERIFIER,
        max_retries=max_retries,
        execute=_scripted(verify_codes),
        **kw,
    )


class TestOutcome:
    def test_verifier_passes_first_try_is_success(self):
        trace = _run([0])
        assert trace.final_outcome == "success"
        assert trace.metrics.retry_count == 0
        assert trace.failure_summary is None

    def test_verifier_always_fails_is_failure_and_exhausts_retries(self):
        trace = _run([1, 1, 1], max_retries=2)
        assert trace.final_outcome == "failure"
        assert trace.metrics.retry_count == 2
        assert trace.failure_summary is not None

    def test_fails_then_passes_is_success_with_one_retry(self):
        trace = _run([1, 0], max_retries=2)
        assert trace.final_outcome == "success"
        assert trace.metrics.retry_count == 1


class TestSteps:
    def test_records_worker_and_verify_step_per_attempt(self):
        trace = _run([1, 1, 1], max_retries=2)  # 3 attempts
        assert len(trace.steps) == 6
        roles = {s.role for s in trace.steps}
        assert roles == {"worker", "verify"}

    def test_verify_step_status_reflects_exit_code(self):
        trace = _run([0])
        verify = next(s for s in trace.steps if s.role == "verify")
        assert verify.status is StepStatus.SUCCESS
        assert verify.tool_calls[0].exit_code == 0


class TestStamping:
    def test_trace_is_stamped(self):
        trace = _run([0], variant_id="v1")
        assert trace.id == "t1"
        assert trace.loop_id == "bugfix"
        assert trace.variant_id == "v1"
        assert trace.created_at == FIXED_TS


class TestEmptyCommand:
    def test_empty_worker_records_failure_instead_of_crashing(self):
        # subprocess.Popen([]) raises an uncaught IndexError; an empty command
        # list must become a recorded failure step, like the missing-binary
        # path, not crash the whole run. (The worker's own exit code never
        # decides final_outcome — only the verifier does — so we assert on
        # the recorded step, not the trace-level outcome.)
        def execute(cmd: list[str]) -> tuple[int, str]:
            return (0, "should never be called for an empty command")

        trace = run_command_loop(
            loop_id="bugfix",
            trace_id="t1",
            created_at=FIXED_TS,
            worker=[],
            verifier=VERIFIER,
            max_retries=0,
            execute=execute,
        )
        worker_step = next(s for s in trace.steps if s.role == "worker")
        assert worker_step.status.value == "failure"
        assert worker_step.tool_calls[0].exit_code == 127

    def test_empty_verifier_records_failure_instead_of_crashing(self):
        def execute(cmd: list[str]) -> tuple[int, str]:
            return (0, "worker ran fine")

        trace = run_command_loop(
            loop_id="bugfix",
            trace_id="t1",
            created_at=FIXED_TS,
            worker=WORKER,
            verifier=[],
            max_retries=0,
            execute=execute,
        )
        assert trace.final_outcome == "failure"
        verify_step = next(s for s in trace.steps if s.role == "verify")
        assert verify_step.tool_calls[0].exit_code == 127
