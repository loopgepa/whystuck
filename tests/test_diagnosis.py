"""Tests for whystuck.core.diagnosis — deterministic trace analysis (ADR-005)."""

from conftest import FIXED_TS

from whystuck.core.diagnosis import infer_failure_modes, summarize_traces
from whystuck.core.models import (
    ConfidenceBand,
    FinalOutcome,
    LoopMetrics,
    LoopStepTrace,
    LoopTrace,
    StepStatus,
)


def _trace(
    tid: str,
    outcome: FinalOutcome,
    failure_summary: str | None = None,
    failed_role: str | None = None,
    retry_count: int = 0,
) -> LoopTrace:
    steps = []
    if failed_role is not None:
        steps.append(LoopStepTrace(step_id="s1", role=failed_role, status=StepStatus.FAILURE))
    return LoopTrace(
        id=tid,
        loop_id="bugfix",
        created_at=FIXED_TS,
        final_outcome=outcome,
        failure_summary=failure_summary,
        steps=steps,
        metrics=LoopMetrics(retry_count=retry_count),
    )


class TestSummarizeTraces:
    def test_success_rate(self):
        traces = [
            _trace("t1", "success"),
            _trace("t2", "success"),
            _trace("t3", "failure"),
            _trace("t4", "failure"),
        ]
        summary = summarize_traces("bugfix", traces)
        assert summary.trace_count == 4
        assert summary.success_rate == 0.5

    def test_common_failure_summaries_ranked(self):
        traces = [
            _trace("t1", "failure", failure_summary="repair repeats mistake"),
            _trace("t2", "failure", failure_summary="repair repeats mistake"),
            _trace("t3", "failure", failure_summary="verifier too weak"),
        ]
        summary = summarize_traces("bugfix", traces)
        assert summary.common_failure_summaries[0] == "repair repeats mistake"

    def test_avg_retry_count(self):
        traces = [_trace("t1", "failure", retry_count=2), _trace("t2", "success", retry_count=0)]
        summary = summarize_traces("bugfix", traces)
        assert summary.avg_retry_count == 1.0

    def test_empty_traces_zero_success_rate(self):
        summary = summarize_traces("bugfix", [])
        assert summary.trace_count == 0
        assert summary.success_rate == 0.0


class TestInferFailureModes:
    def test_no_failures_returns_empty(self):
        modes = infer_failure_modes([_trace("t1", "success"), _trace("t2", "success")])
        assert modes == []

    def test_single_dominant_mode_collects_evidence(self):
        traces = [
            _trace(f"t{i}", "failure", failure_summary="repair repeats mistake") for i in range(6)
        ]
        modes = infer_failure_modes(traces)
        assert len(modes) == 1
        assert set(modes[0].evidence_trace_ids) == {f"t{i}" for i in range(6)}

    def test_modes_ranked_by_evidence_count(self):
        traces = [
            _trace("t1", "failure", failure_summary="A"),
            _trace("t2", "failure", failure_summary="A"),
            _trace("t3", "failure", failure_summary="B"),
        ]
        modes = infer_failure_modes(traces)
        assert modes[0].title == "A"
        assert len(modes[0].evidence_trace_ids) == 2

    def test_few_traces_is_low_confidence(self):
        traces = [
            _trace("t1", "failure", failure_summary="A"),
            _trace("t2", "failure", failure_summary="A"),
        ]
        modes = infer_failure_modes(traces)
        assert modes[0].confidence.band is ConfidenceBand.LOW

    def test_verify_step_failure_implicates_verifier_rubric(self):
        traces = [
            _trace(f"t{i}", "failure", failure_summary="bad", failed_role="verify")
            for i in range(6)
        ]
        modes = infer_failure_modes(traces)
        assert "verifier_rubric" in modes[0].likely_genes
