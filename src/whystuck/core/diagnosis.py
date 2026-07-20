"""Deterministic trace analysis: summaries and failure-mode inference.

No LLM here — this produces the evidence and confidence (ADR-005); Claude does
the reflective interpretation on top of it. Confidence is a function of trace
volume and how concentrated the failures are in one mode.
"""

from __future__ import annotations

from collections import Counter, OrderedDict

from pydantic import BaseModel, Field

from whystuck.core.models import Confidence, LoopTrace, StepStatus, assess_confidence

# Map a failing step's role to the loop genes most likely responsible. Used to
# point diagnosis at the right knob; deliberately small and conservative.
_ROLE_TO_GENES: dict[str, list[str]] = {
    "verify": ["verifier_rubric"],
    "verifier": ["verifier_rubric"],
    "repair": ["retry_policy", "prompt"],
    "retry": ["retry_policy"],
    "plan": ["skill"],
}
_DEFAULT_GENES = ["skill"]


class TraceSummary(BaseModel):
    loop_id: str
    trace_count: int
    success_rate: float
    common_failure_summaries: list[str] = Field(default_factory=list)
    common_failed_steps: list[str] = Field(default_factory=list)
    avg_retry_count: float = 0.0
    avg_latency_seconds: float | None = None


class FailureMode(BaseModel):
    id: str
    title: str
    evidence_trace_ids: list[str]
    likely_genes: list[str]
    confidence: Confidence


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_traces(loop_id: str, traces: list[LoopTrace]) -> TraceSummary:
    n = len(traces)
    successes = sum(1 for t in traces if t.final_outcome == "success")
    failing = [t for t in traces if t.final_outcome == "failure"]

    summaries = Counter(t.failure_summary for t in failing if t.failure_summary)
    failed_steps: Counter[str] = Counter(
        s.role or s.step_id for t in failing for s in t.steps if s.status is StepStatus.FAILURE
    )
    latencies = [t.metrics.latency_seconds for t in traces if t.metrics.latency_seconds is not None]

    return TraceSummary(
        loop_id=loop_id,
        trace_count=n,
        success_rate=round(successes / n, 4) if n else 0.0,
        common_failure_summaries=[s for s, _ in summaries.most_common()],
        common_failed_steps=[s for s, _ in failed_steps.most_common()],
        avg_retry_count=round(_mean([float(t.metrics.retry_count) for t in traces]), 4),
        avg_latency_seconds=round(_mean(latencies), 4) if latencies else None,
    )


def _group_key(trace: LoopTrace) -> str:
    if trace.failure_summary:
        return trace.failure_summary
    for step in trace.steps:
        if step.status is StepStatus.FAILURE:
            return f"step:{step.role or step.step_id}"
    return "unknown failure"


def _likely_genes(group: list[LoopTrace]) -> list[str]:
    genes: list[str] = []
    for trace in group:
        for step in trace.steps:
            if step.status is StepStatus.FAILURE and step.role:
                for gene in _ROLE_TO_GENES.get(step.role.lower(), []):
                    if gene not in genes:
                        genes.append(gene)
    return genes or list(_DEFAULT_GENES)


def infer_failure_modes(traces: list[LoopTrace]) -> list[FailureMode]:
    failing = [t for t in traces if t.final_outcome == "failure"]
    if not failing:
        return []

    groups: OrderedDict[str, list[LoopTrace]] = OrderedDict()
    for trace in failing:
        groups.setdefault(_group_key(trace), []).append(trace)

    ranked = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    n_total = len(traces)
    n_failing = len(failing)

    modes: list[FailureMode] = []
    for i, (title, group) in enumerate(ranked, start=1):
        cluster_share = len(group) / n_failing
        modes.append(
            FailureMode(
                id=f"fm-{i:03d}",
                title=title,
                evidence_trace_ids=[t.id for t in group],
                likely_genes=_likely_genes(group),
                confidence=assess_confidence(n_traces=n_total, cluster_share=cluster_share),
            )
        )
    return modes
