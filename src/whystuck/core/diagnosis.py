"""Deterministic trace analysis: summaries and failure-mode inference.

No LLM here — this produces the evidence and confidence (ADR-005); Claude does
the reflective interpretation on top of it. Confidence is a function of trace
volume and how concentrated the failures are in one mode.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from typing import Any

from pydantic import BaseModel, Field

from whystuck.core.models import Confidence, LoopTrace, StepStatus, assess_confidence
from whystuck.core.util import mean

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
    #: The base rate of already-correct first drafts, and what it implies. Paper I's answer to
    #: "should I optimize the loop or the prompt" is conditional on this number and inverts
    #: across it, so a diagnosis that omits it cannot support the recommendation that follows.
    p0_base_rate: float | None = None
    loop_lever: dict[str, Any] = Field(default_factory=dict)


class FailureMode(BaseModel):
    id: str
    title: str
    evidence_trace_ids: list[str]
    likely_genes: list[str]
    confidence: Confidence


def summarize_traces(loop_id: str, traces: list[LoopTrace]) -> TraceSummary:
    n = len(traces)
    p0 = p0_base_rate(traces)
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
        avg_retry_count=round(mean([float(t.metrics.retry_count) for t in traces]), 4),
        avg_latency_seconds=round(mean(latencies), 4) if latencies else None,
        p0_base_rate=p0,
        loop_lever=loop_lever_verdict(p0),
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


# --- What the three papers found, as things this tool reports ------------------
#
# These come from the series in docs/papers/. They are here rather than in eval/ because the
# research harness already knew all three and the served tool did not, which is the gap between
# having measured something and shipping it.

#: p0 regimes, named for the levels the p0 sweep actually measured.
LOOP_LEVER_LOW = "low"
LOOP_LEVER_MID = "mid"
LOOP_LEVER_HIGH = "high"

#: Paper I measured p0 at these three levels. The boundaries below sit between them rather than
#: on them, because a threshold placed exactly on a measured point claims a precision the three
#: levels cannot support.
_P0_LOW_CEILING = 0.15
_P0_HIGH_FLOOR = 0.50

#: Paper II, decomposing a perfect oracle's +0.0437: the verdict contributes +0.0110 (t = 1.37,
#: which is nothing) and the repair text +0.0327 (t = 4.94). The advice does not change with p0.
_FIX_SIDE_ADVICE = (
    "Spend on the repair text before the detector. Holding the checker perfect and varying only "
    "the repair prose moves held-out score +0.0613 (t = 8.08), the best-supported effect in the "
    "series, while a perfect verdict carrying no actionable text buys +0.0110 (t = 1.37)."
)


def p0_base_rate(traces: list[LoopTrace]) -> float | None:
    """The share of runs whose first draft already satisfied every requirement.

    Operationalised as a run that succeeded needing no retry. Returns None on no traces, which is
    a different answer from 0.0: one says first drafts are never right, the other says nobody has
    looked, and they lead to opposite recommendations.
    """
    if not traces:
        return None
    clean = sum(
        1 for t in traces if t.final_outcome == "success" and (t.metrics.retry_count or 0) == 0
    )
    return round(clean / len(traces), 4)


def loop_lever_verdict(p0: float | None) -> dict[str, object]:
    """Whether optimizing the loop is indicated at this base rate, and what to do instead.

    Paper I's answer is conditional, and the condition is p0. At p0 = 0.067 the reflective search
    converged on a verifier that never accepts anything, on 11 of 12 evolved genes, and evolving
    it bought nothing measurable; at p0 = 0.267 the contrast turned. Reporting a recommendation
    without this number is reporting the coin flip.
    """
    if p0 is None:
        return {
            "p0": None,
            "regime": None,
            "loop_optimization_indicated": None,
            "evidence": "p0 not measured: record traces first, since the recommendation inverts "
            "across it and an unmeasured base rate is not a low one.",
            "advice": _FIX_SIDE_ADVICE,
        }
    if p0 < _P0_LOW_CEILING:
        return {
            "p0": p0,
            "regime": LOOP_LEVER_LOW,
            "loop_optimization_indicated": False,
            "evidence": "At p0 = 0.067 the search converged on a verifier that never accepts, "
            "and the loop-level contrast was negative. Unconditional refinement is near-free "
            "when almost no first draft is already correct, so evolving the verifier is "
            "Goodhart's law operating inside the loop.",
            "advice": _FIX_SIDE_ADVICE,
        }
    regime = LOOP_LEVER_MID if p0 < _P0_HIGH_FLOOR else LOOP_LEVER_HIGH
    return {
        "p0": p0,
        "regime": regime,
        "loop_optimization_indicated": True,
        "evidence": "At p0 = 0.267 the evolved critique began emitting PASS (four of thirty) and "
        "the loop-level arm stopped losing; at p0 = 0.667 it emitted six. The measured swing "
        "across the axis is 0.102.",
        "advice": _FIX_SIDE_ADVICE,
    }


def screen_optimizer_result(
    genes_before: dict[str, str],
    genes_after: dict[str, str],
    verify_pass: int | None = None,
    verify_total: int | None = None,
) -> list[str]:
    """Two screens Paper III and Paper I pay for, run on an optimizer's output.

    The no-op screen is the one the third paper is named for: across fifteen contrast runs the
    winning genome's critique gene was byte-identical to the seed in 15/15, so ten runs reported
    a held-out score for an experiment that never ran.

    The PASS-rate screen catches the degenerate verifier. It is a screen on the search's RESULT,
    not a gate before its spend: it saves the wrong conclusion and the downstream spend, not the
    search cost.

    An unmeasured PASS rate returns a warning rather than silence, because a screen that could
    not look must not read as a screen that looked and found nothing.
    """
    warnings: list[str] = []

    changed = [k for k, v in genes_after.items() if genes_before.get(k) != v]
    if not changed:
        warnings.append(
            "no-op: every gene is byte-identical to the seed, so any score reported here "
            "measures the seed genome and not the optimizer (Paper III, defect 1)."
        )

    if verify_total:
        if verify_pass == 0:
            warnings.append(
                f"the evolved verifier never accepts: 0 PASS of {verify_total}. At a low base "
                "rate the search converges on this because unconditional refinement is cheaper "
                "than discriminating correctly (Paper I)."
            )
    else:
        warnings.append(
            "verifier PASS rate not measured, so the degenerate-verifier screen did not run. "
            "Pass verify_pass and verify_total to enable it."
        )
    return warnings
