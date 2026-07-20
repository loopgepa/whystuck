"""Turn a loop's metrics into scores for optimization.

`scalarize` is what GEPA maximizes (one float, success-dominant). `objective_vector`
keeps the full multi-objective view for the Pareto frontier and reporting — we
never collapse everything into one number for the user (IMPLEMENTATION §17.4).
"""

from __future__ import annotations

from whystuck.core.models import LoopMetrics

# A success always outranks any failure: failures score 0, successes score in
# [_SUCCESS_FLOOR, 1.0], with the remainder rewarding efficiency.
_SUCCESS_FLOOR = 0.6
_RETRY_FULL_PENALTY_AT = 5  # retries at/above this exhaust the retry share
_DIFF_FULL_PENALTY_AT = 200  # diff lines at/above this exhaust the diff share


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def scalarize(metrics: LoopMetrics) -> float:
    if not metrics.success:
        return 0.0
    retry_pen = min(1.0, metrics.retry_count / _RETRY_FULL_PENALTY_AT)
    diff_pen = min(1.0, (metrics.diff_size_lines or 0) / _DIFF_FULL_PENALTY_AT)
    efficiency = 1.0 - 0.5 * retry_pen - 0.5 * diff_pen
    return round(_SUCCESS_FLOOR + (1.0 - _SUCCESS_FLOOR) * max(0.0, efficiency), 4)


def objective_vector(metrics: LoopMetrics) -> dict[str, float]:
    """Multi-objective view, each in [0, 1], higher = better."""
    return {
        "success": 1.0 if metrics.success else 0.0,
        "cost_inverse": _clamp01(1.0 / (1.0 + (metrics.cost_usd or 0.0))),
        "latency_inverse": _clamp01(1.0 / (1.0 + (metrics.latency_seconds or 0.0))),
        "diff_inverse": _clamp01(1.0 / (1.0 + (metrics.diff_size_lines or 0))),
        "safety": _clamp01(metrics.safety_score if metrics.safety_score is not None else 1.0),
    }
