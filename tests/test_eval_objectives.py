"""Tests for whystuck.eval.objectives — the scalarizer + objective vector."""

from whystuck.core.models import LoopMetrics
from whystuck.eval.objectives import objective_vector, scalarize


def _m(**kw) -> LoopMetrics:
    return LoopMetrics(**kw)


class TestScalarize:
    def test_failure_scores_zero(self):
        assert scalarize(_m(success=False)) == 0.0

    def test_none_success_treated_as_failure(self):
        assert scalarize(_m(success=None)) == 0.0

    def test_clean_success_scores_one(self):
        assert scalarize(_m(success=True, retry_count=0, diff_size_lines=0)) == 1.0

    def test_any_success_beats_any_failure(self):
        worst_success = scalarize(_m(success=True, retry_count=99, diff_size_lines=9999))
        assert worst_success > scalarize(_m(success=False))

    def test_fewer_retries_scores_higher(self):
        lean = scalarize(_m(success=True, retry_count=0))
        churny = scalarize(_m(success=True, retry_count=4))
        assert lean > churny


class TestObjectiveVector:
    def test_vector_has_expected_keys(self):
        v = objective_vector(_m(success=True, cost_usd=0.5, latency_seconds=2.0))
        assert {"success", "cost_inverse", "latency_inverse", "safety"} <= set(v)

    def test_all_values_in_unit_interval(self):
        v = objective_vector(_m(success=True, cost_usd=3.0, latency_seconds=10.0, safety_score=0.9))
        assert all(0.0 <= x <= 1.0 for x in v.values())
