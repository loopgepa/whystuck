"""Tests for whystuck.eval.cost — the budget guard for the one paid study run.

This is the safety rail that keeps the live run under the user's hard cap. It is
pure arithmetic over a pricing table + a per-call plan, so it's fully offline.
"""

from __future__ import annotations

import pytest

from whystuck.eval.cost import (
    MODEL_PRICES,
    BudgetExceeded,
    CallGroup,
    CostGuard,
    estimate_usd,
    plan_study_calls,
)


def test_estimate_usd_matches_hand_computation():
    # gpt-4.1-mini @ $0.80/$3.20 per 1M: 1000 calls * (600 in, 250 out)
    # = 1000 * (600*0.8 + 250*3.2)/1e6 = 1000 * (480 + 800)/1e6 = 1000 * 0.00128
    groups = [CallGroup(model="openai/gpt-4.1-mini", calls=1000, in_tokens=600, out_tokens=250)]
    assert estimate_usd(groups) == pytest.approx(1.28)


def test_estimate_usd_sums_multiple_groups():
    groups = [
        CallGroup(model="openai/gpt-4.1-mini", calls=10, in_tokens=600, out_tokens=250),
        CallGroup(model="openai/gpt-5.6-terra", calls=10, in_tokens=4000, out_tokens=1500),
    ]
    # worker: 10 * 0.00128 = 0.0128 ; reflection: 10 * (4000*2.5+1500*15)/1e6 = 10*0.0325 = 0.325
    assert estimate_usd(groups) == pytest.approx(0.0128 + 0.325)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        estimate_usd([CallGroup(model="openai/does-not-exist", calls=1, in_tokens=1, out_tokens=1)])


def test_guard_allows_plan_under_ceiling():
    guard = CostGuard(max_usd=5.0)
    groups = [CallGroup(model="openai/gpt-4.1-mini", calls=100, in_tokens=600, out_tokens=250)]
    # 100 * 0.00128 = 0.128 < 5 — returns the estimate, no raise
    assert guard.check(groups) == pytest.approx(0.128)


def test_guard_refuses_plan_over_ceiling():
    guard = CostGuard(max_usd=1.0)
    groups = [CallGroup(model="openai/gpt-5.6-sol", calls=1000, in_tokens=4000, out_tokens=1500)]
    with pytest.raises(BudgetExceeded) as exc:
        guard.check(groups)
    assert "1.0" in str(exc.value)  # names the ceiling it would breach


def test_guard_allows_exact_ceiling_boundary():
    # estimate == max_usd is allowed (strict `>`): pin the off-by-one boundary.
    groups = [CallGroup(model="openai/gpt-4.1-mini", calls=1000, in_tokens=600, out_tokens=250)]
    estimate = estimate_usd(groups)  # 1.28
    assert CostGuard(max_usd=estimate).check(groups) == pytest.approx(estimate)


def test_guard_zero_ceiling_refuses_any_positive_cost():
    groups = [CallGroup(model="openai/gpt-4.1-mini", calls=1, in_tokens=600, out_tokens=250)]
    with pytest.raises(BudgetExceeded):
        CostGuard(max_usd=0.0).check(groups)


def test_guard_negative_ceiling_refuses_even_zero_cost_plan():
    # Fail-closed: a negative ceiling refuses everything, even an empty plan.
    with pytest.raises(BudgetExceeded):
        CostGuard(max_usd=-1.0).check([])


def _plan(arms, *, n_train=5, n_heldout=5, mmc=20, refl=3, reflection="openai/gpt-5.6-terra"):
    return plan_study_calls(
        arms=arms,
        n_train=n_train,
        n_heldout=n_heldout,
        max_metric_calls=mmc,
        reflection_calls=refl,
        worker_model="openai/gpt-4.1-mini",
        reflection_model=reflection,
    )


def test_plan_study_calls_counts_conservatively():
    # worker: A=n_h ; B/D=mmc+n_train+n_h each ; C=2*n_h.
    # = 5 + (20+5+5) + 10 + (20+5+5) = 5+30+10+30 = 75+10 = 85 ; reflection=2*3=6
    groups = _plan(("A", "B", "C", "D"))
    by_model = {g.model: g for g in groups}
    assert by_model["openai/gpt-4.1-mini"].calls == 85
    assert by_model["openai/gpt-5.6-terra"].calls == 6


@pytest.mark.parametrize(
    "arm,worker,reflection",
    [
        ("A", 31, 0),  # seed held-out eval only
        ("B", 31 + 150 + 31 + 31, 15),  # seed + search + train-rescore + winner-heldout
        ("C", 62, 0),  # seed + winner held-out; command opt/rescore is free
        ("D", 31 + 150 + 31 + 31, 15),  # same shape as B
    ],
)
def test_plan_study_calls_per_arm_isolation(arm, worker, reflection):
    groups = _plan((arm,), n_train=31, n_heldout=31, mmc=150, refl=15)
    by_model = {g.model: g.calls for g in groups}
    assert by_model["openai/gpt-4.1-mini"] == worker
    assert by_model["openai/gpt-5.6-terra"] == reflection


def test_plan_study_calls_zero_reflection_when_no_llm_optimizing_arms():
    groups = _plan(("A", "C"))
    assert all(g.model == "openai/gpt-4.1-mini" for g in groups if g.calls > 0)
    assert sum(g.calls for g in groups if g.model == "openai/gpt-5.6-terra") == 0


def test_plan_study_calls_ignores_unknown_arm_letters():
    # Library-level behavior: unknown letters contribute nothing (the CLI is what
    # rejects them loudly — see test_run_validation_study). Pins the internal
    # contract so it can't drift silently.
    assert estimate_usd(_plan(("A", "B", "X"), n_train=31, n_heldout=31, mmc=150, refl=15)) == (
        pytest.approx(estimate_usd(_plan(("A", "B"), n_train=31, n_heldout=31, mmc=150, refl=15)))
    )


def test_n_seeds_scales_the_whole_plan():
    one = _plan(("A", "B", "C", "D"), n_train=31, n_heldout=31, mmc=150, refl=15)
    two = plan_study_calls(
        arms=("A", "B", "C", "D"),
        n_train=31,
        n_heldout=31,
        max_metric_calls=150,
        reflection_calls=15,
        worker_model="openai/gpt-4.1-mini",
        reflection_model="openai/gpt-5.6-terra",
        n_seeds=2,
    )
    assert estimate_usd(two) == pytest.approx(2 * estimate_usd(one))


def test_two_seed_sol_run_exceeds_cap_but_terra_fits():
    # The guard must force a choice: Sol + 2 seeds is over $5; Terra + 2 seeds fits.
    def plan(reflection):
        return plan_study_calls(
            arms=("A", "B", "C", "D"),
            n_train=31,
            n_heldout=31,
            max_metric_calls=150,
            reflection_calls=15,
            worker_model="openai/gpt-4.1-mini",
            reflection_model=reflection,
            n_seeds=2,
        )

    assert estimate_usd(plan("openai/gpt-5.6-terra")) < 5.0
    assert estimate_usd(plan("openai/gpt-5.6-sol")) > 5.0


def test_full_study_estimate_is_within_five_dollars():
    # The real planned run (31/31 split, 150 metric calls) must sit under the $5
    # cap for BOTH reflection models — the spec's headline budget claim.
    for reflection in ("openai/gpt-5.6-terra", "openai/gpt-5.6-sol"):
        groups = _plan(
            ("A", "B", "C", "D"), n_train=31, n_heldout=31, mmc=150, refl=15, reflection=reflection
        )
        assert estimate_usd(groups) < 5.0


def test_pricing_table_has_the_study_models():
    for m in ("openai/gpt-4.1-mini", "openai/gpt-5.6-terra", "openai/gpt-5.6-sol"):
        assert m in MODEL_PRICES
        inp, out = MODEL_PRICES[m]
        assert inp > 0 and out > 0
