"""Tests for whystuck.eval.cost — the budget guard for the one paid study run.

This is the safety rail that keeps the live run under the user's hard cap. It is
pure arithmetic over a pricing table + a per-call plan, so it's fully offline.
"""

from __future__ import annotations

import pytest

from whystuck.eval.cost import (
    MODEL_PRICES,
    REFLECTION_MAX_OUTPUT_TOKENS,
    REFLECTION_TOKENS,
    WORKER_MAX_OUTPUT_TOKENS,
    WORKER_TOKENS,
    BudgetExceeded,
    CallGroup,
    CostGuard,
    estimate_usd,
    plan_study_calls,
)


def test_estimate_usd_matches_hand_computation():
    # gpt-4.1-mini @ $0.40/$1.60 per 1M: 1000 calls * (600 in, 250 out)
    # = 1000 * (600*0.4 + 250*1.6)/1e6 = 1000 * (240 + 400)/1e6 = 1000 * 0.00064
    groups = [CallGroup(model="openai/gpt-4.1-mini", calls=1000, in_tokens=600, out_tokens=250)]
    assert estimate_usd(groups) == pytest.approx(0.64)


def test_estimate_usd_sums_multiple_groups():
    groups = [
        CallGroup(model="openai/gpt-4.1-mini", calls=10, in_tokens=600, out_tokens=250),
        CallGroup(model="openai/gpt-5.6-terra", calls=10, in_tokens=4000, out_tokens=1500),
    ]
    # worker: 10 * 0.00064 = 0.0064 ; reflection: 10 * (4000*2.5+1500*15)/1e6 = 10*0.0325 = 0.325
    assert estimate_usd(groups) == pytest.approx(0.0064 + 0.325)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        estimate_usd([CallGroup(model="openai/does-not-exist", calls=1, in_tokens=1, out_tokens=1)])


def test_guard_allows_plan_under_ceiling():
    guard = CostGuard(max_usd=5.0)
    groups = [CallGroup(model="openai/gpt-4.1-mini", calls=100, in_tokens=600, out_tokens=250)]
    # 100 * 0.00064 = 0.064 < 5 — returns the estimate, no raise
    assert guard.check(groups) == pytest.approx(0.064)


def test_guard_refuses_plan_over_ceiling():
    guard = CostGuard(max_usd=1.0)
    groups = [CallGroup(model="openai/gpt-5.6-sol", calls=1000, in_tokens=4000, out_tokens=1500)]
    with pytest.raises(BudgetExceeded) as exc:
        guard.check(groups)
    assert "1.0" in str(exc.value)  # names the ceiling it would breach


def test_guard_allows_exact_ceiling_boundary():
    # estimate == max_usd is allowed (strict `>`): pin the off-by-one boundary.
    groups = [CallGroup(model="openai/gpt-4.1-mini", calls=1000, in_tokens=600, out_tokens=250)]
    estimate = estimate_usd(groups)  # 0.64
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
    # Single-shot-worker costing (worker_calls_per_task=1): the guard must force a
    # reflection-model choice — Sol + 2 seeds is over $5; Terra + 2 seeds fits. (The
    # actual paid run is the cheaper loop config in test_default_loop_config_fits_*;
    # this pins the reflection-price sensitivity in isolation.)
    def plan(reflection):
        return plan_study_calls(
            arms=("A", "B", "C", "D"),
            n_train=31,
            n_heldout=31,
            max_metric_calls=200,
            reflection_calls=15,
            worker_model="openai/gpt-4.1-mini",
            reflection_model=reflection,
            n_seeds=2,
        )

    assert estimate_usd(plan("openai/gpt-5.6-terra")) < 5.0  # ≈ $3.17
    assert estimate_usd(plan("openai/gpt-5.6-sol")) > 5.0  # ≈ $5.12


def test_full_study_estimate_is_within_five_dollars():
    # Single-shot costing sanity (worker_calls_per_task=1): a 31/31 split at 150
    # metric calls sits under $5 for BOTH reflection models. This bounds the search
    # budget; the loop multiplier is priced separately (test_default_loop_config_*).
    for reflection in ("openai/gpt-5.6-terra", "openai/gpt-5.6-sol"):
        groups = _plan(
            ("A", "B", "C", "D"), n_train=31, n_heldout=31, mmc=150, refl=15, reflection=reflection
        )
        assert estimate_usd(groups) < 5.0


def test_worker_calls_per_task_scales_only_the_worker_group():
    # The self-refine loop makes 1+2*retries LLM calls per task-eval. That multiplier
    # must scale the WORKER call count (and only it) — reflection is per-proposal,
    # untouched. A regression here would silently under-count the guard once the
    # worker is a multi-call loop, defeating the whole $5 rail.
    def plan(wcpt):
        return plan_study_calls(
            arms=("A", "B", "C", "D"),
            n_train=31,
            n_heldout=31,
            max_metric_calls=150,
            reflection_calls=15,
            worker_model="openai/gpt-4.1-mini",
            reflection_model="openai/gpt-5.6-terra",
            worker_calls_per_task=wcpt,
        )

    one = {g.model: g.calls for g in plan(1)}
    five = {g.model: g.calls for g in plan(5)}  # loop_max_retries=2 → 1+2*2
    assert five["openai/gpt-4.1-mini"] == 5 * one["openai/gpt-4.1-mini"]
    assert five["openai/gpt-5.6-terra"] == one["openai/gpt-5.6-terra"]  # reflection unchanged


def test_default_loop_config_fits_five_dollars():
    # The CLI's default paid config — 2 seeds, 16/16 split, mmc 64, 10 proposals,
    # loop_max_retries=1 (3 calls/eval) — must sit under the $5 cap with real margin.
    # Pins the headline budget claim in the script docstring against future drift.
    groups = plan_study_calls(
        arms=("A", "B", "C", "D"),
        n_train=16,
        n_heldout=16,
        max_metric_calls=64,
        reflection_calls=10,
        worker_model="openai/gpt-4.1-mini",
        reflection_model="openai/gpt-5.6-terra",
        n_seeds=2,
        worker_calls_per_task=1 + 2 * 1,
    )
    assert estimate_usd(groups) < 3.0  # ≈ $2.92, comfortably under the $5 ceiling


def test_token_shapes_price_output_at_the_enforced_cap():
    # The (in, out) shapes must use the ENFORCED max_tokens as their output size, so
    # the priced cost and the live run's hard output cap are the same number. If they
    # drifted, the guard would price a cheaper/dearer run than actually executes.
    assert WORKER_TOKENS[1] == WORKER_MAX_OUTPUT_TOKENS
    assert REFLECTION_TOKENS[1] == REFLECTION_MAX_OUTPUT_TOKENS


def test_worker_input_shape_covers_evolved_gene_prompts():
    # The worker input estimate must cover more than the seed-gene prompts (measured
    # 545/709/771): GEPA rewrites the genes, and an evolved instruction embeds into
    # every later worker prompt. The shape must comfortably cover a realistic evolved
    # gene (~360 tokens) plus the worst-case refine payload (draft + critique, each
    # ≤ the output cap) — else the estimate under-counts the arms that actually evolve.
    assert WORKER_TOKENS[0] >= 771 + 360  # seed refine worst-case + a paragraph-scale gene


def test_pricing_table_has_the_study_models():
    for m in ("openai/gpt-4.1-mini", "openai/gpt-5.6-terra", "openai/gpt-5.6-sol"):
        assert m in MODEL_PRICES
        inp, out = MODEL_PRICES[m]
        assert inp > 0 and out > 0
