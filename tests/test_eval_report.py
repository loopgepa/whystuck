"""Tests for whystuck.eval.report — the validation-study report generator.

Pure Markdown rendering from per-seed run dicts + meta. Deterministic (the
timestamp is passed in via meta, never read from the clock), so offline.
"""

from __future__ import annotations

from whystuck.eval.report import render_report

_META = {
    "dataset": "IFBench",
    "n_train": 31,
    "n_heldout": 31,
    "worker_model": "openai/gpt-4.1-mini",
    "reflection_model": "openai/gpt-5.6-terra",
    "max_metric_calls": 150,
    "seeds": [0, 1],
    "estimated_usd": 3.43,
    "generated": "2026-07-22",
    "loop_max_retries": 1,
    "worker_max_output_tokens": 320,
}


def _arm(evolvable, seed_h, opt_h):
    return {
        "evolvable_genes": list(evolvable),
        "seed_heldout": seed_h,
        "opt_heldout": opt_h,
        "delta": round(opt_h - seed_h, 4),
        "variant_id": None if not evolvable else "variant-x",
    }


def test_output_cap_caveat_is_task_set_aware():
    # Regression: the output-cap caveat used to hardcode an IFBench-specific sentence
    # ("The IFBench subset used here has only short-output constraints") that fired on
    # every task set — factually wrong on a code/bcb report, where the cap is larger
    # precisely because solutions are long. It must now describe the actual task set.
    ifb = render_report([], meta={**_META, "task_set": "ifbench"})
    assert "short outputs" in ifb
    assert "Code solutions" not in ifb

    for ts in ("code", "bcb"):
        code = render_report([], meta={**_META, "task_set": ts})
        assert "Code solutions can be long" in code
        assert "IFBench subset used here has only short-output constraints" not in code


def _runs_where_d_beats_b():
    # 2 seeds; D clearly > B on held-out, A flat, C ~ A.
    return [
        {
            "A": _arm([], 0.40, 0.40),
            "B": _arm(["generate"], 0.40, 0.55),
            "C": _arm(["generate", "ensure_correct"], 0.40, 0.41),
            "D": _arm(["generate", "ensure_correct"], 0.40, 0.80),
        },
        {
            "A": _arm([], 0.42, 0.42),
            "B": _arm(["generate"], 0.42, 0.57),
            "C": _arm(["generate", "ensure_correct"], 0.42, 0.42),
            "D": _arm(["generate", "ensure_correct"], 0.42, 0.82),
        },
    ]


def test_method_describes_the_self_refine_loop():
    # The report must state HOW B differs from D mechanistically: the worker is a
    # generate→critique→refine loop whose critique is driven by the verify gene,
    # which only D evolves. Without this, a reader can't tell why loop-level beats
    # prompt-only. Pins that the loop (and its retry depth) surfaces in Method.
    md = render_report([], meta={**_META, "loop_max_retries": 2})
    low = md.lower()
    assert "critique" in low and "refine" in low
    assert "2 refine round" in low  # the configured retry depth is reported


def test_method_loop_depth_reflects_meta():
    # A different loop_max_retries must change the reported depth (no hard-coded 1).
    md = render_report([], meta={**_META, "loop_max_retries": 3})
    assert "3 refine round" in md.lower()


def test_pending_report_has_method_and_no_verdict():
    md = render_report([], meta=_META)
    assert "# " in md  # a title
    assert "## Method" in md
    assert "IFBench" in md and "gpt-4.1-mini" in md
    assert "Pending" in md  # results not yet available
    for arm in ("A", "B", "C", "D"):
        assert arm in md


def test_full_report_supported_verdict_when_d_beats_b():
    md = render_report(_runs_where_d_beats_b(), meta=_META)
    assert "SUPPORTED" in md
    # D mean opt = (0.80+0.82)/2 = 0.81 ; B mean = (0.55+0.57)/2 = 0.56
    assert "0.81" in md
    assert "0.56" in md


def test_report_flags_ceiling_when_seed_baseline_saturated():
    # If the seed baseline already scores ~1.0 there is no headroom for ANY arm, so
    # a mechanical NOT SUPPORTED verdict is misleading — the report must flag the
    # ceiling and call the run inconclusive-by-saturation.
    saturated = [{a: _arm([], 0.97, 0.97) for a in ("A", "B", "C", "D")} for _ in range(5)]
    md = render_report(saturated, meta=_META)
    low = md.lower()
    assert "ceiling" in low or "saturat" in low
    assert "headroom" in low


def test_report_no_ceiling_note_when_seed_has_headroom():
    # A healthy seed (0.4) with real movement must NOT trigger the ceiling/floor flag.
    # (Key on "headroom" — unique to the note — since the word "ceiling" also appears
    # in the unrelated reflection-budget caveat.)
    md = render_report(_runs_where_d_beats_b(), meta=_META)
    assert "headroom" not in md.lower()
    assert "ceiling effect" not in md.lower() and "floor effect" not in md.lower()


def test_single_seed_is_inconclusive_never_supported():
    # A single seed has zero measured noise; even a positive D>B gap must NOT be
    # reported SUPPORTED (the overclaim bug this guard prevents).
    runs = [
        {
            "B": _arm(["generate"], 0.50, 0.501),
            "D": _arm(["generate", "ensure_correct"], 0.50, 0.502),  # D>B by a hair
        }
    ]
    md = render_report(runs, meta={**_META, "seeds": [0]})
    assert "## Verdict — INCONCLUSIVE" in md
    assert "## Verdict — SUPPORTED" not in md
    assert "single seed" in md or "≥2 seeds" in md


def test_two_seed_null_result_is_not_supported():
    runs = [
        {"B": _arm(["generate"], 0.5, 0.7), "D": _arm(["gen", "ver"], 0.5, 0.7)},
        {"B": _arm(["generate"], 0.4, 0.6), "D": _arm(["gen", "ver"], 0.4, 0.6)},  # D==B
    ]
    md = render_report(runs, meta=_META)
    assert "NOT SUPPORTED" in md


def test_inconclusive_when_gap_within_pooled_noise():
    # B=[0.60,0.50] (μ0.55, σ0.05); D=[0.62,0.56] (μ0.59, σ0.03).
    # gap 0.04 < summed noise 0.08 → INCONCLUSIVE, not SUPPORTED.
    runs = [
        {"B": _arm(["generate"], 0.4, 0.60), "D": _arm(["gen", "ver"], 0.4, 0.62)},
        {"B": _arm(["generate"], 0.4, 0.50), "D": _arm(["gen", "ver"], 0.4, 0.56)},
    ]
    md = render_report(runs, meta=_META)
    assert "## Verdict — INCONCLUSIVE" in md


def test_delta_column_shows_signed_value():
    # opt (0.30) worse than seed (0.40) → the Δ column must render "-0.10".
    runs = [
        {"B": _arm(["generate"], 0.40, 0.30), "D": _arm(["gen", "ver"], 0.40, 0.30)},
        {"B": _arm(["generate"], 0.40, 0.30), "D": _arm(["gen", "ver"], 0.40, 0.30)},
    ]
    md = render_report(runs, meta=_META)
    assert "-0.10" in md


def test_partial_arm_set_does_not_crash_and_is_inconclusive():
    # C and D entirely absent → pending rows for them, INCONCLUSIVE verdict, no KeyError.
    runs = [{"A": _arm([], 0.4, 0.4), "B": _arm(["generate"], 0.4, 0.6)}]
    md = render_report(runs, meta=_META)
    assert "INCONCLUSIVE" in md
    assert "Pending" in md  # C/D rows


def test_report_states_seed_count_and_models():
    md = render_report(_runs_where_d_beats_b(), meta=_META)
    assert "2" in md  # seed count
    assert "openai/gpt-5.6-terra" in md
    assert "150" in md  # max_metric_calls
