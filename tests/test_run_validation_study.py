"""Tests for the scripts/run_validation_study.py CLI — offline paths only.

Covers the safety-critical control flow (dry run, budget refusal, --yes without a
key). The live run is deliberately not exercised here — it's the one paid step.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_validation_study.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_validation_study", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(argv):
    cli = _load_cli()
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


def test_dry_run_prints_estimate_and_does_not_spend():
    code, out, _ = _run([])  # defaults: dry run, no --yes
    assert code == 0
    assert "Estimated total" in out
    assert "Dry run only" in out


def test_over_budget_is_refused():
    code, _, err = _run(["--max-usd", "0.0001"])
    assert code == 2
    assert "REFUSED" in err


def test_yes_without_api_key_errors(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code, _, err = _run(["--yes"])
    assert code == 3
    assert "OPENAI_API_KEY" in err


def test_dry_run_does_not_require_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code, _, _ = _run([])
    assert code == 0  # dry run never touches the key


@pytest.mark.parametrize("arms", ["ABCD", "AB", "AC"])
def test_estimate_stays_under_default_ceiling_for_real_config(arms):
    code, out, _ = _run(["--arms", arms])
    assert code == 0
    assert "Dry run only" in out  # within budget → reached the dry-run exit


@pytest.mark.parametrize("bad", ["abcd", "ABCE", "Z"])
def test_invalid_arms_rejected_before_spending(bad):
    # A typo'd --arms must fail loud at the cheap dry-run step, not plan to $0.
    code, out, err = _run(["--arms", bad])
    assert code == 2
    assert "unknown arm" in err.lower()
    assert "Estimated total" not in out  # refused before even printing a plan


def test_unknown_model_rejected_with_friendly_error():
    code, _, err = _run(["--worker-model", "openai/nope"])
    assert code == 2
    assert "MODEL_PRICES" in err or "no price" in err


def test_default_is_two_seeds():
    code, out, _ = _run([])
    assert code == 0
    assert "seeds=[0, 1]" in out


def test_seeds_flag_parses_multiple_values():
    # 3 seeds × ABCD × Terra with the 5-call/eval loop is well over the $5 cap, so
    # raise --max-usd to reach the dry-run and exercise the parse (the refusal
    # itself is tested below).
    code, out, _ = _run(["--seeds", "0,1,2", "--max-usd", "20"])
    assert code == 0
    assert "seeds=[0, 1, 2]" in out


def test_three_seeds_default_cap_is_refused():
    # Documents that 3 seeds over all arms exceeds the $5 default — the guard's job.
    code, _, err = _run(["--seeds", "0,1,2"])
    assert code == 2
    assert "REFUSED" in err


def test_empty_seeds_is_refused():
    code, _, err = _run(["--seeds", ""])
    assert code == 2
    assert "at least one" in err


def test_cli_estimate_matches_independent_computation():
    # The figure the CLI prints must equal what the guard/estimator computes.
    from whystuck.eval.cost import estimate_usd, plan_study_calls

    code, out, _ = _run(
        [
            "--arms",
            "AB",
            "--seeds",
            "0",
            "--n-train",
            "5",
            "--n-heldout",
            "5",
            "--max-metric-calls",
            "20",
            "--reflection-calls",
            "3",
            "--loop-max-retries",
            "2",
        ]
    )
    expected = estimate_usd(
        plan_study_calls(
            arms=("A", "B"),
            n_train=5,
            n_heldout=5,
            max_metric_calls=20,
            reflection_calls=3,
            worker_model="openai/gpt-4.1-mini",
            reflection_model="openai/gpt-5.6-terra",
            worker_calls_per_task=1 + 2 * 2,  # loop makes 1 + 2*retries calls/eval
        )
    )
    assert code == 0
    assert f"${expected:.4f}" in out


def test_plan_gene_prices_one_extra_worker_call_per_eval():
    # --plan-gene adds a plan step, so the loop makes 2 + 2*retries calls/eval;
    # the estimate must reflect the +1 or the $5 guard is priced too low.
    from whystuck.eval.cost import estimate_usd, plan_study_calls

    common = [
        "--arms", "AB", "--seeds", "0", "--n-train", "5", "--n-heldout", "5",
        "--max-metric-calls", "20", "--reflection-calls", "3", "--loop-max-retries", "1",
    ]  # fmt: skip
    _, plain, _ = _run(common)
    _, planned, _ = _run([*common, "--plan-gene"])
    expected_plan = estimate_usd(
        plan_study_calls(
            arms=("A", "B"),
            n_train=5,
            n_heldout=5,
            max_metric_calls=20,
            reflection_calls=3,
            worker_model="openai/gpt-4.1-mini",
            reflection_model="openai/gpt-5.6-terra",
            worker_calls_per_task=2 + 2 * 1,  # +1 plan step
        )
    )
    assert f"${expected_plan:.4f}" in planned
    assert planned != plain  # the plan step is priced, not ignored


def test_feedback_stats_flag_parses_and_dry_runs():
    # The feedback lever costs no extra LLM calls (stats computed locally), so it
    # must dry-run cleanly and not change the estimate.
    code, out, _ = _run(["--arms", "A", "--seeds", "0", "--feedback-stats"])
    assert code == 0
    assert "Estimated total" in out
