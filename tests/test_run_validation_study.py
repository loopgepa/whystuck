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
    # 3 seeds × ABCD at the corrected gpt-4.1-mini price (≈ $4.38) fits under the $5
    # default, so no --max-usd bump is needed to reach the dry-run and exercise the
    # parse (the refusal itself is tested below with a genuinely over-budget config).
    code, out, _ = _run(["--seeds", "0,1,2"])
    assert code == 0
    assert "seeds=[0, 1, 2]" in out


def test_four_seeds_default_cap_is_refused():
    # 3 seeds now fits (≈ $4.38); 4 seeds over all arms (≈ $5.84) exceeds the $5
    # default — the guard's job to refuse pre-flight.
    code, _, err = _run(["--seeds", "0,1,2,3"])
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


_BASE = ["--arms", "AB", "--seeds", "0", "--n-train", "5", "--n-heldout", "5",
         "--max-metric-calls", "20", "--reflection-calls", "3"]  # fmt: skip


def _estimate_from(out: str) -> float:
    for line in out.splitlines():
        if line.startswith("Estimated total"):
            return float(line.split("$")[1].split()[0])
    raise AssertionError("no estimate printed")


def test_oracle_critique_prices_fewer_calls():
    # The oracle gate makes NO critique LLM call, so 1 + retries calls/eval instead
    # of 1 + 2*retries — the estimate must drop, or the ceiling arm is mispriced.
    common = _BASE + ["--loop-max-retries", "2"]
    _, plain, _ = _run(common)
    _, oracle, _ = _run([*common, "--oracle-critique"])
    assert _estimate_from(oracle) < _estimate_from(plain)


def test_critique_model_priced_conservatively_at_the_pricier_model():
    # A pricier cross-family critic must raise the estimate (worker-side calls priced
    # at the more expensive model), never silently keep the cheaper worker price.
    common = _BASE + ["--loop-max-retries", "1"]
    _, plain, _ = _run(common)
    _, crit, _ = _run([*common, "--critique-model", "openai/gpt-5.6-luna"])
    assert _estimate_from(crit) > _estimate_from(plain)


def test_unknown_critique_model_rejected():
    code, _, err = _run(["--arms", "A", "--seeds", "0", "--critique-model", "openai/nope"])
    assert code == 2
    assert "MODEL_PRICES" in err


def test_replay_bypasses_the_api_key_gate(monkeypatch):
    # --replay reproduces offline: it must reach the run path WITHOUT an API key
    # (no spend possible), unlike --yes which errors without one.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cli = _load_cli()
    called = {}

    def _fake_run_live(*a, **k):
        called["hit"] = True
        return 0

    monkeypatch.setattr(cli, "_run_live", _fake_run_live)
    code = cli.main(["--arms", "A", "--seeds", "0", "--replay"])
    assert code == 0
    assert called.get("hit"), "replay did not reach the run path"


def test_yes_without_key_still_errors_but_replay_does_not(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cli = _load_cli()
    assert cli.main(["--arms", "A", "--seeds", "0", "--yes"]) == 3  # --yes still needs a key


# --- feedback-actionability lever (D2'): --feedback-level {lo,mid,hi} -----------


def test_feedback_level_defaults_to_mid():
    # Pin the default directly (the dry-run path never prints the level), so a changed
    # default or a deleted flag is caught — not a tautology on unrelated dry-run text.
    cli = _load_cli()
    assert cli._parse_args(["--arms", "A", "--seeds", "0"]).feedback_level == "mid"


def test_feedback_level_lo_is_exempt_from_the_multi_task_set_restriction():
    # Only `hi` needs a multi task set (its edit directives are mc-specific); `lo` is a
    # constant string and must run on any task set. Proves lo's oracle-only guard is
    # independent of the task-set guard — a future merge of the two would break this.
    code, out, _ = _run(
        ["--arms", "A", "--seeds", "0", "--oracle-critique", "--feedback-level", "lo",
         "--task-set", "ifbench"]
    )  # fmt: skip
    assert code == 0
    assert "Estimated total" in out


def test_build_oracle_feedback_lo_ignores_feedback_and_emits_bare_verdict():
    cli = _load_cli()

    def fb(_o, _t):
        return "WHICH_CONSTRAINT_PROSE"

    fn = cli._build_oracle_feedback("lo", fb)
    assert fn("any draft", object()) == "not all requirements are satisfied; revise"


def test_build_oracle_feedback_hi_appends_measured_edits_after_feedback():
    # The exact production string for the "hi" level: feedback_fn's text, then the
    # separator, then mc_edit_directives' output — in that order. Guards the closure a
    # paid run actually uses (argparse tests never reach it).
    from whystuck.eval.benchmark import BenchmarkTask
    from whystuck.eval.multi_constraint import mc_edit_directives, mc_feedback

    cli = _load_cli()
    task = BenchmarkTask(
        id="t", payload={"instruction_id_list": ["exact_word_count"], "kwargs": [{"n": 20}]}
    )
    draft = "two words"
    fn = cli._build_oracle_feedback("hi", mc_feedback)
    got = fn(draft, task)
    expected = f"{mc_feedback(draft, task)}\n\nConcrete edits:\n{mc_edit_directives(draft, task)}"
    assert got == expected
    assert "Concrete edits:" in got and "exactly 20" in got


def test_build_oracle_feedback_mid_and_unknown_return_none():
    cli = _load_cli()
    assert cli._build_oracle_feedback("mid", lambda o, t: "x") is None
    assert cli._build_oracle_feedback("anything-else", lambda o, t: "x") is None


def test_feedback_level_lo_requires_oracle_critique():
    # Off the oracle path the level is silently inert (the LLM writes the critique),
    # so spending on it would waste money on a no-op. Fail loud at the dry-run step.
    code, _, err = _run(["--arms", "A", "--seeds", "0", "--feedback-level", "lo"])
    assert code == 2
    assert "oracle" in err.lower()


def test_feedback_level_hi_requires_oracle_critique():
    code, _, err = _run(["--arms", "A", "--seeds", "0", "--feedback-level", "hi"])
    assert code == 2
    assert "oracle" in err.lower()


def test_feedback_level_hi_requires_a_multi_task_set():
    # The "hi" edit directives are multi_constraint-specific; on the single-constraint
    # ifbench set they don't exist, so hi+ifbench is a misconfig that must fail before
    # spending, not silently fall back.
    code, _, err = _run(
        ["--arms", "A", "--seeds", "0", "--oracle-critique", "--feedback-level", "hi",
         "--task-set", "ifbench"]
    )  # fmt: skip
    assert code == 2
    assert "multi" in err.lower()


def test_feedback_level_lo_and_hi_dry_run_on_multi_hard_with_oracle():
    for level in ("lo", "hi"):
        code, out, _ = _run(
            ["--arms", "A", "--seeds", "0", "--oracle-critique",
             "--task-set", "multi-hard", "--feedback-level", level]
        )  # fmt: skip
        assert code == 0, level
        assert "Estimated total" in out


def test_feedback_level_does_not_change_the_cost_estimate():
    # The repair feedback is computed locally (no extra LLM calls), so lo/mid/hi must
    # all price identically — otherwise the guard would be mis-calibrated per level.
    base = ["--arms", "AB", "--seeds", "0", "--n-train", "5", "--n-heldout", "5",
            "--max-metric-calls", "20", "--reflection-calls", "3", "--loop-max-retries", "3",
            "--oracle-critique", "--task-set", "multi-hard"]  # fmt: skip
    ests = {}
    for level in ("lo", "mid", "hi"):
        _, out, _ = _run([*base, "--feedback-level", level])
        ests[level] = _estimate_from(out)
    assert ests["lo"] == ests["mid"] == ests["hi"]
