"""Tests for the scripts/run_validation_study.py CLI — offline paths only.

Covers the safety-critical control flow (dry run, budget refusal, --yes without a
key). The live run is deliberately not exercised here — it's the one paid step.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import re

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
    # What this test is for is the PARSE, so it raises the ceiling to reach the dry-run.
    # 3 seeds × ABCD used to fit the $5 default at ≈$4.38; raising
    # REFLECTION_MAX_OUTPUT_TOKENS from 1500 to 3000 (so a reasoning model has room to emit
    # content after reasoning — the empty-completion defect) moves it to ≈$5.73, so the
    # guard now refuses it at the default. That refusal is correct and is covered by
    # test_four_seeds_default_cap_is_refused; it must not be what makes this test pass.
    code, out, _ = _run(["--seeds", "0,1,2", "--max-usd", "6"])
    assert code == 0
    assert "seeds=[0, 1, 2]" in out


def test_four_seeds_default_cap_is_refused():
    # At the current reflection cap even 3 seeds over all arms exceeds the $5 default
    # (≈$5.73), and 4 seeds exceeds it by more — either way the guard's job is to refuse
    # pre-flight rather than start spending and discover the ceiling mid-run.
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


# --- hand-written gene overrides (--seed-gene) --------------------------------
#
# The verifier ablation (C1) needs author-supplied gene text: does a BETTER
# hand-written verify gene move held-out at all? Text comes from a FILE, not the
# command line — a rubric is multi-line, shell quoting mangles it silently, and the
# file is a committable record of the condition that was actually run.


def _gene_file(tmp_path, text="Count the words, then reply PASS or the exact edit."):
    p = tmp_path / "verify.txt"
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_gene_overrides_reads_the_text_from_the_file(tmp_path):
    cli = _load_cli()
    f = _gene_file(tmp_path)
    got = cli._parse_gene_overrides([f"ensure_correct={f}"], active=["generate", "ensure_correct"])
    assert got == {"ensure_correct": f.read_text(encoding="utf-8")}


def test_parse_gene_overrides_rejects_a_spec_without_an_equals(tmp_path):
    cli = _load_cli()
    with pytest.raises(ValueError, match="GENE=FILE"):
        cli._parse_gene_overrides(["ensure_correct"], active=["ensure_correct"])


def test_parse_gene_overrides_rejects_a_gene_absent_from_this_run(tmp_path):
    # `actuator` exists only under --actuator. Accepting it there would run the
    # UN-ablated condition and bill for it, so name the genes that are actually live.
    cli = _load_cli()
    f = _gene_file(tmp_path)
    with pytest.raises(ValueError, match="actuator"):
        cli._parse_gene_overrides([f"actuator={f}"], active=["generate", "ensure_correct"])


def test_parse_gene_overrides_names_the_available_genes_in_the_error(tmp_path):
    cli = _load_cli()
    f = _gene_file(tmp_path)
    with pytest.raises(ValueError, match="ensure_correct"):
        cli._parse_gene_overrides([f"typo={f}"], active=["generate", "ensure_correct"])


def test_parse_gene_overrides_rejects_a_missing_file(tmp_path):
    cli = _load_cli()
    with pytest.raises(ValueError, match="no such file"):
        cli._parse_gene_overrides(
            [f"ensure_correct={tmp_path / 'nope.txt'}"], active=["ensure_correct"]
        )


def test_parse_gene_overrides_rejects_a_blank_file(tmp_path):
    cli = _load_cli()
    f = _gene_file(tmp_path, text="   \n\n")
    with pytest.raises(ValueError, match="blank"):
        cli._parse_gene_overrides([f"ensure_correct={f}"], active=["ensure_correct"])


def test_parse_gene_overrides_rejects_the_same_gene_twice(tmp_path):
    # Last-wins would silently pick one of two conditions the operator asked for.
    cli = _load_cli()
    a, b = _gene_file(tmp_path, "first"), tmp_path / "second.txt"
    b.write_text("second", encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        cli._parse_gene_overrides(
            [f"ensure_correct={a}", f"ensure_correct={b}"], active=["ensure_correct"]
        )


def test_seed_gene_dry_run_reports_the_condition(tmp_path):
    # The operator must be able to see WHICH text they are about to pay to measure.
    f = _gene_file(tmp_path)
    code, out, _ = _run(["--arms", "A", "--seeds", "0", "--seed-gene", f"ensure_correct={f}"])
    assert code == 0
    assert "ensure_correct" in out
    assert "Count the words" in out


def test_seed_gene_for_an_inactive_gene_is_refused_before_spending(tmp_path):
    f = _gene_file(tmp_path)
    code, _, err = _run(["--arms", "A", "--seeds", "0", "--seed-gene", f"actuator={f}"])
    assert code == 2
    assert "actuator" in err


def test_seed_gene_does_not_change_the_cost_estimate(tmp_path):
    # Same prompts, same call counts: only the text changes. A different estimate would
    # mean the flag altered the loop shape, which is not what an ablation may do.
    f = _gene_file(tmp_path)
    base = _estimate_from(_run(["--arms", "A", "--seeds", "0"])[1])
    with_override = _estimate_from(
        _run(["--arms", "A", "--seeds", "0", "--seed-gene", f"ensure_correct={f}"])[1]
    )
    assert base == with_override


# --- Multi-provider pre-spend gates (D3, the cross-provider critic) --------------------
#
# Both were written after a near miss on 2026-08-06: the D3 critic's provider refused this
# host at its edge, and the ONLY thing preventing a live attempt was that its key happened
# to be unset. Had it been set, the run would have billed the worker's provider in full and
# then died on the first critique call, leaving a partial cassette and no result.


def _probe(status=None, error=None):
    """A `probe_provider` stand-in, so no test touches the network."""
    return lambda provider, key, **kw: (status == 200, error or f"HTTP {status}")


def test_a_second_providers_missing_key_is_caught_before_spending(monkeypatch):
    """The old gate checked OPENAI_API_KEY only, so a groq critic passed it."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    code, _, err = _run(["--yes", "--critique-model", "groq/llama-3.3-70b-versatile"])
    assert code == 3
    assert "GROQ_API_KEY" in err and "groq" in err


def test_the_workers_key_alone_still_satisfies_a_single_provider_run(monkeypatch):
    """The new gate must not start demanding keys a run will never use."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cli = _load_cli()
    monkeypatch.setattr(cli, "probe_provider", _probe(200))
    monkeypatch.setattr(cli, "_run_live", lambda *a, **k: 0)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        assert cli.main(["--yes"]) == 0


def test_a_present_but_unusable_key_is_caught_before_spending(monkeypatch):
    """Presence is not reachability. This is the actual near miss."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    cli = _load_cli()
    calls = []

    def probe(provider, key, **kw):
        calls.append(provider)
        return (provider != "groq", "HTTP 200" if provider != "groq" else "HTTP 403")

    monkeypatch.setattr(cli, "probe_provider", probe)
    monkeypatch.setattr(cli, "_run_live", lambda *a, **k: pytest.fail("must not spend"))
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["--yes", "--critique-model", "groq/llama-3.3-70b-versatile"])
    assert code == 4
    assert "Nothing was spent" in err.getvalue()
    assert "groq" in calls and "openai" in calls, (
        "every provider must be probed, not just the odd one"
    )


def test_skip_preflight_bypasses_the_probe_and_nothing_else(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cli = _load_cli()
    monkeypatch.setattr(cli, "probe_provider", lambda *a, **k: pytest.fail("probe must not run"))
    monkeypatch.setattr(cli, "_run_live", lambda *a, **k: 0)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        assert cli.main(["--yes", "--skip-preflight"]) == 0


def test_skip_preflight_does_not_bypass_the_missing_key_gate(monkeypatch):
    """An escape hatch for the probe must not become an escape hatch for everything."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    code, _, err = _run(
        ["--yes", "--skip-preflight", "--critique-model", "groq/llama-3.3-70b-versatile"]
    )
    assert code == 3 and "GROQ_API_KEY" in err


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (401, "credential rejected"),
        (403, "refused before auth"),
        (429, "quota or rate limit exhausted"),
    ],
)
def test_the_probe_distinguishes_the_failures_you_act_on_differently(monkeypatch, code, expected):
    """403 and 429 need opposite responses -- fix the network vs top up the account -- and a
    mid-run exception conflates them. This distinction is the reason the probe reports a string
    rather than a bool."""
    import urllib.error

    cli = _load_cli()

    def _raise(*a, **k):
        raise urllib.error.HTTPError("u", code, "", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", _raise)
    ok, status = cli.probe_provider("groq", "k")
    assert not ok and expected in status


def test_an_unknown_provider_is_not_silently_treated_as_broken(monkeypatch):
    cli = _load_cli()
    ok, status = cli.probe_provider("some-new-vendor", "k")
    assert ok and "not checked" in status


def test_the_probe_does_not_identify_itself_as_python_urllib(monkeypatch):
    """A false-negative preflight is worse than no preflight.

    Measured on 2026-08-06: Groq's edge returned 403 to `Python-urllib/3.12` and 200 to curl and to
    litellm from the same host, seconds apart. The probe therefore reported the provider unusable
    while the run's actual client could reach it perfectly — which blocks a good run and, worse,
    teaches the operator to pass --skip-preflight, disabling the check that catches the real thing.
    """
    import urllib.error

    cli = _load_cli()
    seen = {}

    def _capture(request, *a, **k):
        seen["ua"] = request.get_header("User-agent")
        raise urllib.error.HTTPError("u", 500, "", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", _capture)
    cli.probe_provider("groq", "k")
    assert seen["ua"] and "urllib" not in seen["ua"].lower(), (
        f"probe identified itself as {seen['ua']!r}; edges block that string"
    )


def test_the_probe_sends_the_credential_each_provider_expects(monkeypatch):
    """Anthropic takes x-api-key, not a Bearer token. Sending the wrong one reads as a dead key."""
    import urllib.error

    cli = _load_cli()
    seen = {}

    def _capture(request, *a, **k):
        seen.update({k.lower(): v for k, v in request.header_items()})
        raise urllib.error.HTTPError("u", 500, "", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", _capture)
    cli.probe_provider("anthropic", "secret")
    assert seen.get("X-api-key".lower()) == "secret"
    assert "authorization" not in seen


# --- verdict/prose splice (E1): pricing and the incompatible-flag refusal ---------


def test_verdict_model_prices_the_extra_critique_call():
    # The splice issues TWO LLM calls per critique step (one for the verdict, one for
    # the prose) on the same prompt. If the planner keeps charging one, the pre-flight
    # guard stops being a ceiling -- which is the only property it has.
    from whystuck.eval.cost import estimate_usd, plan_study_calls

    common = _BASE + ["--loop-max-retries", "1"]
    _, plain, _ = _run(common)
    _, spliced, _ = _run([*common, "--verdict-model", "openai/gpt-4.1-mini"])
    assert _estimate_from(spliced) > _estimate_from(plain)

    expected = estimate_usd(
        plan_study_calls(
            arms=("A", "B"),
            n_train=5,
            n_heldout=5,
            max_metric_calls=20,
            reflection_calls=3,
            worker_model="openai/gpt-4.1-mini",
            reflection_model="openai/gpt-5.6-terra",
            worker_calls_per_task=1 + 3 * 1,  # generate + (verdict + prose + refine)
        )
    )
    assert f"${expected:.4f}" in spliced


def test_verdict_model_is_refused_with_the_oracle_gate():
    # The oracle gate makes no critique LLM call at all, so there is no verdict to
    # route anywhere. Accepting the pair would silently ignore one of the two flags
    # and record a run whose metadata claims a splice that never happened.
    code, _, err = _run(
        [
            "--arms",
            "A",
            "--seeds",
            "0",
            "--verdict-model",
            "openai/gpt-4.1-mini",
            "--oracle-critique",
        ]  # fmt: skip
    )
    assert code == 2
    assert "verdict-model" in err and "oracle-critique" in err


def test_unknown_verdict_model_rejected():
    code, _, err = _run(["--arms", "A", "--seeds", "0", "--verdict-model", "openai/nope"])
    assert code == 2
    assert "MODEL_PRICES" in err


def test_an_expensive_verdict_model_raises_the_estimate_even_with_a_cheap_critic():
    # The conservative rule prices every worker-side call at the priciest model any of
    # them can route to. The verdict model is one of those routes, so leaving it out of
    # the max makes the guard an UNDER-estimate for a cheap-prose / expensive-verdict
    # cell -- and an under-estimating ceiling is not a ceiling.
    common = _BASE + ["--loop-max-retries", "1"]
    _, cheap, _ = _run([*common, "--verdict-model", "openai/gpt-4.1-mini"])
    _, pricey, _ = _run([*common, "--verdict-model", "openai/gpt-5.6-luna"])
    assert _estimate_from(pricey) > _estimate_from(cheap)


# --- --temperature must reach EVERY sampled surface, not just the worker ---------


def _wire_and_capture(argv, tmp_path, monkeypatch):
    """Run main() with the provider and the study faked, returning run_study's kwargs.

    Nothing here touches the network: `litellm_complete` is replaced by a factory whose
    callables record the kwargs they are handed, and `run_study` is replaced by a capture.
    """
    cli = _load_cli()
    recorded: dict[str, list] = {}

    def fake_litellm_complete(model, **_factory_kw):
        def call(prompt, **kw):
            recorded.setdefault(model, []).append(kw)
            return "PASS"

        return call

    captured: dict[str, object] = {}

    def fake_run_study(_store, **kw):
        captured.update(kw)
        raise SystemExit(0)

    # Both names are imported INSIDE main(), so the patch has to land on the defining
    # modules rather than on the CLI module's namespace.
    import whystuck.eval.llm_worker as lw
    import whystuck.eval.study as st

    monkeypatch.setattr(lw, "litellm_complete", fake_litellm_complete)
    monkeypatch.setattr(st, "run_study", fake_run_study)
    # Record mode, not replay: the cassette wrapper must actually reach the fake provider
    # for its kwargs to be observable. No network is possible -- the provider is a stub --
    # and the preflight probe is skipped so nothing dials out.
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-not-a-real-key")
    with contextlib.suppress(SystemExit), contextlib.redirect_stdout(io.StringIO()):
        cli.main([
            *argv, "--yes", "--skip-preflight",
            "--record", str(tmp_path / "c.json"),
            "--out", str(tmp_path / "o.json"),
            "--report", str(tmp_path / "r.md"),
        ])  # fmt: skip
    return captured, recorded


def test_temperature_reaches_the_critique_and_verdict_calls_not_only_the_worker(
    tmp_path, monkeypatch
):
    # --temperature was wired onto the worker callable alone. The critique and verdict
    # calls therefore ran at the provider default while the run's metadata recorded a
    # pinned temperature -- a flag reported as applied that was not, which is this
    # series' own subject matter appearing in its harness. It was caught only because
    # two calls on one prompt at temperature 0 returned different text 30 times out of 30.
    captured, recorded = _wire_and_capture(
        [
            "--arms",
            "A",
            "--seeds",
            "0",
            "--n-heldout",
            "2",
            "--n-train",
            "2",
            "--temperature",
            "0",
            "--worker-model",
            "openai/gpt-4.1-mini",
            "--critique-model",
            "openai/gpt-5.6-luna",
            "--verdict-model",
            "openai/gpt-4.1-nano",
        ],  # fmt: skip
        tmp_path,
        monkeypatch,
    )
    for role in ("critique_complete", "verdict_complete"):
        fn = captured.get(role)
        assert fn is not None, f"{role} was not wired at all"
        fn("a prompt")
    assert recorded, "no calls were recorded"
    for model, calls in recorded.items():
        assert calls, f"{model} never called"
        assert all(kw.get("temperature") == 0.0 for kw in calls), (
            f"{model} was called without the pinned temperature: {calls}"
        )


def test_without_temperature_no_sampling_kwargs_are_added_anywhere(tmp_path, monkeypatch):
    # The invariant that keeps every recorded cassette replayable: absent the flags, the
    # call is unchanged, so its hashed kwargs are empty and existing keys still resolve.
    captured, recorded = _wire_and_capture(
        [
            "--arms",
            "A",
            "--seeds",
            "0",
            "--n-heldout",
            "2",
            "--n-train",
            "2",
            "--worker-model",
            "openai/gpt-4.1-mini",
            "--critique-model",
            "openai/gpt-5.6-luna",
        ],  # fmt: skip
        tmp_path,
        monkeypatch,
    )
    captured["critique_complete"]("a prompt")
    assert all(kw == {} for calls in recorded.values() for kw in calls)


def test_the_pre_fix_exhibit_shows_the_flag_never_reached_the_two_surfaces():
    """Paper III's defect 10 claims two calls on one prompt at temperature 0 differed 30/30.

    Derived from the exhibit rather than asserted. The cassette cannot be replayed under HEAD --
    the fix puts the sampling kwargs into the hashed key, so every key in it is stale -- but the
    claim does not need a replay: if NO `critic` completion equals ANY `verdict` completion, then
    a fortiori none matched its own pair, and 30 requested-deterministic prompts produced 60
    distinct answers.
    """
    exhibit = pathlib.Path("tests/fixtures/evidence/study-splice-G1-pre-tempfix.json")
    entries = json.loads(exhibit.read_text(encoding="utf-8"))["entries"]
    critic = [v["output"] for v in entries.values() if v.get("name") == "critic"]
    verdict = [v["output"] for v in entries.values() if v.get("name") == "verdict"]
    assert len(critic) == len(verdict) == 30, (
        f"the exhibit should hold 30 completions per surface, found {len(critic)}/{len(verdict)}"
    )
    assert not set(critic) & set(verdict), (
        "a critic completion matches a verdict completion, so the paper's 30-of-30 claim is no "
        "longer supported by this exhibit"
    )
    assert len(set(critic) | set(verdict)) == 60, "the 60 completions are not all distinct"

    claimed = re.search(
        r"returned different text thirty times out of thirty",
        pathlib.Path("docs/papers/p3-silent-no-op/main.tex").read_text(encoding="utf-8"),
    )
    assert claimed, "Paper III no longer states the 30-of-30 claim this exhibit backs"


#: The Groq splice registration's RESULT section. Pre-registrations are not scraped by
#: `verify_paper_claims`, which reads the papers, so a number written into a deviation record has
#: nothing pinning it -- and a deviation record is exactly where a convenient number would do the
#: most damage. These four are derived here instead.
_G1_SPEC = pathlib.Path(
    "docs/superpowers/specs/2026-08-08-groq-verdict-prose-splice-preregistration.md"
)


def test_the_g1_diagonal_is_inert_by_measurement_not_construction():
    """G1's four recorded facts, against the artifact and the words claiming them.

    The registration predicted the diagonal would be inert *by construction* once
    `--temperature 0` reached both surfaces. It is not: one prompt of thirty still returns
    different text on the two surfaces, because temperature 0 is a sampling setting and not a
    provider determinism guarantee. That withdrawal is the point of the section, so the test
    asserts the number is 29 rather than asserting agreement -- a guard written as
    `assert all_identical` would have to be deleted to record the truth, and a guard you delete
    to tell the truth is a guard that was pushing you to tell something else.
    """
    entries = json.loads(
        pathlib.Path("tests/fixtures/cassettes/study-splice-G1.json").read_text(encoding="utf-8")
    )["entries"]
    critic = [v["output"] for v in entries.values() if v.get("name") == "critic"]
    verdict = [v["output"] for v in entries.values() if v.get("name") == "verdict"]
    assert len(critic) == len(verdict) == 30

    # 30 distinct prompts per surface, so a shared completion is a matched pair and the union
    # size counts them: 31 distinct means 29 of 30 pairs agreed.
    assert len(set(critic)) == len(set(verdict)) == 30, "a surface repeated a completion"
    agreed = 60 - len(set(critic) | set(verdict))
    spec = _G1_SPEC.read_text(encoding="utf-8")

    assert "**31 distinct**" in spec and "29 of the 30 prompts" in spec, (
        "the G1 registration no longer states the diagonal agreement in the form this guard "
        "reads; update both together rather than letting the record and the artifact part"
    )
    assert agreed == 29, f"the recorded diagonal now agrees on {agreed} of 30, not 29"

    # The registered INVALID rule reports 0 fallbacks -- because the prose source never passes,
    # not because the surfaces agree. Both halves are asserted, since the section's claim is
    # that the first is true and uninformative.
    passes = [t for t in critic if t.strip().upper().startswith("PASS")]
    assert not passes, "the 8B critic now emits PASS, so the vacuity argument needs rewriting"
    assert "**0 of 30**" in spec and "vacuous on this worker" in spec

    runs = json.loads(
        pathlib.Path("docs/validation-study-splice-G1.json").read_text(encoding="utf-8")
    )["runs"]
    seeds = [r["A"]["seed_heldout"] for r in runs]
    assert [r["A"]["opt_heldout"] for r in runs] == seeds, "--arms A optimized something"
    assert f"{sum(seeds) / len(seeds):.4f}" == "0.7033", "G1's p0-gate mean moved"
    assert "**0.7033**" in spec, "the registration no longer states the mean it was gated on"
    assert 0.15 < sum(seeds) / len(seeds) < 0.95, (
        "the gate would now trip; the record says it did not"
    )
