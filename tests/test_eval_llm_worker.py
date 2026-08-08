"""Tests for whystuck.eval.llm_worker — proves text genes move the score."""

import sys
import types

import pytest

from whystuck.core.models import GeneType, LoopGene, LoopGenome
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.ifbench import ifbench_feedback, ifbench_score
from whystuck.eval.llm_worker import (
    default_build_prompt,
    keyword_score,
    litellm_complete,
    make_run_task,
    make_self_refine_run_task,
)


def _genome(skill: str) -> LoopGenome:
    return LoopGenome(
        id="bugfix",
        name="Bugfix",
        genes=[LoopGene(id="skill", type=GeneType.SKILL, name="S", content=skill)],
    )


# Fake LLM: writes a "minimal" patch only when the prompt asks for one.
def _complete(prompt: str) -> str:
    return "a minimal patch" if "minimal" in prompt.lower() else "a sprawling patch"


def _score(output: str, task: BenchmarkTask) -> bool:
    return "minimal" in output.lower()


def test_better_text_gene_yields_success():
    run_task = make_run_task(_complete, _score)
    good = run_task(_genome("Fix with the minimal patch."), BenchmarkTask(id="t1"))
    bad = run_task(_genome("Fix the bug."), BenchmarkTask(id="t1"))
    assert good.final_outcome == "success"
    assert bad.final_outcome == "failure"


def test_prompt_includes_text_genes_and_goal():
    prompt = default_build_prompt(
        _genome("Prefer minimal diffs."), BenchmarkTask(id="t1", payload={"goal": "fix login"})
    )
    assert "Prefer minimal diffs." in prompt
    assert "fix login" in prompt


def test_keyword_score_matches_declared_expectation():
    task = BenchmarkTask(id="t1", payload={"expect": "Minimal"})
    assert keyword_score("here is a minimal patch", task) is True  # case-insensitive
    assert keyword_score("here is a sprawling patch", task) is False


def test_keyword_score_accepts_any_of_a_list():
    task = BenchmarkTask(id="t1", payload={"expect_any": ["fixed", "resolved"]})
    assert keyword_score("the bug is resolved", task) is True
    assert keyword_score("the bug remains", task) is False


def test_keyword_score_without_expectation_is_not_a_free_win():
    # No declared criterion means nothing was verified — it must not score as a pass.
    assert keyword_score("anything at all", BenchmarkTask(id="t1")) is False


def test_litellm_complete_without_litellm_is_actionable(hide_import):
    # Absence is simulated rather than inherited from the environment --- see `hide_import`.
    hide_import("litellm")
    complete = litellm_complete("openai/gpt-4o-mini", max_tokens=256)
    # The error must name the real fix. litellm is now part of the `whystuck[eval]`
    # extra (added so the one paid run reproduces from a single install), so the
    # extra IS the fix — superseding the earlier state where litellm was separate.
    with pytest.raises(RuntimeError, match=r"whystuck\[eval\]"):
        complete("hi")


def test_litellm_complete_retries_an_empty_completion_then_fails_loudly(monkeypatch):
    # Measured in C2: two reflection completions came back with `content = None`, which
    # `content or ""` turned into an empty string. GEPA proposed empty gene text and on one
    # seed accepted it, so the arm carrying the loop-level treatment ran with no critique.
    # A reasoning model can exhaust max_tokens on reasoning before emitting any content, so
    # an empty completion is retried once and then raised rather than passed downstream.
    calls: list[int] = []

    class _Msg:
        def __init__(self, c):
            self.content = c

    class _Choice:
        def __init__(self, c):
            self.message = _Msg(c)

    class _Resp:
        def __init__(self, c):
            self.choices = [_Choice(c)]

    fake = types.SimpleNamespace(completion=lambda **kw: (calls.append(1), _Resp(None))[1])
    monkeypatch.setitem(sys.modules, "litellm", fake)
    complete = litellm_complete("openai/gpt-5.6-luna", max_tokens=1500)
    with pytest.raises(RuntimeError, match="empty completion"):
        complete("reflect on this")
    assert len(calls) == 2, "an empty completion must be retried exactly once before raising"

    # A retry that succeeds is used, so one transient empty response does not abort a run.
    seq = [None, "the real proposal"]
    fake.completion = lambda **kw: _Resp(seq.pop(0))
    assert litellm_complete("openai/gpt-5.6-luna", max_tokens=1500)("x") == "the real proposal"


# --- reflective-signal capture (fixes GEPA reflection starvation) -------------


def test_run_task_captures_task_input_and_worker_output():
    # The trace must carry the task instruction and the worker's real output so
    # the adapter can build a GEPA reflective dataset with actual content.
    run_task = make_run_task(lambda p: "solid", lambda o, t: 1.0)
    trace = run_task(_genome("skill"), BenchmarkTask(id="t1", payload={"goal": "do the thing"}))
    assert trace.task_summary == "do the thing"
    assert trace.task_id == "t1"
    step = next(s for s in trace.steps if s.role == "worker")
    assert step.output_summary == "solid"
    assert step.input_summary == "do the thing"


def test_run_task_stores_fractional_eval_score_and_partial_is_not_success():
    # A fractional scorer (0.5) must be stored verbatim in metadata['eval_score'],
    # and a partial result is NOT a full success.
    run_task = make_run_task(lambda p: "out", lambda o, t: 0.5)
    trace = run_task(_genome("skill"), BenchmarkTask(id="t"))
    assert trace.metadata["eval_score"] == 0.5
    assert trace.metrics.success is False
    assert trace.final_outcome == "failure"


def test_run_task_full_score_is_success():
    run_task = make_run_task(lambda p: "out", lambda o, t: 1.0)
    trace = run_task(_genome("skill"), BenchmarkTask(id="t"))
    assert trace.metrics.success is True
    assert trace.metadata["eval_score"] == 1.0


def test_run_task_feedback_hook_lands_in_metadata():
    run_task = make_run_task(
        lambda p: "out", lambda o, t: 0.0, feedback=lambda o, t: "constraint X failed"
    )
    trace = run_task(_genome("skill"), BenchmarkTask(id="t"))
    assert trace.metadata["feedback"] == "constraint X failed"


# --- self-refine loop worker (generate -> critique[verify gene] -> refine) -----


def _two_gene(generate: str, verify: str) -> LoopGenome:
    return LoopGenome(
        id="loop",
        name="L",
        genes=[
            LoopGene(id="generate", type=GeneType.SKILL, name="Generate", content=generate),
            LoopGene(id="ensure_correct", type=GeneType.VERIFIER_RUBRIC, name="V", content=verify),
        ],
    )


def _loop_complete(prompt: str) -> str:
    # A controllable 3-role LLM keyed on the prompt section it is answering:
    #  - critique step ("# Draft answer"): only a rubric containing CHECK_WHITESPACE
    #    catches the whitespace violation; a weak rubric wrongly passes it.
    #  - refine step ("# Problems to fix"): fixes the draft to a no-whitespace token.
    #  - initial generate: emits a failing draft with whitespace.
    if "# Draft answer" in prompt:
        return "FAIL: contains whitespace, remove it" if "CHECK_WHITESPACE" in prompt else "PASS"
    if "# Problems to fix" in prompt:
        return "solid"
    return "x x"


def _nw_task():
    return BenchmarkTask(
        id="t", payload={"goal": "answer", "instruction_id_list": ["format:no_whitespace"]}
    )


def test_self_refine_loops_generate_critique_refine():
    calls = []

    def spy(prompt):
        calls.append(prompt)
        return _loop_complete(prompt)

    run_task = make_self_refine_run_task(
        spy, ifbench_score, feedback=ifbench_feedback, max_retries=2
    )
    run_task(_two_gene("write it", "CHECK_WHITESPACE: verify no spaces"), _nw_task())
    # a real loop: at least generate + critique + refine happened (>1 call).
    assert len(calls) >= 3
    assert any("# Draft answer" in c for c in calls)  # a critique step ran
    assert any("# Problems to fix" in c for c in calls)  # a refine step ran


@pytest.mark.parametrize("retries", [1, 2, 3])
def test_self_refine_makes_exactly_1_plus_2r_calls_when_critique_never_passes(retries):
    # The cost guard (eval/cost.py) prices the loop as EXACTLY 1 + 2*loop_max_retries
    # LLM calls per eval. If the retry loop is off by one, the $5 guard silently
    # under-counts a real paid run. Pin the formula against the real implementation
    # (not just against itself): the CHECK_WHITESPACE rubric makes the critique key
    # on the RUBRIC text, so it FAILs every round regardless of the draft — the loop
    # runs to exhaustion, the worst case the guard must bound.
    calls = []

    def spy(prompt):
        calls.append(prompt)
        return _loop_complete(prompt)

    trace = make_self_refine_run_task(spy, ifbench_score, max_retries=retries)(
        _two_gene("write it", "CHECK_WHITESPACE: verify no spaces"), _nw_task()
    )
    assert len(calls) == 1 + 2 * retries
    assert trace.metrics.retry_count == retries  # one refine per exhausted round


def test_self_refine_rejects_zero_retries():
    # max_retries=0 would skip the critique entirely — the verify gene never enters
    # a prompt, so arm D silently collapses to arm B (the exact null the redesign
    # exists to fix). The factory must refuse it loudly, not produce a dead loop.
    with pytest.raises(ValueError, match="max_retries"):
        make_self_refine_run_task(_loop_complete, ifbench_score, max_retries=0)


def _three_gene_act(generate: str, verify: str, actuator: str) -> LoopGenome:
    return LoopGenome(
        id="loop",
        name="L",
        genes=[
            LoopGene(id="generate", type=GeneType.SKILL, name="Generate", content=generate),
            LoopGene(id="ensure_correct", type=GeneType.VERIFIER_RUBRIC, name="V", content=verify),
            LoopGene(id="actuator", type=GeneType.PROMPT, name="A", content=actuator),
        ],
    )


def test_actuator_gene_produces_the_repair_text_fed_to_refine():
    # Actuator experiment: on a failed critique, the actuator gene (over its context)
    # generates the repair guidance the refine step consumes — NOT the raw critique.
    # The detector (verify gene) still gates; the actuator only shapes the fix.
    calls = []

    def spy(prompt):
        calls.append(prompt)
        if "ACTUATOR_MARK" in prompt:  # the actuator gene's own step
            return "ACTUATOR_REPAIR_GUIDANCE"
        if "# Draft answer" in prompt:  # detector: fail so a refine round happens
            return "FAIL: not correct"
        if "# Problems to fix" in prompt:  # refine
            return "solid"
        return "x x"  # initial failing draft

    ctx_calls = []

    def ctx(draft, task):
        ctx_calls.append((draft, task))
        return "VISIBLE_TEST_SOURCE_BLOB"

    run_task = make_self_refine_run_task(
        spy,
        ifbench_score,
        feedback=ifbench_feedback,
        max_retries=1,
        actuator_gene="actuator",
        actuator_context=ctx,
    )
    genome = _three_gene_act("write it", "verify it", "ACTUATOR_MARK: turn tests into a fix")
    run_task(genome, _nw_task())

    # the actuator step ran, fed by its context (not the raw critique)
    assert ctx_calls, "actuator_context was never consulted"
    assert any("ACTUATOR_MARK" in c and "VISIBLE_TEST_SOURCE_BLOB" in c for c in calls)
    # the refine step consumed the actuator's output, not the detector's verdict
    refine = next(c for c in calls if "# Problems to fix" in c)
    assert "ACTUATOR_REPAIR_GUIDANCE" in refine
    assert "FAIL: not correct" not in refine


def test_actuator_is_off_by_default_refine_uses_the_critique():
    # Default (no actuator_gene): the refine step consumes the critique verbatim, so
    # every pre-actuator run is byte-identical.
    calls = []

    def spy(prompt):
        calls.append(prompt)
        if "# Draft answer" in prompt:
            return "FAIL: fix the thing"
        if "# Problems to fix" in prompt:
            return "solid"
        return "x x"

    make_self_refine_run_task(spy, ifbench_score, max_retries=1)(
        _two_gene("write it", "verify it"), _nw_task()
    )
    refine = next(c for c in calls if "# Problems to fix" in c)
    assert "FAIL: fix the thing" in refine  # the critique text, no actuator indirection


def test_self_refine_missing_gene_raises_not_silent_empty():
    # A genome lacking the generate/verify gene (a wiring bug) must fail loudly, not
    # silently run the loop with a blank instruction and report a plausible low score.
    bad = LoopGenome(
        id="loop",
        name="L",
        genes=[LoopGene(id="generate", type=GeneType.SKILL, name="G", content="draft it")],
    )  # no ensure_correct gene
    run_task = make_self_refine_run_task(_loop_complete, ifbench_score, max_retries=1)
    with pytest.raises(KeyError, match="ensure_correct"):
        run_task(bad, _nw_task())


def test_self_refine_critique_step_status_reflects_verdict():
    # The critique step's status must record its PASS/FAIL verdict (observability):
    # a weak rubric PASSes → SUCCESS; the CHECK_WHITESPACE rubric FAILs → FAILURE.
    from whystuck.core.models import StepStatus

    passed = make_self_refine_run_task(_loop_complete, ifbench_score, max_retries=1)(
        _two_gene("write it", "looks fine"), _nw_task()
    )
    crit = next(s for s in passed.steps if s.step_id.startswith("critique"))
    assert crit.status == StepStatus.SUCCESS


def test_verify_gene_changes_loop_outcome():
    # THE crux: evolving the verify gene must change the final score, so arm D
    # (evolves it) can beat arm B (does not). Same generate gene both times.
    good = make_self_refine_run_task(_loop_complete, ifbench_score, max_retries=2)(
        _two_gene("write it", "CHECK_WHITESPACE: verify the answer has no spaces"), _nw_task()
    )
    seed = make_self_refine_run_task(_loop_complete, ifbench_score, max_retries=2)(
        _two_gene("write it", "make sure it is correct"), _nw_task()
    )
    assert good.metadata["eval_score"] == 1.0  # good rubric -> caught + fixed
    assert seed.metadata["eval_score"] == 0.0  # weak rubric -> missed, no fix
    assert good.metadata["eval_score"] > seed.metadata["eval_score"]


def test_self_refine_stops_when_critique_passes():
    # If the critique passes on the first check, no wasteful refine call is made.
    calls = []

    def spy(prompt):
        calls.append(prompt)
        return _loop_complete(prompt)

    # weak rubric -> critique returns PASS immediately -> generate + 1 critique only.
    trace = make_self_refine_run_task(spy, ifbench_score, max_retries=3)(
        _two_gene("write it", "looks fine"), _nw_task()
    )
    assert not any("# Problems to fix" in c for c in calls)  # never refined
    assert len(calls) == 2  # generate + one critique
    # On PASS the loop must KEEP the (unrefined) draft and score IT — not adopt the
    # critique text ("PASS") as the answer. The draft "x x" fails no_whitespace → 0.0.
    final = [s for s in trace.steps if s.role == "worker"][-1]
    assert final.output_summary == "x x"
    assert trace.metadata["eval_score"] == 0.0


def test_self_refine_final_output_is_scored_programmatically():
    trace = make_self_refine_run_task(_loop_complete, ifbench_score, max_retries=2)(
        _two_gene("write it", "CHECK_WHITESPACE"), _nw_task()
    )
    # The refined answer "solid" (no whitespace) is the LAST worker step and is what
    # gets scored by the real ifbench_score — not the LLM critique prose, not the
    # failing initial draft. "solid" satisfies no_whitespace → eval_score 1.0.
    final = [s for s in trace.steps if s.role == "worker"][-1]
    assert final.output_summary == "solid"
    assert trace.metadata["eval_score"] == 1.0
    assert trace.metadata["eval_score"] == 1.0


# --- feedback-richness lever: a non-oracle stats tool feeds the critique --------


def test_stats_fn_output_is_injected_into_the_critique_prompt():
    # The measured-stats readout must reach the critique step so the verify gene
    # can reflect on it (Huang lever: tool-backed correction, not intrinsic).
    calls = []

    def spy(prompt):
        calls.append(prompt)
        return _loop_complete(prompt)

    run_task = make_self_refine_run_task(
        spy, ifbench_score, max_retries=1, stats_fn=lambda _out: "STATS_MARKER words=2"
    )
    run_task(_two_gene("write it", "CHECK_WHITESPACE"), _nw_task())
    critiques = [c for c in calls if "# Draft answer" in c]
    assert critiques and all("STATS_MARKER words=2" in c for c in critiques)


def test_no_stats_fn_leaves_critique_prompt_free_of_stats():
    calls = []

    def spy(prompt):
        calls.append(prompt)
        return _loop_complete(prompt)

    make_self_refine_run_task(spy, ifbench_score, max_retries=1)(
        _two_gene("write it", "CHECK_WHITESPACE"), _nw_task()
    )
    assert not any("STATS_MARKER" in c for c in calls)


# --- complexity lever: an optional third `plan` gene (plan -> generate -> ...) --


def _three_gene(generate: str, verify: str, plan: str) -> LoopGenome:
    return LoopGenome(
        id="loop3",
        name="L3",
        genes=[
            LoopGene(id="generate", type=GeneType.SKILL, name="Generate", content=generate),
            LoopGene(id="ensure_correct", type=GeneType.VERIFIER_RUBRIC, name="V", content=verify),
            LoopGene(id="plan", type=GeneType.PROMPT, name="Plan", content=plan),
        ],
    )


def _plan_spy(calls):
    def spy(prompt):
        calls.append(prompt)
        if "PLANMARK" in prompt:
            return "PLAN_OUT"
        if "# Draft answer" in prompt:
            return "FAIL: fix it"  # never passes -> full retry budget runs
        if "# Problems to fix" in prompt:
            return "solid"
        return "x x"  # generate

    return spy


def test_plan_gene_runs_a_plan_step_that_feeds_generate():
    calls = []
    trace = make_self_refine_run_task(
        _plan_spy(calls), ifbench_score, max_retries=1, plan_gene="plan"
    )(_three_gene("write it", "CHECK_WHITESPACE", "PLANMARK: outline the constraints"), _nw_task())
    # the plan gene content reached the LLM as the FIRST call...
    assert "PLANMARK" in calls[0]
    # ...a plan step is recorded...
    assert any(s.step_id == "plan" for s in trace.steps)
    # ...and its output is threaded into the generate prompt.
    assert any("# Plan\nPLAN_OUT" in c for c in calls)


@pytest.mark.parametrize("retries", [1, 2])
def test_plan_gene_adds_exactly_one_call(retries):
    # cost guard: with a plan step the loop makes 2 + 2*retries calls/eval (the
    # extra +1 is the plan). Mispricing this would void the $5 budget bound.
    calls = []
    make_self_refine_run_task(
        _plan_spy(calls), ifbench_score, max_retries=retries, plan_gene="plan"
    )(_three_gene("write it", "CHECK_WHITESPACE", "PLANMARK"), _nw_task())
    assert len(calls) == 2 + 2 * retries


def test_plan_gene_missing_raises_not_silent():
    run_task = make_self_refine_run_task(
        _loop_complete, ifbench_score, max_retries=1, plan_gene="plan"
    )
    with pytest.raises(KeyError, match="plan"):
        run_task(_two_gene("write it", "CHECK_WHITESPACE"), _nw_task())  # no plan gene


# --- cross-family critic lever (run 6): critique runs on a different model -------


def test_critique_uses_the_separate_critic_model_when_given():
    # The verify/critique step must route to `critique_complete`, while generation
    # and refinement stay on the worker `complete` — this is what makes the critic
    # a different model than the generator (self-preference-bias mitigation).
    worker_calls, crit_calls = [], []

    def worker(p):
        worker_calls.append(p)
        return "x x"  # generate + refine (both fail no_whitespace)

    def critic(p):
        crit_calls.append(p)
        return "FAIL: remove whitespace" if "# Draft answer" in p else "?"

    make_self_refine_run_task(worker, ifbench_score, max_retries=1, critique_complete=critic)(
        _two_gene("write it", "CHECK_WHITESPACE"), _nw_task()
    )
    assert crit_calls and all("# Draft answer" in c for c in crit_calls)  # critique -> critic
    assert not any("# Draft answer" in c for c in worker_calls)  # never on the worker
    assert any("# Problems to fix" in c for c in worker_calls)  # refine -> worker


# --- oracle-critique lever (run 7): the ceiling arm; critique = the true metric --


@pytest.mark.parametrize("retries", [1, 2])
def test_oracle_critique_makes_no_critique_llm_call(retries):
    # With the oracle critique the gate uses the programmatic score directly, so
    # there is NO critique LLM call: the loop makes 1 + retries calls (generate +
    # one refine per unmet round), and the verify gene is never sent to any model.
    calls = []

    def spy(p):
        calls.append(p)
        return "x x"  # every draft keeps whitespace -> score < 1 -> never passes

    make_self_refine_run_task(spy, ifbench_score, max_retries=retries, oracle_critique=True)(
        _two_gene("write it", "SECRET_VERIFY_GENE"), _nw_task()
    )
    assert len(calls) == 1 + retries
    assert not any("SECRET_VERIFY_GENE" in c for c in calls)  # verify gene inert under oracle


def test_oracle_critique_ignores_a_provided_critic_model():
    # Precedence: under the oracle gate there is NO critique LLM call, so a
    # critique_complete passed alongside must never be invoked — else a refactor
    # could silently spend on the critic model under the "no critique call" guarantee.
    critic_calls = []

    def critic(p):
        critic_calls.append(p)
        return "PASS"

    make_self_refine_run_task(
        lambda p: "x x",
        ifbench_score,
        max_retries=2,
        oracle_critique=True,
        critique_complete=critic,
    )(_two_gene("write it", "V"), _nw_task())
    assert critic_calls == []


def test_oracle_critique_passes_immediately_on_a_correct_draft():
    calls = []

    def spy(p):
        calls.append(p)
        return "solid"  # no whitespace -> score 1.0 -> oracle passes at once

    trace = make_self_refine_run_task(spy, ifbench_score, max_retries=2, oracle_critique=True)(
        _two_gene("write it", "V"), _nw_task()
    )
    assert len(calls) == 1  # generate only; no refine, no critique
    assert trace.metadata["eval_score"] == 1.0


# --- feedback-actionability lever (D2'): oracle_feedback shapes the refine text -


def test_oracle_feedback_overrides_the_refine_critique_text():
    # Under the oracle gate, a provided oracle_feedback supplies the repair text
    # handed to the refine step — NOT the generic fallback, NOT the `feedback` fn.
    calls = []

    def spy(p):
        calls.append(p)
        return "x x"  # always fails no_whitespace -> oracle gate fails -> refine runs

    run_task = make_self_refine_run_task(
        spy,
        ifbench_score,
        feedback=lambda o, t: "GENERIC_FEEDBACK",
        max_retries=1,
        oracle_critique=True,
        oracle_feedback=lambda o, t: "ACTIONABLE_DIRECTIVE_MARKER",
    )
    trace = run_task(_two_gene("write it", "V"), _nw_task())
    refine = [c for c in calls if "# Problems to fix" in c]
    assert refine and all("ACTIONABLE_DIRECTIVE_MARKER" in c for c in refine)
    assert not any("GENERIC_FEEDBACK" in c for c in refine)  # oracle_feedback wins over feedback
    # oracle_feedback shapes ONLY the in-loop refine text; the final trace feedback
    # (reflection signal) must still come from `feedback`, not the refine override.
    assert trace.metadata["feedback"] == "GENERIC_FEEDBACK"


def test_oracle_feedback_default_none_preserves_feedback_behaviour():
    # Default (oracle_feedback=None) => the refine text is `feedback(draft,task)`,
    # byte-identical to every prior oracle run — the invariant that lets run-8's
    # cassette replay unchanged when the D2' "mid" level (the default) is selected.
    calls = []

    def spy(p):
        calls.append(p)
        return "x x"

    make_self_refine_run_task(
        spy,
        ifbench_score,
        feedback=lambda o, t: "FEEDBACK_TEXT",
        max_retries=1,
        oracle_critique=True,
    )(_two_gene("write it", "V"), _nw_task())
    refine = [c for c in calls if "# Problems to fix" in c]
    assert refine and all("FEEDBACK_TEXT" in c for c in refine)


def test_oracle_feedback_is_inert_without_oracle_critique():
    # Off the oracle path the critique text comes from the LLM; oracle_feedback must
    # never leak into an intrinsic-critique run (guards against a mis-wired lever).
    calls = []

    def spy(p):
        calls.append(p)
        return _loop_complete(p)

    make_self_refine_run_task(
        spy,
        ifbench_score,
        max_retries=1,
        oracle_critique=False,
        oracle_feedback=lambda o, t: "SHOULD_NOT_APPEAR",
    )(_two_gene("write it", "CHECK_WHITESPACE"), _nw_task())
    assert not any("SHOULD_NOT_APPEAR" in c for c in calls)


def test_oracle_feedback_still_makes_no_critique_llm_call():
    # The lever must not reintroduce a critique LLM call: oracle stays 1 + retries.
    calls = []

    def spy(p):
        calls.append(p)
        return "x x"

    make_self_refine_run_task(
        spy,
        ifbench_score,
        max_retries=2,
        oracle_critique=True,
        oracle_feedback=lambda o, t: "DIRECTIVE",
    )(_two_gene("write it", "SECRET_VERIFY_GENE"), _nw_task())
    assert len(calls) == 1 + 2  # generate + 2 refines, no critique call
    assert not any("SECRET_VERIFY_GENE" in c for c in calls)


# --- exec_feedback: task-aware in-loop execution signal (v2 code) ----------------


def test_exec_feedback_is_injected_into_the_non_oracle_critique_prompt():
    # The code study needs base-test EXECUTION output inside the critique — the signal
    # v1's blind self-critique lacked. Unlike stats_fn, this is task-aware.
    calls = []

    def spy(p):
        calls.append(p)
        return "PASS" if "# Draft answer" in p else "draft"

    make_self_refine_run_task(
        spy,
        ifbench_score,
        max_retries=1,
        exec_feedback=lambda draft, task: "EXEC_MARKER: 2/3 base tests failed",
    )(_two_gene("write it", "verify"), _nw_task())
    critique_prompts = [c for c in calls if "# Draft answer" in c]
    assert critique_prompts and all("EXEC_MARKER" in c for c in critique_prompts)


def test_exec_feedback_is_task_aware():
    # It receives the task, so different tasks can yield different readouts (base tests
    # differ per task) — the whole reason stats_fn(draft) was insufficient for code.
    seen = []

    def spy(p):
        return "PASS" if "# Draft answer" in p else "draft"

    make_self_refine_run_task(
        spy,
        ifbench_score,
        max_retries=1,
        exec_feedback=lambda draft, task: seen.append(task.id) or f"exec for {task.id}",
    )(_two_gene("g", "v"), BenchmarkTask(id="task-XYZ", payload={"goal": "do"}))
    assert "task-XYZ" in seen


def test_no_exec_feedback_leaves_the_critique_prompt_unchanged():
    # Additive lever: with exec_feedback unset there is NO test-results block, so prior
    # cassettes (recorded without it) replay byte-identically.
    calls = []

    def spy(p):
        calls.append(p)
        return "PASS" if "# Draft answer" in p else "draft"

    make_self_refine_run_task(spy, ifbench_score, max_retries=1)(_two_gene("g", "v"), _nw_task())
    assert not any("# Test results" in c for c in calls)


def test_exec_feedback_is_ignored_on_the_oracle_path():
    # The oracle path gates on the true metric and uses oracle_feedback; exec_feedback is
    # a non-oracle critique tool and must not enter that path.
    calls = []

    def spy(p):
        calls.append(p)
        return "x x"

    make_self_refine_run_task(
        spy,
        ifbench_score,
        max_retries=2,
        oracle_critique=True,
        exec_feedback=lambda draft, task: "EXEC_SHOULD_NOT_APPEAR",
    )(_two_gene("g", "v"), _nw_task())
    assert not any("EXEC_SHOULD_NOT_APPEAR" in c for c in calls)


# --- verdict/prose splice lever (E1): one model gates, another writes the repair -


def _splice_task(verdict, prose, **kw):
    """Run the no-whitespace task with the verdict and prose on different models.

    Returns (worker_prompts, verdict_prompts, prose_prompts).
    """
    worker_calls, verdict_calls, prose_calls = [], [], []

    def worker(p):
        worker_calls.append(p)
        return "x x"  # always fails no_whitespace

    def verdict_model(p):
        verdict_calls.append(p)
        return verdict

    def prose_model(p):
        prose_calls.append(p)
        return prose

    make_self_refine_run_task(
        worker,
        ifbench_score,
        max_retries=1,
        critique_complete=prose_model,
        verdict_complete=verdict_model,
        **kw,
    )(_two_gene("write it", "CHECK"), _nw_task())
    return worker_calls, verdict_calls, prose_calls


def test_verdict_comes_from_the_verdict_model_and_prose_from_the_critique_model():
    # The whole point of the splice: the PASS/FAIL bit and the repair text come from
    # DIFFERENT models on the SAME prompt. If either leaks from the other, the
    # experiment measures nothing -- it is the two-at-once confound it exists to break.
    worker, verdicts, prose = _splice_task("FAIL: something", "PROSE_MARKER")
    assert verdicts and prose, "both models must be called"
    assert verdicts == prose, "both must see the identical critique prompt"
    refine = [c for c in worker if "# Problems to fix" in c]
    assert refine, "a FAIL verdict must trigger refinement"
    assert all("PROSE_MARKER" in c for c in refine), "refine must be told the PROSE model's text"


def test_the_verdict_model_alone_decides_whether_to_refine():
    # Prose says PASS, verdict says FAIL -> the loop must refine. The reverse of the
    # unspliced loop, where the prose text IS the gate.
    worker, _, _ = _splice_task("FAIL: fix it", "PASS")
    assert [c for c in worker if "# Problems to fix" in c], "the verdict model gates, not the prose"

    # Verdict says PASS, prose complains -> the loop must stop.
    worker, _, _ = _splice_task("PASS", "FAIL: still broken")
    assert not [c for c in worker if "# Problems to fix" in c], "a PASS verdict must stop the loop"


def test_a_passing_prose_source_falls_back_to_the_generic_revise_directive():
    # When the verdict says FAIL but the prose model said PASS, the prose model has
    # no repair text to give. Handing refine the literal token "PASS" as guidance
    # would be nonsense, so it gets the same generic directive the oracle path uses
    # when no feedback function is supplied. That case is not a defect of the splice:
    # it is the prose source having nothing to say, made explicit.
    worker, _, _ = _splice_task("FAIL: fix it", "PASS")
    refine = [c for c in worker if "# Problems to fix" in c]
    assert refine and all("not all requirements are satisfied" in c for c in refine)
    assert not any("# Problems to fix\nPASS" in c for c in refine)


def test_the_splice_is_off_by_default_so_every_recorded_run_replays_unchanged():
    # verdict_complete=None must leave the loop byte-identical: the critique text is
    # both gate and prose, exactly as every cassette in the release recorded it.
    calls = []

    def worker(p):
        calls.append(p)
        return "x x"

    def critic(p):
        return "PASS"

    make_self_refine_run_task(worker, ifbench_score, max_retries=1, critique_complete=critic)(
        _two_gene("write it", "CHECK"), _nw_task()
    )
    assert not [c for c in calls if "# Problems to fix" in c], "critique PASS must still gate"
