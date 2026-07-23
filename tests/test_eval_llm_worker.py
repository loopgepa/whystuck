"""Tests for whystuck.eval.llm_worker — proves text genes move the score."""

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


def test_litellm_complete_without_litellm_is_actionable():
    complete = litellm_complete("openai/gpt-4o-mini", max_tokens=256)
    # The error must name the real fix. litellm is now part of the `whystuck[eval]`
    # extra (added so the one paid run reproduces from a single install), so the
    # extra IS the fix — superseding the earlier state where litellm was separate.
    with pytest.raises(RuntimeError, match=r"whystuck\[eval\]"):
        complete("hi")


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
