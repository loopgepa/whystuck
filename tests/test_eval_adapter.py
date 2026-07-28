"""Tests for whystuck.eval.adapter — the loop-genome ↔ GEPA-candidate bridge.

A fake run_task (outcome depends on the candidate text) keeps this offline and
lets us prove a better candidate actually scores higher.
"""

from conftest import FIXED_TS

from whystuck.core.models import GeneType, LoopGene, LoopGenome, LoopMetrics, LoopTrace
from whystuck.eval.adapter import (
    LoopGenomeAdapter,
    candidate_from_genome,
    genome_from_candidate,
)
from whystuck.eval.benchmark import BenchmarkTask


def _genome() -> LoopGenome:
    return LoopGenome(
        id="bugfix",
        name="Bugfix",
        genes=[
            LoopGene(id="skill", type=GeneType.SKILL, name="S", content="Fix the bug."),
            LoopGene(id="retry", type=GeneType.RETRY_POLICY, name="R", content={"max_retries": 1}),
        ],
    )


def _run_task(genome: LoopGenome, task: BenchmarkTask) -> LoopTrace:
    # Outcome depends on the candidate's skill text — success iff it says "minimal".
    skill = next(g for g in genome.genes if g.id == "skill").content
    ok = isinstance(skill, str) and "minimal" in skill
    return LoopTrace(
        id=f"run-{task.id}",
        loop_id="bugfix",
        created_at=FIXED_TS,
        final_outcome="success" if ok else "failure",
        failure_summary=None if ok else "patch was not minimal",
        metrics=LoopMetrics(success=ok),
    )


def test_candidate_extracts_only_text_genes():
    cand = candidate_from_genome(_genome())
    assert cand == {"skill": "Fix the bug."}  # retry (dict) excluded


def test_genome_from_candidate_replaces_without_mutating_base():
    base = _genome()
    new = genome_from_candidate(base, {"skill": "Fix with the smallest/minimal patch."})
    assert next(g for g in new.genes if g.id == "skill").content.endswith("patch.")
    assert next(g for g in base.genes if g.id == "skill").content == "Fix the bug."  # base intact


def test_evaluate_scores_better_candidate_higher():
    adapter = LoopGenomeAdapter(base_genome=_genome(), run_task=_run_task)
    tasks = [BenchmarkTask(id="t1"), BenchmarkTask(id="t2")]
    bad = adapter.evaluate(tasks, {"skill": "Fix the bug."})
    good = adapter.evaluate(tasks, {"skill": "Fix with the minimal patch."})
    assert sum(good.scores) > sum(bad.scores)
    assert len(good.scores) == 2


def test_make_reflective_dataset_uses_failing_feedback():
    adapter = LoopGenomeAdapter(base_genome=_genome(), run_task=_run_task)
    tasks = [BenchmarkTask(id="t1")]
    batch = adapter.evaluate(tasks, {"skill": "Fix the bug."}, capture_traces=True)
    ds = adapter.make_reflective_dataset({"skill": "Fix the bug."}, batch, ["skill"])
    assert "skill" in ds
    assert "minimal" in ds["skill"][0]["Feedback"] or "not minimal" in ds["skill"][0]["Feedback"]


# --- fractional scoring + real reflective content (GEPA reflection fix) -------

from whystuck.core.models import LoopStepTrace, StepStatus  # noqa: E402


def _trace_with(eval_score, *, task_summary=None, output=None, feedback=None):
    md = {"eval_score": eval_score}
    if feedback is not None:
        md["feedback"] = feedback
    steps = []
    if output is not None:
        steps = [
            LoopStepTrace(
                step_id="worker", role="worker", status=StepStatus.FAILURE, output_summary=output
            )
        ]
    return LoopTrace(
        id="r",
        loop_id="x",
        created_at=FIXED_TS,
        task_summary=task_summary,
        final_outcome="failure",
        steps=steps,
        metrics=LoopMetrics(success=False),
        metadata=md,
    )


def test_evaluate_prefers_fractional_eval_score_over_binary_scalarize():
    # metadata['eval_score']=0.5 must surface as 0.5, NOT 0.0 from scalarize(success=False).
    def rt(genome, task):
        return _trace_with(0.5)

    adapter = LoopGenomeAdapter(base_genome=_genome(), run_task=rt)
    b = adapter.evaluate([BenchmarkTask(id="t")], {"skill": "x"})
    assert b.scores == [0.5]


def test_evaluate_falls_back_to_scalarize_without_eval_score():
    # command-loop traces (no eval_score) keep the v0.1 success-dominant scoring.
    adapter = LoopGenomeAdapter(base_genome=_genome(), run_task=_run_task)
    good = adapter.evaluate([BenchmarkTask(id="t")], {"skill": "the minimal fix"})
    # exact value, not just >0: scalarize(success=True, no retry/diff) == 1.0. A
    # fallback that silently returned some other positive constant must not pass.
    assert good.scores == [1.0]


def test_evaluate_zero_eval_score_is_preserved_not_dropped():
    # eval_score == 0.0 is a real score, not "unset": it must survive as 0.0 and
    # NOT fall through to scalarize (the `is not None` check, not truthiness).
    def rt(genome, task):
        return _trace_with(0.0)

    adapter = LoopGenomeAdapter(base_genome=_genome(), run_task=rt)
    assert adapter.evaluate([BenchmarkTask(id="t")], {"skill": "x"}).scores == [0.0]


def test_reflective_dataset_surfaces_task_output_and_feedback():
    def rt(genome, task):
        return _trace_with(
            0.0, task_summary="FOLLOW THIS", output="WORKER SAID", feedback="Failed: X"
        )

    adapter = LoopGenomeAdapter(base_genome=_genome(), run_task=rt)
    b = adapter.evaluate([BenchmarkTask(id="t")], {"skill": "x"}, capture_traces=True)
    rec = adapter.make_reflective_dataset({"skill": "x"}, b, ["skill"])["skill"][0]
    assert "FOLLOW THIS" in str(rec["Inputs"])
    assert rec["Generated Outputs"] == "WORKER SAID"
    assert "Failed: X" in rec["Feedback"]


def _self_refine_trace(*, feedback="Failed: whitespace"):
    # A genuine generate→critique→refine trace: worker draft, verifier critique,
    # worker FINAL answer. Mirrors what make_self_refine_run_task emits.
    return LoopTrace(
        id="r",
        loop_id="x",
        created_at=FIXED_TS,
        task_summary="FOLLOW THIS",
        final_outcome="success",
        steps=[
            LoopStepTrace(step_id="generate-0", role="worker", output_summary="draft v1 (bad)"),
            LoopStepTrace(step_id="critique-0", role="verifier", output_summary="FAIL: fix it"),
            LoopStepTrace(step_id="refine-0", role="worker", output_summary="FINAL ANSWER"),
        ],
        metrics=LoopMetrics(success=True),
        metadata={"eval_score": 1.0, "feedback": feedback},
    )


def test_worker_output_preserves_empty_string_not_outcome_string():
    # A content-filter refusal / empty completion yields output_summary "" — a REAL
    # (if empty) worker answer. It must survive as "", NOT be silently swapped for the
    # literal outcome word "failure"/"success" (which would feed GEPA fabricated text).
    from whystuck.eval.adapter import _worker_output

    trace = LoopTrace(
        id="r",
        loop_id="x",
        created_at=FIXED_TS,
        final_outcome="failure",
        steps=[LoopStepTrace(step_id="generate-0", role="worker", output_summary="")],
        metrics=LoopMetrics(success=False),
    )
    assert _worker_output(trace) == ""  # not "failure"


def test_reflective_dataset_preserves_explicit_empty_feedback():
    # If the feedback callable explicitly returns "" it must survive, not be replaced
    # by the synthesized "score=...; ..." fallback (presence check, not truthiness).
    def rt(genome, task):
        return _self_refine_trace(feedback="")

    adapter = LoopGenomeAdapter(base_genome=_genome(), run_task=rt)
    b = adapter.evaluate([BenchmarkTask(id="t")], {"skill": "x"}, capture_traces=True)
    rec = adapter.make_reflective_dataset({"skill": "x"}, b, ["skill"])["skill"][0]
    assert rec["Feedback"] == ""  # explicit empty preserved, not "score=..."


def test_reflective_dataset_uses_final_refined_answer_not_draft_or_critique():
    # THE reflection-signal crux: GEPA's "Generated Outputs" must be the loop's LAST
    # worker step (the refined answer), NOT the initial failing draft and NOT the
    # verifier's critique prose. If _worker_output picked steps[0] or a verifier
    # step, GEPA would reflect on the wrong text and the D-vs-B advantage would be
    # silently neutered. This is the only place a real multi-step trace is exercised.
    def rt(genome, task):
        return _self_refine_trace()

    adapter = LoopGenomeAdapter(base_genome=_genome(), run_task=rt)
    b = adapter.evaluate([BenchmarkTask(id="t")], {"skill": "x"}, capture_traces=True)
    rec = adapter.make_reflective_dataset({"skill": "x"}, b, ["skill"])["skill"][0]
    assert rec["Generated Outputs"] == "FINAL ANSWER"  # not "draft v1 (bad)", not "FAIL: fix it"


def _judged_trace(*, critique: str, draft: str = "draft v1 (bad)"):
    """A trace whose critique step records BOTH the draft it judged and its verdict."""
    return LoopTrace(
        id="r",
        loop_id="x",
        created_at=FIXED_TS,
        task_summary="FOLLOW THIS",
        final_outcome="success",
        steps=[
            LoopStepTrace(step_id="generate-0", role="worker", output_summary=draft),
            LoopStepTrace(
                step_id="critique-0", role="verifier", input_summary=draft, output_summary=critique
            ),
            LoopStepTrace(step_id="refine-0", role="worker", output_summary="FINAL ANSWER"),
        ],
        metrics=LoopMetrics(success=True),
        metadata={"eval_score": 1.0, "feedback": "Failed: whitespace"},
    )


def test_verify_gene_records_are_the_critique_not_the_generator():
    """THE root-cause test for arm D.

    Previously every component received identical records whose "Generated Outputs" was
    the GENERATOR's final answer, so GEPA optimized the critique gene while being shown
    nothing about what the critique did — and duly proposed generation prose into the
    critique slot. A detector cannot be optimized without seeing its own verdict, so the
    verify component's records must show the draft it judged, the verdict it returned,
    and whether that verdict was right.
    """

    def rt(genome, task):
        return _judged_trace(critique="PASS")

    adapter = LoopGenomeAdapter(
        base_genome=_genome(),
        run_task=rt,
        verify_gene="verify",
        # Ground truth for the judged draft: it did NOT actually satisfy the task.
        verify_truth=lambda output, task: (False, "exact_word_count: got 25, want 20"),
    )
    b = adapter.evaluate([BenchmarkTask(id="t")], {"skill": "x"}, capture_traces=True)
    ds = adapter.make_reflective_dataset({"skill": "x"}, b, ["skill", "verify"])

    # The generate component is untouched: still the final worker answer.
    assert ds["skill"][0]["Generated Outputs"] == "FINAL ANSWER"

    # The verify component sees ITS OWN output, the draft it judged, and its error.
    vrec = ds["verify"][0]
    assert vrec["Generated Outputs"] == "PASS"
    assert "draft v1 (bad)" in vrec["Inputs"]
    assert "false pass" in vrec["Feedback"].lower()
    assert "exact_word_count: got 25, want 20" in vrec["Feedback"]


def test_verify_gene_records_flag_a_false_alarm():
    # The opposite detector error: the critique demanded fixes on an already-correct draft.
    def rt(genome, task):
        return _judged_trace(critique="FIX: add more words")

    adapter = LoopGenomeAdapter(
        base_genome=_genome(),
        run_task=rt,
        verify_gene="verify",
        verify_truth=lambda output, task: (True, "all requirements satisfied"),
    )
    b = adapter.evaluate([BenchmarkTask(id="t")], {"skill": "x"}, capture_traces=True)
    vrec = adapter.make_reflective_dataset({"skill": "x"}, b, ["verify"])["verify"][0]
    assert "false alarm" in vrec["Feedback"].lower()


def test_verify_gene_records_confirm_a_correct_verdict():
    def rt(genome, task):
        return _judged_trace(critique="PASS")

    adapter = LoopGenomeAdapter(
        base_genome=_genome(),
        run_task=rt,
        verify_gene="verify",
        verify_truth=lambda output, task: (True, "all requirements satisfied"),
    )
    b = adapter.evaluate([BenchmarkTask(id="t")], {"skill": "x"}, capture_traces=True)
    vrec = adapter.make_reflective_dataset({"skill": "x"}, b, ["verify"])["verify"][0]
    assert "correct" in vrec["Feedback"].lower()


def test_detector_records_are_opt_in_so_committed_replays_are_unchanged():
    # Without verify_gene/verify_truth the adapter must behave exactly as before:
    # every component gets the identical generator-shaped records. Anything else would
    # change the reflection prompts and invalidate the released cassette replays.
    def rt(genome, task):
        return _judged_trace(critique="PASS")

    adapter = LoopGenomeAdapter(base_genome=_genome(), run_task=rt)
    b = adapter.evaluate([BenchmarkTask(id="t")], {"skill": "x"}, capture_traces=True)
    ds = adapter.make_reflective_dataset({"skill": "x"}, b, ["skill", "verify"])
    assert ds["skill"] == ds["verify"]
    assert ds["verify"][0]["Generated Outputs"] == "FINAL ANSWER"
