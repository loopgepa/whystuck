"""LLM-backed worker — the piece that makes text-gene optimization actually bite.

The command+verifier runner's outcome doesn't depend on the loop's prose genes,
so GEPA can't optimize them through it. This worker builds a prompt *from* the
text genes, calls an LLM, and scores the output — so changing a skill/verifier
rubric changes the result, and GEPA has a gradient to climb.

`complete` (the LLM call) and `score` (the task's success check) are injected, so
this is fully offline-testable. The production `complete` is litellm-backed
(network); litellm ships with the `whystuck[eval]` extra, not the base install.
"""

from __future__ import annotations

from collections.abc import Callable

from whystuck.core.models import (
    LoopGenome,
    LoopMetrics,
    LoopStepTrace,
    LoopTrace,
    outcome_for,
    step_status_for,
)
from whystuck.core.util import utc_now_iso
from whystuck.eval.adapter import RunTask
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.prompts import render_text_genes

Complete = Callable[[str], str]
# A task scorer: fraction of requirements satisfied ∈ [0, 1]. A bool scorer still
# works (True/False coerce to 1.0/0.0) — the fractional form just gives GEPA a
# smoother gradient on partially-satisfied tasks.
Score = Callable[[str, BenchmarkTask], float]
# Optional per-task feedback prose for GEPA's reflective dataset (e.g. which
# constraints passed/failed). Without it, a generic success/failure message is used.
Feedback = Callable[[str, BenchmarkTask], str]
BuildPrompt = Callable[[LoopGenome, BenchmarkTask], str]


def _task_text(task: BenchmarkTask) -> str:
    """The task's instruction, as shown to the worker and to GEPA's reflector."""
    payload = task.payload or {}
    return str(payload.get("goal") or payload.get("task_summary") or task.id)


def default_build_prompt(genome: LoopGenome, task: BenchmarkTask) -> str:
    """Assemble a prompt from the genome's text genes + the task goal."""
    parts = render_text_genes(genome)
    parts.append(f"# Task\n{_task_text(task)}")
    return "\n\n".join(parts)


def keyword_score(output: str, task: BenchmarkTask) -> bool:
    """Default ``Score``: the output must contain the task's declared expectation.

    ``payload.expect`` is a substring, or ``payload.expect_any`` a list of them;
    matching is case-insensitive. With no expectation declared nothing can be
    verified, so the task scores ``False`` — an unscored task must not count as a
    free win, or it would flatten GEPA's gradient for that task."""
    payload = task.payload or {}
    expect = payload.get("expect")
    candidates = payload.get("expect_any") or ([expect] if expect else [])
    low = output.lower()
    return any(isinstance(e, str) and e.lower() in low for e in candidates)


def make_run_task(
    complete: Complete,
    score: Score,
    *,
    feedback: Feedback | None = None,
    build_prompt: BuildPrompt = default_build_prompt,
) -> RunTask:
    """Build a run_task(genome, task) → LoopTrace that runs the loop via an LLM.

    The trace carries the real reflective signal GEPA needs to optimize: the task
    instruction (`task_summary` + the worker step's `input_summary`), the worker's
    actual output text (`output_summary`), the fractional score
    (`metadata['eval_score']`, which the adapter maximizes), and per-task feedback
    (`metadata['feedback']`). Without these, GEPA's reflector is blind to the task
    and cannot propose targeted improvements.
    """

    def run_task(genome: LoopGenome, task: BenchmarkTask) -> LoopTrace:
        prompt = build_prompt(genome, task)
        output = complete(prompt)
        eval_score = float(score(output, task))
        ok = eval_score >= 1.0  # "success" == every requirement satisfied
        task_text = _task_text(task)
        fb = (
            feedback(output, task)
            if feedback is not None
            else ("all requirements satisfied" if ok else "verifier rejected the worker output")
        )
        return LoopTrace(
            id=f"llm-{task.id}",
            loop_id=genome.id,
            created_at=utc_now_iso(),
            task_id=task.id,
            task_summary=task_text,
            final_outcome=outcome_for(ok),
            failure_summary=None if ok else fb,
            steps=[
                LoopStepTrace(
                    step_id="worker",
                    role="worker",
                    status=step_status_for(ok),
                    input_summary=task_text,
                    output_summary=output,
                )
            ],
            metrics=LoopMetrics(success=ok),
            metadata={"eval_score": eval_score, "feedback": fb},
        )

    return run_task


def _gene_content(genome: LoopGenome, gene_id: str) -> str:
    """The text of one gene by id. Fails loudly on a missing or non-text gene: a
    silent empty-string fallback would run the loop with a blank instruction and
    report a plausible low score, hiding a wiring bug (typo'd id, wrong gene type)
    behind a result that looks like 'the genome performed badly'."""
    for gene in genome.genes:
        if gene.id == gene_id:
            if not isinstance(gene.content, str):
                raise TypeError(
                    f"gene {gene_id!r} is not text-typed (got {type(gene.content).__name__}); "
                    "the self-refine worker needs a string skill/rubric"
                )
            return gene.content
    raise KeyError(
        f"genome {genome.id!r} has no gene {gene_id!r} (required by the self-refine worker)"
    )


def make_self_refine_run_task(
    complete: Complete,
    score: Score,
    *,
    feedback: Feedback | None = None,
    generate_gene: str = "generate",
    verify_gene: str = "ensure_correct",
    plan_gene: str | None = None,
    stats_fn: Callable[[str], str] | None = None,
    exec_feedback: Feedback | None = None,
    critique_complete: Complete | None = None,
    verdict_complete: Complete | None = None,
    oracle_critique: bool = False,
    oracle_feedback: Feedback | None = None,
    actuator_gene: str | None = None,
    actuator_context: Feedback | None = None,
    max_retries: int = 2,
) -> RunTask:
    """A real generate -> critique -> refine loop (not single-shot).

    Each role uses a *different* gene, so evolving each gene has a distinct effect
    (unlike the single-shot worker, where both genes were just concatenated text):
    - **generate** (`generate_gene`) drafts and later refines the answer;
    - **critique** (`verify_gene`) drives an LLM self-check that decides whether to
      retry and what to fix. Its verdict is the retry gate (intrinsic, no oracle:
      the programmatic scorer is never consulted inside the loop). A better verify
      rubric catches more failures and steers better refinements, which is exactly
      why arm D (evolves both genes) can beat arm B (evolves only `generate`).

    Optional levers, each gated off by default (so existing runs are unchanged):
    - **plan_gene** (complexity / DSPy lever): when set, a `plan` step runs FIRST and
      its output is threaded into the generate prompt, giving the loop a third
      interacting module for whole-genome optimization to co-tune. Adds one LLM call
      per eval (up to 2 + 2*max_retries; fewer if the critique passes early). Missing
      gene id fails loud via `_gene_content`.
    - **stats_fn** (feedback-richness / Huang lever): when set, `stats_fn(draft)` is
      injected into every critique (and refine) prompt as a measured, NON-oracle tool
      readout — the external-verification signal Huang et al. 2023 found necessary for
      self-correction to help. It measures the text only; the programmatic scorer is
      still never consulted in the loop, so this is a tool, not oracle contamination.
    - **exec_feedback** (task-aware execution signal, v2 code): like `stats_fn` but it
      receives the *task* too — `exec_feedback(draft, task)` — so it can run the task's
      VISIBLE tests against the draft and inject the result (pass/fail + traceback) into
      the non-oracle critique. This is the reliable, executable feedback v1's blind
      self-critique lacked (the Reflexion regime) and the lever that lets arm D — which
      evolves how the verify gene *reads* that execution output — beat arm B on code.
      Non-oracle only (the oracle path gates on the true metric, so no critique tool
      runs); base tests ≠ the hidden scorer, so it is a tool, not oracle contamination.
      Additive: unset → no block, so prior cassettes replay byte-identically.
    - **critique_complete** (cross-family critic lever, run 6): when set, the critique
      LLM call routes to this model instead of the worker `complete`, so the verifier
      is a DIFFERENT model than the generator. Motivated by the finding that a judge
      from the same family inflates agreement and carries a self-preference bias
      (Valmeekam et al. 2023; LLM-as-judge 2026 surveys). NOTE: within one provider
      (e.g. gpt-5.6 critic over a gpt-4.1 worker) this is cross-*lineage*, which breaks
      self-grading but is not a true cross-*provider* family test — that remains future
      work needing a second provider's key.
    - **oracle_critique** (ceiling lever, run 7): when set, the critique gate uses the
      true programmatic metric (`score`/`feedback`) instead of an LLM self-check — the
      deliberate oracle condition (Huang et al.'s "with oracle" setting). It makes NO
      critique LLM call (up to 1 + max_retries calls/eval) and leaves the verify gene inert;
      it exists to establish the loop's performance CEILING given a perfect detector,
      isolating whether the null is about signal quality vs loop mechanics. Not a
      D-vs-B test (the verify gene cannot matter under a perfect oracle).
    - **oracle_feedback** (feedback-actionability lever, D2'): ONLY consulted under
      `oracle_critique`. When set, its `(draft, task) -> str` output becomes the repair
      text fed to the refine step, replacing the default `feedback`. Detection is
      unchanged (the gate is still the true metric) — this varies *how actionable* the
      repair instruction is, isolating whether the residual fix-step gap (runs 7-8) is
      feedback-limited or generator-limited. Default None falls back to `feedback`, so
      it is a no-op unless explicitly set (and inert entirely off the oracle path).

    - **actuator_gene** / **actuator_context** (actuator experiment): when
      `actuator_gene` is set, a failing critique no longer feeds its own text to the
      refine step; instead this gene runs an LLM step over `actuator_context(draft,
      task)` (the raw test signal + visible-test source, e.g. bcb `hi` feedback) and
      its output becomes the repair guidance. Detection is unchanged (the `verify_gene`
      critique still gates), so evolving THIS gene (arm D) tests whether GEPA helps when
      it optimizes the fixer's feedback rather than the detector. Non-oracle only;
      default None → `repair_text` stays the critique, so existing runs are unchanged.

    The final answer is scored by `score` (the ground-truth programmatic metric),
    so the loop cannot game it. `max_retries` bounds the critique/refine rounds and
    MUST be ≥ 1 — at 0 the critique never runs, the verify gene never enters a
    prompt, and arm D silently collapses into arm B (the exact null this loop fixes).
    """
    if max_retries < 1:
        raise ValueError(
            f"max_retries must be >= 1 (got {max_retries}); at 0 the critique never runs "
            "and the verify gene is inert, collapsing loop-level GEPA into prompt-only"
        )

    def run_task(genome: LoopGenome, task: BenchmarkTask) -> LoopTrace:
        gen, ver = _gene_content(genome, generate_gene), _gene_content(genome, verify_gene)
        task_text = _task_text(task)
        steps: list[LoopStepTrace] = []

        # Complexity lever: optional plan step before the first draft.
        plan_block = ""
        if plan_gene is not None:
            plan = complete(f"{_gene_content(genome, plan_gene)}\n\n# Task\n{task_text}")
            steps.append(
                LoopStepTrace(
                    step_id="plan", role="planner", input_summary=task_text, output_summary=plan
                )
            )
            plan_block = f"\n\n# Plan\n{plan}"

        draft = complete(f"{gen}\n\n# Task\n{task_text}{plan_block}")
        steps.append(
            LoopStepTrace(
                step_id="generate-0", role="worker", input_summary=task_text, output_summary=draft
            )
        )
        critic = critique_complete or complete
        for i in range(max_retries):
            if oracle_critique:
                # Ceiling lever: the gate is the true metric, not an LLM self-check.
                passed = score(draft, task) >= 1.0
                # Feedback-actionability lever (D2'): `oracle_feedback` overrides ONLY
                # the repair text handed to the refine step — detection (the gate
                # above) is unchanged (still the true metric) and the final reflection
                # `feedback` below is unchanged. Default None falls back to `feedback`,
                # so oracle runs that don't set it are byte-identical to before (this
                # is what lets run-8's cassette replay as the "mid" level unchanged).
                refine_fb = oracle_feedback if oracle_feedback is not None else feedback
                critique = (
                    "PASS"
                    if passed
                    else (
                        refine_fb(draft, task)
                        if refine_fb is not None
                        else "not all requirements are satisfied; revise"
                    )
                )
            else:
                # Feedback lever: a measured (non-oracle) stats readout for the critique.
                stats_block = (
                    f"\n\n# Measured text statistics\n{stats_fn(draft)}" if stats_fn else ""
                )
                # Execution lever (v2 code): the task's VISIBLE test results on this draft.
                exec_block = (
                    f"\n\n# Test results\n{exec_feedback(draft, task)}" if exec_feedback else ""
                )
                critique_prompt = (
                    f"{ver}\n\n# Task\n{task_text}\n\n# Draft answer\n{draft}"
                    f"{stats_block}{exec_block}\n\n"
                    "Reply exactly PASS if every requirement is satisfied; "
                    "otherwise list what to fix."
                )
                critique = critic(critique_prompt)
                if verdict_complete is None:
                    passed = critique.strip().upper().startswith("PASS")
                else:
                    # Splice lever (E1): the gate and the repair text come from DIFFERENT
                    # models given the IDENTICAL prompt. A model swap otherwise moves both
                    # halves of the loop at once, which is why the cross-lineage critic's
                    # lift cannot be attributed to detection or to feedback as measured.
                    passed = verdict_complete(critique_prompt).strip().upper().startswith("PASS")
                    if not passed and critique.strip().upper().startswith("PASS"):
                        # The prose source saw nothing to fix, so it has no repair text to
                        # give. Passing refine the literal token "PASS" as guidance would be
                        # nonsense; it gets the same generic directive the oracle path uses
                        # when no feedback function is supplied. This is not noise in the
                        # experiment -- it is the prose source having nothing to say, and a
                        # bare "revise" is already this series' low-actionability baseline.
                        critique = "not all requirements are satisfied; revise"
            steps.append(
                LoopStepTrace(
                    step_id=f"critique-{i}",
                    role="verifier",
                    status=step_status_for(passed),
                    input_summary=draft,
                    output_summary=critique,
                )
            )
            if passed:
                break
            # Actuator experiment: a dedicated gene turns the test signal into the repair
            # guidance fed to refine, SEPARATE from the detector (`verify_gene`) that gated
            # above. Evolving this gene (arm D) asks whether GEPA helps when it optimizes the
            # fixer's feedback rather than the detector. Non-oracle only. Default off, so
            # every existing run is byte-identical (`repair_text` stays the critique).
            repair_text = critique
            if actuator_gene is not None and not oracle_critique:
                actuator_ctx = actuator_context(draft, task) if actuator_context else ""
                repair_text = complete(
                    f"{_gene_content(genome, actuator_gene)}\n\n# Task\n{task_text}\n\n"
                    f"# Draft answer\n{draft}\n\n# Test results\n{actuator_ctx}\n\n"
                    "Produce concrete, specific instructions to fix the draft so it passes."
                )
                steps.append(
                    LoopStepTrace(
                        step_id=f"actuator-{i}",
                        role="worker",
                        input_summary=actuator_ctx,
                        output_summary=repair_text,
                    )
                )
            draft = complete(
                f"{gen}\n\n# Task\n{task_text}{plan_block}\n\n# Your previous answer\n{draft}\n\n"
                f"# Problems to fix\n{repair_text}\n\nProduce a corrected answer."
            )
            steps.append(
                LoopStepTrace(
                    step_id=f"refine-{i}",
                    role="worker",
                    # Record what actually drove the refinement: the actuator's repair text
                    # when the actuator gene is on, else the critique (they're identical when
                    # it is off). GEPA reads traces to optimize genes, so this must be faithful.
                    input_summary=repair_text,
                    output_summary=draft,
                )
            )

        eval_score = float(score(draft, task))
        ok = eval_score >= 1.0
        fb = (
            feedback(draft, task)
            if feedback is not None
            else ("all requirements satisfied" if ok else "verifier rejected the worker output")
        )
        return LoopTrace(
            id=f"loop-{task.id}",
            loop_id=genome.id,
            created_at=utc_now_iso(),
            task_id=task.id,
            task_summary=task_text,
            final_outcome=outcome_for(ok),
            failure_summary=None if ok else fb,
            steps=steps,
            metrics=LoopMetrics(
                success=ok, retry_count=sum(1 for s in steps if "refine" in s.step_id)
            ),
            metadata={"eval_score": eval_score, "feedback": fb},
        )

    return run_task


def litellm_complete(model: str, *, max_tokens: int) -> Complete:
    """Production `complete` backed by litellm (network; ships with `whystuck[eval]`).

    `max_tokens` is REQUIRED (no default): it caps the completion length so output
    tokens — the dominant cost — are a hard ceiling matching what the budget guard
    priced. Making it required is deliberate fail-loud design — a silent unbounded
    default would let a future caller void the budget bound with no error (the same
    footgun the rest of this module's guards exist to prevent)."""

    def complete(
        prompt: str, *, temperature: float | None = None, sample: int | None = None
    ) -> str:
        """`temperature` and `sample` are passed PER CALL, never baked into the factory.

        That is load-bearing for the cassette, which keys on ``(name, args, kwargs)`` of this
        call. A factory-level temperature would not appear in the key, so two runs at
        different temperatures would silently serve each other's completions --- a
        contamination of exactly the kind this study had to retract a result over. Passed as
        kwargs, they are part of the key.

        ``sample`` is a cache-busting salt and is deliberately NOT sent to the provider: K
        identical calls otherwise collapse to one cassette entry and K ``independent'' samples
        come back byte-identical, which would read as zero sampling variance. Leaving both
        ``None`` reproduces the original single-argument call exactly, so every committed
        cassette key is preserved.
        """
        del sample  # salt only; its effect is on the cassette key, upstream of here
        try:
            import litellm
        except ImportError as exc:
            raise RuntimeError(
                "the LLM worker needs litellm: pip install 'whystuck[eval]'"
            ) from exc
        # An empty completion is retried once and then raised. Reasoning models can spend
        # the whole `max_tokens` allowance on reasoning and return `content = None`, and the
        # old `content or ""` turned that into an empty string that flowed downstream: in the
        # C2 run GEPA received it as a proposal of empty text for the verify gene, accepted
        # it on one seed because a loop with no critique scores better on the training
        # subsample than one with a near-random critique, and the arm carrying the loop-level
        # treatment then ran with no critique at all. Nothing distinguished that from an
        # evolved gene. Raising costs at most one aborted run, and the cassette persists per
        # call, so a resumed run re-reads every completion already paid for.
        for _attempt in range(2):
            # Two explicit calls rather than a splatted kwargs dict, so the no-temperature
            # path is provably the original call and not a construction that merely
            # evaluates to it.
            if temperature is None:
                resp = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
            else:
                resp = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            content = resp.choices[0].message.content
            if content and content.strip():
                return str(content)
        raise RuntimeError(
            f"{model} returned an empty completion twice in a row (max_tokens={max_tokens}); "
            "a reasoning model can exhaust the allowance before emitting content, so raise "
            "max_tokens rather than letting an empty string reach the caller"
        )

    return complete
