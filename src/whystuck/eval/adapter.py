"""Bridge a WhyStuck loop genome onto GEPA's candidate/evaluation interface.

GEPA evolves a `candidate: dict[str, str]` of named text components. Here those
are the loop's evolvable **text** genes (skill/agent/prompt/verifier rubric). The
runner-backed dict genes (retry/stop/budget) are not text-optimized in this alpha.

`evaluate` runs each benchmark task through an injected `run_task` (the loop
executor) and scores the resulting trace — so this module is fully offline-testable
and never imports `gepa`. `optimize.py` adapts it to the real GEPA optimizer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from whystuck.core.models import LoopGenome, LoopTrace
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.objectives import scalarize

RunTask = Callable[[LoopGenome, BenchmarkTask], LoopTrace]


def _score_of(trace: LoopTrace) -> float:
    """The score GEPA maximizes. Prefer a fractional ``metadata['eval_score']``
    (the LLM worker records constraint-satisfaction fraction there); fall back to
    the success-dominant ``scalarize`` for command-loop traces that don't set it,
    so v0.1 scoring is unchanged."""
    raw = trace.metadata.get("eval_score")
    return float(raw) if raw is not None else scalarize(trace.metrics)


def _worker_output(trace: LoopTrace) -> str:
    """The worker's *final* output text, for reflection.

    For a self-refine loop this is the last generate/refine step (role "worker"),
    NOT the initial draft and NOT a "verifier" critique step; for the single-shot
    worker there is exactly one such step. Falls back to any last step with output
    text, then to the outcome string (command-loop traces keep output on ToolCall,
    not the step, so they land on the outcome).

    An empty string is a *legitimate* worker answer (a content-filter refusal or an
    empty completion), so we return it as-is rather than masking it with the outcome
    word — a fabricated "success"/"failure" fed to GEPA's reflector would be worse
    than an honest empty output. The `is not None` filter (not truthiness) is the
    same discipline `_score_of` uses to keep a real 0.0 from looking 'unset'."""
    for step in reversed(trace.steps):
        if step.role == "worker" and step.output_summary is not None:
            return step.output_summary  # narrowed to str; "" is kept, not masked
    for step in reversed(trace.steps):
        if step.output_summary is not None:
            return step.output_summary
    return trace.final_outcome


@dataclass
class EvalBatch:
    outputs: list[str]
    scores: list[float]
    trajectories: list[LoopTrace] | None = None


def candidate_from_genome(genome: LoopGenome) -> dict[str, str]:
    """The evolvable text genes as GEPA's candidate."""
    return {g.id: g.content for g in genome.genes if g.evolvable and isinstance(g.content, str)}


def genome_from_candidate(base_genome: LoopGenome, candidate: dict[str, str]) -> LoopGenome:
    """A copy of the genome with its text genes replaced from the candidate.

    A blank replacement raises. This is not defensive tidiness: in the C2 run the
    reflection LM returned an empty completion twice, GEPA turned that into a proposal of
    empty text for the verify gene, and on one seed the empty gene scored *better* on the
    training subsample --- deleting a near-random critique helps training --- so it was
    accepted and became the winner. The arm carrying the loop-level treatment thus shipped
    a loop with no critique at all, and every downstream number (held-out score,
    `genes_changed`, the paired delta) reported it as an evolved gene. That is precisely
    the failure this paper documents: a broken treatment indistinguishable from an
    ineffective one. An optimizer may legitimately want a *shorter* rubric, so only
    all-whitespace content is refused.
    """
    new = base_genome.model_copy(deep=True)
    for gene in new.genes:
        if gene.id in candidate and isinstance(gene.content, str):
            replacement = candidate[gene.id]
            if not replacement.strip():
                raise ValueError(
                    f"candidate proposes blank text for gene {gene.id!r}; an empty gene "
                    "silently disables that step of the loop, so it is rejected rather "
                    "than scored (usual cause: the reflection LM returned an empty "
                    "completion, e.g. a reasoning model whose max_tokens was consumed "
                    "before it emitted content)"
                )
            gene.content = replacement
    return new


VerifyTruth = Callable[[str, BenchmarkTask], tuple[bool, str]]


def _critique_steps(trace: LoopTrace) -> list[tuple[str, str]]:
    """(judged draft, verdict text) for each critique step, in loop order.

    The self-refine loop records its critique as role ``verifier`` with
    ``input_summary`` = the draft under judgement and ``output_summary`` = the verdict,
    so the detector's own behaviour is fully recoverable from the trace.
    """
    return [
        (step.input_summary or "", step.output_summary or "")
        for step in trace.steps
        if step.role == "verifier" and step.output_summary is not None
    ]


class LoopGenomeAdapter:
    def __init__(
        self,
        base_genome: LoopGenome,
        run_task: RunTask,
        *,
        verify_gene: str | None = None,
        verify_truth: VerifyTruth | None = None,
    ) -> None:
        """`verify_gene` + `verify_truth` enable component-specific reflection.

        Both default to None, under which every component receives the same
        generator-shaped records — the behaviour that produced the superseded runs, and the
        reason their loop-level arm was never meaningfully optimized. Callers that evolve a
        critique gene should supply both; `run_validation_study` does so for every text task
        set. The None default remains only because a truth signal is not available for every
        task family (see below), not as a compatibility path.

        `verify_truth(draft, task) -> (satisfied, why)` supplies ground truth for a draft
        the critique judged, so the verify gene's records can report the detector's ERROR
        rather than the generator's output. Supply the signal appropriate to the task
        family: for text, the programmatic scorer; for code, the VISIBLE test results
        (never the hidden suite, which would both cost real execution and put the held-out
        signal into the optimizer's channel). This is train-split-only — GEPA's reflection
        channel already sees per-task feedback for the generate gene, and this is the same
        standard optimizer signal made component-appropriate — so it must never be wired
        into held-out scoring, where it would be oracle contamination.
        """
        self.base_genome = base_genome
        self.run_task = run_task
        self.verify_gene = verify_gene
        self.verify_truth = verify_truth
        # Tasks seen by `evaluate`, keyed by id, so `_detector_records` can score a
        # judged draft against the REAL task. GEPA hands `make_reflective_dataset` an
        # `eval_batch` carrying only traces (and re-wraps ours in its own
        # `EvaluationBatch`, dropping any field we add), and a `LoopTrace` records the
        # task's id and summary but not its payload — so the task itself has to be kept
        # here or the truth signal has nothing to score against. Keyed rather than
        # positional because GEPA's multi-task sampling evaluates every parent minibatch
        # BEFORE reflecting on each in turn, so "the last batch" is the wrong batch for
        # all but one of them.
        self._tasks: dict[str, BenchmarkTask] = {}

    def evaluate(
        self, batch: list[BenchmarkTask], candidate: dict[str, str], capture_traces: bool = False
    ) -> EvalBatch:
        genome = genome_from_candidate(self.base_genome, candidate)
        outputs: list[str] = []
        scores: list[float] = []
        traces: list[LoopTrace] = []
        for task in batch:
            # Ids are unique within a split, but NOT across configurations: the
            # multi_constraint generator emits index-based ids, so two seeds produce the
            # same id with different constraints. One adapter only ever sees one split, so
            # this cannot fire in the current wiring — and if it ever did, a silent
            # overwrite would score drafts against the wrong task's constraints, which is
            # invisible in the output. Repeats of an identical task are normal (GEPA
            # re-evaluates the same minibatch each iteration) and pass through.
            known = self._tasks.get(task.id)
            if known is not None and known != task:
                raise ValueError(
                    f"two different tasks share id {task.id!r}; the verify gene's reflection "
                    "records would be scored against the wrong task's constraints"
                )
            self._tasks[task.id] = task
            trace = self.run_task(genome, task)
            outputs.append(_worker_output(trace))
            scores.append(_score_of(trace))
            traces.append(trace)
        return EvalBatch(
            outputs=outputs, scores=scores, trajectories=traces if capture_traces else None
        )

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: EvalBatch, components_to_update: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Build GEPA reflection records with the real task signal.

        Each record surfaces what the reflector needs to propose a better prompt:
        the task instruction (Inputs), the worker's actual output (Generated
        Outputs), and per-task Feedback (which requirements passed/failed). GEPA cannot
        improve a component it can't see the effect of, so these must be the real
        task/output, not internal status codes."""
        records: list[dict[str, Any]] = []
        traj = eval_batch.trajectories or []
        for i, trace in enumerate(traj):
            score = eval_batch.scores[i] if i < len(eval_batch.scores) else _score_of(trace)
            # Presence check, not truthiness: an explicit "" feedback from the callable
            # is a real (if empty) signal and must not be overwritten by the synth string.
            raw_fb = trace.metadata.get("feedback")
            feedback = (
                raw_fb
                if raw_fb is not None
                else f"score={score:.2f}; " + (trace.failure_summary or "succeeded")
            )
            records.append(
                {
                    "Inputs": trace.task_summary or trace.task_id or "",
                    "Generated Outputs": _worker_output(trace),
                    "Feedback": feedback,
                }
            )
        out: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            if component == self.verify_gene and self.verify_truth is not None:
                out[component] = self._detector_records(traj)
            else:
                out[component] = records
        return out

    def _task_for(self, trace: LoopTrace) -> BenchmarkTask:
        """The benchmark task this trace came from, for scoring the drafts it judged.

        Raises rather than substituting a stand-in: a payload-less task makes every
        programmatic scorer return 0.0 by design (an unscored task is not a vacuous
        win), which silently turns the detector's truth signal into the constant
        "nothing was satisfied" — labelling every PASS a false pass and every rejection
        correct, i.e. a directional bias toward a verifier that never accepts. That is
        indistinguishable from a real result in the output, so it must be loud.
        """
        task = self._tasks.get(trace.task_id or "")
        if task is None:
            raise KeyError(
                f"trace {trace.id!r} reports task_id {trace.task_id!r}, which `evaluate` "
                "never saw; the verify gene's reflection records cannot be scored without "
                "the task's payload (run_task must set task_id from the task it was given)"
            )
        return task

    def _detector_records(self, traces: list[LoopTrace]) -> list[dict[str, Any]]:
        """Reflection records for the VERIFY gene: its own verdict, scored against truth.

        A detector is improvable only from its own errors. Each record pairs the draft the
        critique judged with the verdict it returned, and labels that verdict a false pass
        (said PASS on a draft that violated the task), a false alarm (demanded fixes on a
        compliant draft), or correct. Naming the error type explicitly matters because the
        two failure modes call for opposite prompt changes — stricter checking versus less
        trigger-happy rejection — and a reflector shown only "score=0.4" cannot tell which
        way to move.
        """
        assert self.verify_truth is not None  # guarded by the caller
        records: list[dict[str, Any]] = []
        for trace in traces:
            task_text = trace.task_summary or trace.task_id or ""
            task = self._task_for(trace)
            for draft, verdict in _critique_steps(trace):
                satisfied, why = self.verify_truth(draft, task)
                said_pass = verdict.strip().upper().startswith("PASS")
                if said_pass and not satisfied:
                    error = (
                        f"WRONG (false pass): you accepted a draft that violated the task. {why}"
                    )
                elif not said_pass and satisfied:
                    error = "WRONG (false alarm): you demanded fixes on a compliant draft."
                else:
                    error = f"correct verdict ({'pass' if said_pass else 'fix'}). {why}"
                records.append(
                    {
                        "Inputs": f"# Task\n{task_text}\n\n# Draft under judgement\n{draft}",
                        "Generated Outputs": verdict,
                        "Feedback": error,
                    }
                )
        return records
