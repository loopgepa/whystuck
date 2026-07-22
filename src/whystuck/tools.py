"""MCP tool layer — thin handlers over whystuck.core, plus the FastMCP server.

Handlers are plain functions taking a store so they are unit-testable without a
running server; `build_server` wraps them as FastMCP tools. The core diagnosis/
mutation tools call no LLM (ADR: reflection is the client's job). The v0.2 eval
tools are registered only when the optional `eval` extra is installed, so the
default surface stays minimal (6 tools) — only they may use the word "optimized",
and only after measured evaluation (ADR-003, ADR-011).
"""

from __future__ import annotations

import importlib.util
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from whystuck.core.diagnosis import infer_failure_modes, summarize_traces
from whystuck.core.errors import NotFoundError
from whystuck.core.models import (
    Confidence,
    ConfidenceBand,
    LoopGenome,
    LoopTrace,
    LoopVariant,
    MutationProposal,
    assess_confidence,
)
from whystuck.core.mutation import apply_mutation
from whystuck.core.promote import apply_promotion, plan_promotion
from whystuck.core.runner import Executor, run_command_loop
from whystuck.core.safety import assert_safe, check_mutation_set
from whystuck.core.storage import WhyStuckStore
from whystuck.core.util import utc_now_iso


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def record_trace(store: WhyStuckStore, trace: dict[str, Any]) -> dict[str, Any]:
    parsed = LoopTrace.model_validate(trace)
    store.append_trace(parsed)
    return {
        "trace_id": parsed.id,
        "loop_id": parsed.loop_id,
        "trace_count": len(store.list_traces(parsed.loop_id)),
    }


def run_loop(
    store: WhyStuckStore,
    *,
    loop_id: str,
    worker: list[str],
    verifier: list[str],
    max_retries: int = 1,
    variant_id: str | None = None,
    execute: Executor | None = None,
) -> dict[str, Any]:
    trace = run_command_loop(
        loop_id=loop_id,
        trace_id=_new_id("trace"),
        created_at=utc_now_iso(),
        worker=worker,
        verifier=verifier,
        max_retries=max_retries,
        variant_id=variant_id,
        execute=execute,
    )
    store.append_trace(trace)
    return {
        "trace_id": trace.id,
        "final_outcome": trace.final_outcome,
        "retry_count": trace.metrics.retry_count,
        "trace_count": len(store.list_traces(loop_id)),
    }


def diagnose(store: WhyStuckStore, loop_id: str, limit: int = 20) -> dict[str, Any]:
    """One-shot diagnosis: trace summary + ranked failure modes with confidence.

    Merges the former summarize/infer tools so the agent picks one obvious tool
    (smaller surface → higher tool-selection accuracy)."""
    traces = store.list_traces(loop_id, limit=limit)
    result = summarize_traces(loop_id, traces).model_dump()
    result["failure_modes"] = [m.model_dump() for m in infer_failure_modes(traces)]
    return result


def _confidence_for_evidence(
    store: WhyStuckStore, loop_id: str, evidence_ids: list[str]
) -> Confidence:
    """Deterministic, server-computed confidence (ADR-005). Never trust a
    client-supplied value; evidence that names no real trace is always LOW.

    A client must not be able to inflate confidence by (a) repeating one real
    trace id many times, or (b) citing real-but-non-failing trace ids. So we
    dedupe the cited ids AND match/count them against the set of FAILING trace
    ids (falling back to all trace ids only when there are zero failing
    traces) — that same set is also the denominator, so cluster_share stays a
    faithful fraction of distinct failing traces actually cited."""
    if not evidence_ids:
        return Confidence(band=ConfidenceBand.LOW, score=0.0)
    traces = store.list_traces(loop_id)
    failing_ids = {t.id for t in traces if t.final_outcome == "failure"}
    reference_ids = failing_ids or {t.id for t in traces}
    matched = {e for e in evidence_ids if e in reference_ids}
    if not matched:
        # Citing only unknown/fabricated/non-failing trace ids is no evidence
        # at all — force LOW, not the MEDIUM/HIGH that raw trace volume alone
        # would grant.
        return Confidence(band=ConfidenceBand.LOW, score=0.0)
    denom = len(reference_ids)
    cluster_share = min(1.0, len(matched) / denom) if denom else 0.0
    return assess_confidence(n_traces=len(traces), cluster_share=cluster_share)


def propose_fix(
    store: WhyStuckStore, loop_id: str, proposals: list[dict[str, Any]]
) -> dict[str, Any]:
    genome = store.load_loop(loop_id)
    parsed = [MutationProposal.model_validate(p) for p in proposals]
    check_mutation_set(parsed)  # ≤3 genes (raises TooManyGenesError)
    new_genome = genome
    for proposal in parsed:
        assert_safe(new_genome, proposal)  # no silent budget increase
        # Overwrite any client-supplied confidence with the server-computed value.
        proposal.confidence = _confidence_for_evidence(store, loop_id, proposal.evidence_trace_ids)
        new_genome = apply_mutation(new_genome, proposal)
    variant = LoopVariant(
        id=_new_id("variant"),
        loop_id=loop_id,
        genome=new_genome,
        created_at=utc_now_iso(),
        mutations=parsed,
    )
    store.save_variant(variant)
    return {
        "variant_id": variant.id,
        "diff_summary": [p.title for p in parsed],
        # Surface the server-computed confidence so the caller can act on it — it
        # is otherwise only persisted on the variant and unreachable by any tool.
        "proposals": [
            {
                "title": p.title,
                "mutation_type": p.mutation_type.value,
                "confidence": p.confidence.model_dump() if p.confidence else None,
            }
            for p in parsed
        ],
    }


def _snapshot_genesis(store: WhyStuckStore, loop_id: str) -> str | None:
    """Save the current live loop genome as a 'genesis' variant and return its id.

    A loop's first promotion otherwise has no prior variant to name, so it could
    never be rolled back. Snapshotting the pre-promotion state gives rollback a
    real, re-promotable target. Returns None if the loop was never saved."""
    try:
        genome = store.load_loop(loop_id)
    except NotFoundError:
        return None
    genesis = LoopVariant(
        id=_new_id("genesis"),
        loop_id=loop_id,
        genome=genome,
        created_at=utc_now_iso(),
        tags=["genesis"],
    )
    store.save_variant(genesis)
    return genesis.id


def promote(
    store: WhyStuckStore,
    loop_id: str,
    variant_id: str,
    *,
    base_dir: str = ".",
    approved_by: str = "human",
    dry_run: bool = True,
) -> dict[str, Any]:
    variant = store.load_variant(loop_id, variant_id)
    base = Path(base_dir)
    if dry_run:
        plan = plan_promotion(variant, base)
        return {
            "promoted": False,
            "dry_run": True,
            "files_to_change": [c.path for c in plan.changes],
            "diffs": [c.diff for c in plan.changes],
        }
    # The current live variant is the target of the last promotion (ordered log),
    # not the alphabetically-first flagged variant. On the first promotion there is
    # none, so snapshot the pre-promotion loop as a genesis fallback — that makes
    # even the first promotion reversible (rollback = re-promote the genesis variant).
    promotions = store.list_promotions(loop_id)
    prev = promotions[-1].to_variant_id if promotions else _snapshot_genesis(store, loop_id)
    record = apply_promotion(
        variant, base, approved_by=approved_by, from_variant_id=prev, timestamp=utc_now_iso()
    )
    store.save_loop(variant.genome)  # runner-backed gene changes take effect next run
    store.record_promotion(record)
    # Keep exactly one promoted variant: clear the previous one, flag this one.
    if prev is not None and prev != variant.id:
        prev_variant = store.load_variant(loop_id, prev)
        prev_variant.promoted = False
        store.save_variant(prev_variant)
    variant.promoted = True
    store.save_variant(variant)
    return {
        "promoted": True,
        "dry_run": False,
        "files_changed": record.files_changed,
        "from_variant_id": record.from_variant_id,
    }


def save_loop(store: WhyStuckStore, genome: dict[str, Any]) -> dict[str, Any]:
    parsed = LoopGenome.model_validate(genome)
    store.save_loop(parsed)
    return {"loop_id": parsed.id, "genes": len(parsed.genes)}


def get_frontier(store: WhyStuckStore, loop_id: str) -> dict[str, Any]:
    return {"frontier": [c.model_dump() for c in store.load_frontier(loop_id)]}


def optimize_loop_via_commands(
    store: WhyStuckStore,
    loop_id: str,
    *,
    reflection_lm: str,
    worker: list[str],
    verifier: list[str],
    max_retries: int = 1,
    tasks_dir: str | None = None,
    max_metric_calls: int = 50,
) -> dict[str, Any]:
    """Run GEPA over the loop's text genes, scoring each candidate by executing a
    worker+verifier per benchmark task. Requires `whystuck[eval]` + a network
    reflection LM. Alpha: most meaningful once the worker is LLM-backed (§7.2)."""
    from whystuck.eval.benchmark import benchmark_from_dir, benchmark_from_traces
    from whystuck.eval.optimize import optimize_loop

    tasks = (
        benchmark_from_dir(Path(tasks_dir))
        if tasks_dir
        else benchmark_from_traces(store.list_traces(loop_id))
    )

    # The command runner ignores the candidate genome (the worker/verifier are
    # fixed), so `_genome` is intentionally unused — text-gene search only bites
    # with the LLM-backed worker (eval/llm_worker.py).
    def run_task(_genome: LoopGenome, task: Any) -> LoopTrace:
        return run_command_loop(
            loop_id=loop_id,
            trace_id=f"opt-{task.id}",
            created_at=utc_now_iso(),
            worker=worker,
            verifier=verifier,
            max_retries=max_retries,
        )

    return optimize_loop(
        store,
        loop_id,
        tasks,
        run_task,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
    )


def optimize_loop_via_llm(
    store: WhyStuckStore,
    loop_id: str,
    *,
    complete: Callable[[str], str],
    reflection_lm: str,
    score: Callable[[str, Any], bool] | None = None,
    tasks_dir: str | None = None,
    max_metric_calls: int = 50,
    _optimize: Any | None = None,
) -> dict[str, Any]:
    """Run GEPA over the loop's text genes, scoring each candidate by running an
    LLM-backed worker whose prompt is built *from* the candidate genome. Unlike
    the command path, the score depends on the genome, so GEPA has a real
    gradient to climb (§7.2). `complete` is the (network) LLM call; `score`
    decides task success (defaults to `keyword_score` over each task's declared
    `expect`). Requires `whystuck[eval]` + a network reflection LM."""
    from whystuck.eval.benchmark import benchmark_from_dir, benchmark_from_traces
    from whystuck.eval.llm_worker import keyword_score, make_run_task
    from whystuck.eval.optimize import optimize_loop

    tasks = (
        benchmark_from_dir(Path(tasks_dir))
        if tasks_dir
        else benchmark_from_traces(store.list_traces(loop_id))
    )
    run_task = make_run_task(complete, score or keyword_score)
    return optimize_loop(
        store,
        loop_id,
        tasks,
        run_task,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        _optimize=_optimize,
    )


def _eval_available() -> bool:
    return importlib.util.find_spec("gepa") is not None


def build_server(store: WhyStuckStore, *, with_eval: bool | None = None) -> FastMCP:
    mcp: FastMCP = FastMCP("WhyStuck")

    @mcp.tool
    def whystuck_run_loop(
        loop_id: str,
        worker: list[str],
        verifier: list[str],
        max_retries: int = 1,
        variant_id: str | None = None,
    ) -> dict[str, Any]:
        """Run a declared loop's worker+verifier commands offline and record a trace."""
        return run_loop(
            store,
            loop_id=loop_id,
            worker=worker,
            verifier=verifier,
            max_retries=max_retries,
            variant_id=variant_id,
        )

    @mcp.tool
    def whystuck_diagnose(loop_id: str, limit: int = 20) -> dict[str, Any]:
        """Diagnose why a loop keeps failing: success rate + ranked failure modes
        with evidence trace ids and a deterministic confidence. Start here."""
        return diagnose(store, loop_id, limit)

    @mcp.tool
    def whystuck_propose_fix(loop_id: str, proposals: list[dict[str, Any]]) -> dict[str, Any]:
        """Draft a reversible fix from mutation proposals (≤3 genes; budget caps
        can't be weakened, and safety-locked genes can't be edited while locked).
        Confidence is recomputed server-side from the cited evidence and returned
        per proposal."""
        return propose_fix(store, loop_id, proposals)

    @mcp.tool
    def whystuck_promote(
        loop_id: str,
        variant_id: str,
        base_dir: str = ".",
        approved_by: str = "human",
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Apply a fix. Dry-run by default: returns diffs without writing. A real
        write needs dry_run=false and is reversible via a rollback record."""
        return promote(
            store,
            loop_id,
            variant_id,
            base_dir=base_dir,
            approved_by=approved_by,
            dry_run=dry_run,
        )

    @mcp.tool
    def whystuck_save_loop(genome: dict[str, Any]) -> dict[str, Any]:
        """Declare or update a loop (its genome) so it can be run and diagnosed."""
        return save_loop(store, genome)

    @mcp.tool
    def whystuck_record_trace(trace: dict[str, Any]) -> dict[str, Any]:
        """(internal) Record one loop-execution trace; usually called by run_loop
        or a session hook, not directly by the agent."""
        return record_trace(store, trace)

    # v0.2 eval tools — registered only when the `eval` extra is installed, so the
    # default surface stays a minimal 6 (ADR-013 principle 4).
    if with_eval if with_eval is not None else _eval_available():

        @mcp.tool
        def whystuck_optimize_loop(
            loop_id: str,
            reflection_lm: str,
            worker: list[str],
            verifier: list[str],
            max_retries: int = 1,
            tasks_dir: str | None = None,
            max_metric_calls: int = 50,
        ) -> dict[str, Any]:
            """Run GEPA using fixed worker+verifier commands to score candidates.
            The command outcome does not depend on the candidate's text genes, so
            this path cannot improve prose genes — use whystuck_optimize_loop_llm
            for that. Kept for scoring command-driven loops. Real GEPA (network)."""
            return optimize_loop_via_commands(
                store,
                loop_id,
                reflection_lm=reflection_lm,
                worker=worker,
                verifier=verifier,
                max_retries=max_retries,
                tasks_dir=tasks_dir,
                max_metric_calls=max_metric_calls,
            )

        @mcp.tool
        def whystuck_optimize_loop_llm(
            loop_id: str,
            model: str,
            reflection_lm: str,
            tasks_dir: str | None = None,
            max_metric_calls: int = 50,
        ) -> dict[str, Any]:
            """Optimize a loop's text genes with GEPA, scoring each candidate by
            running an LLM-backed worker (via `model`) whose prompt is built from
            the candidate genome — so the genes actually move the score, unlike
            the command path. Declare tasks with an `expect` substring in
            tasks_dir. Runs real GEPA + LLM calls (network)."""
            from whystuck.eval.llm_worker import litellm_complete

            return optimize_loop_via_llm(
                store,
                loop_id,
                complete=litellm_complete(model),
                reflection_lm=reflection_lm,
                tasks_dir=tasks_dir,
                max_metric_calls=max_metric_calls,
            )

        @mcp.tool
        def whystuck_get_frontier(loop_id: str) -> dict[str, Any]:
            """Return the stored optimization frontier (evaluated variants)."""
            return get_frontier(store, loop_id)

    return mcp
