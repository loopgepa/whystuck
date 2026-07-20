"""MCP tool layer — thin handlers over loopgepa.core, plus the FastMCP server.

Handlers are plain functions taking a store so they are unit-testable without a
running server; `build_server` wraps them as FastMCP tools. No business logic
lives here (that is in core) and nothing here calls an LLM (ADR: reflection is
the client's job). Nothing is ever described as "optimized" — v0.1 does not
evaluate (ADR-003).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from loopgepa.core.diagnosis import infer_failure_modes, summarize_traces
from loopgepa.core.models import LoopGenome, LoopTrace, LoopVariant, MutationProposal
from loopgepa.core.mutation import apply_mutation
from loopgepa.core.promote import apply_promotion, plan_promotion
from loopgepa.core.runner import Executor, run_command_loop
from loopgepa.core.safety import assert_safe, check_mutation_set
from loopgepa.core.storage import LoopGEPAStore


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def record_trace(store: LoopGEPAStore, trace: dict[str, Any]) -> dict[str, Any]:
    parsed = LoopTrace.model_validate(trace)
    store.append_trace(parsed)
    return {
        "trace_id": parsed.id,
        "loop_id": parsed.loop_id,
        "trace_count": len(store.list_traces(parsed.loop_id)),
    }


def run_loop(
    store: LoopGEPAStore,
    *,
    loop_id: str,
    worker: list[str],
    verifier: list[str],
    max_retries: int = 1,
    variant_id: str | None = None,
    execute: Executor | None = None,
) -> dict[str, Any]:
    extra = {"execute": execute} if execute is not None else {}
    trace = run_command_loop(
        loop_id=loop_id,
        trace_id=_new_id("trace"),
        created_at=_now(),
        worker=worker,
        verifier=verifier,
        max_retries=max_retries,
        variant_id=variant_id,
        **extra,
    )
    store.append_trace(trace)
    return {
        "trace_id": trace.id,
        "final_outcome": trace.final_outcome,
        "retry_count": trace.metrics.retry_count,
        "trace_count": len(store.list_traces(loop_id)),
    }


def summarize(store: LoopGEPAStore, loop_id: str, limit: int = 20) -> dict[str, Any]:
    return summarize_traces(loop_id, store.list_traces(loop_id, limit=limit)).model_dump()


def infer(store: LoopGEPAStore, loop_id: str, limit: int = 20) -> dict[str, Any]:
    modes = infer_failure_modes(store.list_traces(loop_id, limit=limit))
    return {"failure_modes": [m.model_dump() for m in modes]}


def propose_mutations(
    store: LoopGEPAStore, loop_id: str, proposals: list[dict[str, Any]]
) -> dict[str, Any]:
    genome = store.load_loop(loop_id)
    parsed = [MutationProposal.model_validate(p) for p in proposals]
    check_mutation_set(parsed)  # ≤3 genes (raises TooManyGenesError)
    new_genome = genome
    for proposal in parsed:
        assert_safe(new_genome, proposal)  # no silent budget increase
        new_genome = apply_mutation(new_genome, proposal)
    variant = LoopVariant(
        id=_new_id("variant"),
        loop_id=loop_id,
        genome=new_genome,
        created_at=_now(),
        mutations=parsed,
    )
    store.save_variant(variant)
    return {"variant_id": variant.id, "diff_summary": [p.title for p in parsed]}


def list_variants(store: LoopGEPAStore, loop_id: str) -> dict[str, Any]:
    return {
        "variants": [
            {"variant_id": v.id, "promoted": v.promoted, "tags": v.tags}
            for v in store.list_variants(loop_id)
        ]
    }


def promote_variant(
    store: LoopGEPAStore,
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
    prev = next((v.id for v in store.list_variants(loop_id) if v.promoted), None)
    record = apply_promotion(
        variant, base, approved_by=approved_by, from_variant_id=prev, timestamp=_now()
    )
    store.save_loop(variant.genome)  # runner-backed gene changes take effect next run
    store.record_promotion(record)
    variant.promoted = True
    store.save_variant(variant)
    return {
        "promoted": True,
        "dry_run": False,
        "files_changed": record.files_changed,
        "from_variant_id": record.from_variant_id,
    }


def save_loop(store: LoopGEPAStore, genome: dict[str, Any]) -> dict[str, Any]:
    parsed = LoopGenome.model_validate(genome)
    store.save_loop(parsed)
    return {"loop_id": parsed.id, "genes": len(parsed.genes)}


def build_server(store: LoopGEPAStore) -> FastMCP:
    mcp: FastMCP = FastMCP("LoopGEPA")

    @mcp.tool
    def loopgepa_record_trace(trace: dict[str, Any]) -> dict[str, Any]:
        """Record one loop-execution trace locally; returns the loop's trace count."""
        return record_trace(store, trace)

    @mcp.tool
    def loopgepa_run_loop(
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
    def loopgepa_summarize_traces(loop_id: str, limit: int = 20) -> dict[str, Any]:
        """Summarize recent traces: success rate, common failures, avg retries."""
        return summarize(store, loop_id, limit)

    @mcp.tool
    def loopgepa_infer_failure_modes(loop_id: str, limit: int = 20) -> dict[str, Any]:
        """Infer recurring failure modes with evidence trace ids and confidence."""
        return infer(store, loop_id, limit)

    @mcp.tool
    def loopgepa_propose_mutations(loop_id: str, proposals: list[dict[str, Any]]) -> dict[str, Any]:
        """Save a draft variant from mutation proposals (≤3 genes, safety-checked)."""
        return propose_mutations(store, loop_id, proposals)

    @mcp.tool
    def loopgepa_list_variants(loop_id: str) -> dict[str, Any]:
        """List stored variants for a loop and whether each is promoted."""
        return list_variants(store, loop_id)

    @mcp.tool
    def loopgepa_promote_variant(
        loop_id: str,
        variant_id: str,
        base_dir: str = ".",
        approved_by: str = "human",
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Promote a variant. Dry-run by default: returns diffs without writing."""
        return promote_variant(
            store,
            loop_id,
            variant_id,
            base_dir=base_dir,
            approved_by=approved_by,
            dry_run=dry_run,
        )

    @mcp.tool
    def loopgepa_save_loop(genome: dict[str, Any]) -> dict[str, Any]:
        """Register or update a loop genome."""
        return save_loop(store, genome)

    return mcp
