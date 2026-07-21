"""Shared test constants and factories.

Kills the timestamp literal that was hardcoded across the suite and the
near-duplicate trace/genome/proposal builders that had drifted between modules.
Import from here: ``from conftest import FIXED_TS, trace_dict, rubric_genome``.
"""

from __future__ import annotations

from typing import Any

from whystuck.core.models import (
    GeneType,
    LoopGene,
    LoopGenome,
    MutationProposal,
    MutationType,
)

# One fixed instant so tests never depend on the wall clock.
FIXED_TS = "2026-07-20T00:00:00Z"


def trace_dict(tid: str, outcome: str = "failure", summary: str | None = "boom") -> dict[str, Any]:
    """A LoopTrace as a plain dict, for handlers that validate raw input."""
    return {
        "id": tid,
        "loop_id": "bugfix",
        "created_at": FIXED_TS,
        "final_outcome": outcome,
        "failure_summary": summary,
    }


def rubric_genome() -> LoopGenome:
    """A minimal loop: one verifier_rubric gene."""
    return LoopGenome(
        id="bugfix",
        name="Bugfix",
        genes=[LoopGene(id="rubric", type=GeneType.VERIFIER_RUBRIC, name="R", content="check")],
    )


def edit_proposal(target: str, after: str | dict[str, Any], mid: str = "m1") -> MutationProposal:
    """An edit_gene MutationProposal citing one trace."""
    return MutationProposal(
        id=mid,
        loop_id="bugfix",
        mutation_type=MutationType.EDIT_GENE,
        target_gene_id=target,
        title="edit",
        rationale="because",
        evidence_trace_ids=["t1"],
        after=after,
    )
