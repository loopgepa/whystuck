"""Tests for loopgepa.core.safety — the non-negotiable mutation guards.

These are exactly the properties a launch-day security reviewer will attack:
the ≤3-gene cap and the "never silently weaken a budget" rule (SECURITY.md,
IMPLEMENTATION §13).
"""

import pytest

from loopgepa.core.errors import BudgetIncreaseError, TooManyGenesError
from loopgepa.core.models import (
    GeneType,
    LoopGene,
    LoopGenome,
    MutationProposal,
    MutationType,
)
from loopgepa.core.safety import MAX_GENES_PER_MUTATION, assert_safe, check_mutation_set


def _edit(target: str, after, mid: str) -> MutationProposal:
    return MutationProposal(
        id=mid,
        loop_id="bugfix",
        mutation_type=MutationType.EDIT_GENE,
        target_gene_id=target,
        title="edit",
        rationale="r",
        evidence_trace_ids=["t1"],
        after=after,
    )


class TestGeneCap:
    def test_three_genes_is_allowed(self):
        proposals = [_edit(f"g{i}", "x", f"m{i}") for i in range(MAX_GENES_PER_MUTATION)]
        check_mutation_set(proposals)  # must not raise

    def test_four_genes_is_rejected(self):
        proposals = [_edit(f"g{i}", "x", f"m{i}") for i in range(MAX_GENES_PER_MUTATION + 1)]
        with pytest.raises(TooManyGenesError):
            check_mutation_set(proposals)

    def test_repeated_edits_to_same_gene_count_once(self):
        proposals = [_edit("g0", "a", "m1"), _edit("g0", "b", "m2"), _edit("g1", "c", "m3")]
        check_mutation_set(proposals)  # 2 distinct genes, fine


class TestBudgetGuard:
    def _genome(self, budget: dict) -> LoopGenome:
        return LoopGenome(
            id="bugfix",
            name="B",
            genes=[
                LoopGene(id="budget", type=GeneType.BUDGET_POLICY, name="Budget", content=budget)
            ],
        )

    def test_raising_budget_cap_is_rejected(self):
        genome = self._genome({"max_usd": 1.0})
        proposal = _edit("budget", {"max_usd": 5.0}, "m1")
        with pytest.raises(BudgetIncreaseError):
            assert_safe(genome, proposal)

    def test_lowering_budget_cap_is_allowed(self):
        genome = self._genome({"max_usd": 1.0})
        assert_safe(genome, _edit("budget", {"max_usd": 0.5}, "m1"))  # must not raise

    def test_editing_non_budget_gene_is_unaffected(self):
        genome = LoopGenome(
            id="bugfix",
            name="B",
            genes=[LoopGene(id="rubric", type=GeneType.VERIFIER_RUBRIC, name="R", content="check")],
        )
        assert_safe(genome, _edit("rubric", "check harder", "m1"))  # must not raise
