"""Tests for whystuck.core.mutation — applying proposals to a genome."""

import pytest
from conftest import edit_proposal

from whystuck.core.errors import GeneNotFoundError, SafetyLockError
from whystuck.core.models import (
    GeneType,
    LoopGene,
    LoopGenome,
    MutationProposal,
    MutationType,
)
from whystuck.core.mutation import apply_mutation


def _genome() -> LoopGenome:
    return LoopGenome(
        id="bugfix",
        name="Bugfix Loop",
        genes=[
            LoopGene(id="rubric", type=GeneType.VERIFIER_RUBRIC, name="Rubric", content="check it"),
            LoopGene(id="retry", type=GeneType.RETRY_POLICY, name="Retry", content="max: 1"),
        ],
    )


def _unlock(target: str, mid: str = "m1") -> MutationProposal:
    return MutationProposal(
        id=mid,
        loop_id="bugfix",
        mutation_type=MutationType.UNLOCK_GENE,
        target_gene_id=target,
        title="unlock",
        rationale="user approved",
        evidence_trace_ids=[],
    )


class TestEditGene:
    def test_edit_replaces_content(self):
        result = apply_mutation(_genome(), edit_proposal("rubric", "check it thoroughly"))
        gene = next(g for g in result.genes if g.id == "rubric")
        assert gene.content == "check it thoroughly"

    def test_edit_does_not_mutate_original(self):
        genome = _genome()
        apply_mutation(genome, edit_proposal("rubric", "changed"))
        assert genome.genes[0].content == "check it"  # original untouched

    def test_edit_missing_gene_raises(self):
        with pytest.raises(GeneNotFoundError):
            apply_mutation(_genome(), edit_proposal("nonexistent", "x"))

    def test_edit_safety_locked_gene_raises(self):
        genome = _genome()
        genome.genes[0].safety_locked = True
        with pytest.raises(SafetyLockError):
            apply_mutation(genome, edit_proposal("rubric", "changed"))


class TestAddRemoveLock:
    def test_add_gene(self):
        proposal = MutationProposal(
            id="m2",
            loop_id="bugfix",
            mutation_type=MutationType.ADD_GENE,
            title="add stop",
            rationale="need a stop condition",
            evidence_trace_ids=[],
            after={
                "id": "stop",
                "type": "stop_condition",
                "name": "Stop",
                "content": "require_tests_pass: true",
            },
        )
        result = apply_mutation(_genome(), proposal)
        assert any(g.id == "stop" for g in result.genes)

    def test_remove_gene(self):
        proposal = MutationProposal(
            id="m3",
            loop_id="bugfix",
            mutation_type=MutationType.REMOVE_GENE,
            target_gene_id="retry",
            title="remove retry",
            rationale="unused",
            evidence_trace_ids=["t1"],
        )
        result = apply_mutation(_genome(), proposal)
        assert all(g.id != "retry" for g in result.genes)

    def test_lock_gene(self):
        proposal = MutationProposal(
            id="m4",
            loop_id="bugfix",
            mutation_type=MutationType.LOCK_GENE,
            target_gene_id="rubric",
            title="lock",
            rationale="safety",
            evidence_trace_ids=[],
        )
        result = apply_mutation(_genome(), proposal)
        assert next(g for g in result.genes if g.id == "rubric").safety_locked is True

    def test_unlock_gene_clears_the_lock(self):
        # The explicit "unlock to allow" action every safety refusal points to
        # (review finding #2) — must be reachable through a real mutation type.
        genome = _genome()
        genome.genes[0].safety_locked = True
        result = apply_mutation(genome, _unlock("rubric"))
        assert next(g for g in result.genes if g.id == "rubric").safety_locked is False

    def test_unlock_then_edit_succeeds(self):
        genome = _genome()
        genome.genes[0].safety_locked = True
        unlocked = apply_mutation(genome, _unlock("rubric"))
        edited = apply_mutation(unlocked, edit_proposal("rubric", "check it thoroughly"))
        assert next(g for g in edited.genes if g.id == "rubric").content == "check it thoroughly"
