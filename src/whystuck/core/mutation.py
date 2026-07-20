"""Apply a MutationProposal to a genome, returning a NEW genome (ADR-002).

The input genome is never mutated in place, so lineage stays inspectable. Editing
or removing a `safety_locked` gene is refused unless the user first unlocks it
(SECURITY.md §Safety-locked genes).
"""

from __future__ import annotations

from whystuck.core.errors import GeneNotFoundError, SafetyLockError
from whystuck.core.models import LoopGene, LoopGenome, MutationProposal, MutationType


def apply_mutation(genome: LoopGenome, proposal: MutationProposal) -> LoopGenome:
    new = genome.model_copy(deep=True)

    # MutationProposal's validator guarantees the per-type field shape, so we can
    # trust `after`/`target_gene_id` here.
    if proposal.mutation_type is MutationType.ADD_GENE:
        gene = LoopGene.model_validate(proposal.after)
        if any(g.id == gene.id for g in new.genes):
            raise ValueError(f"gene id already exists: {gene.id!r}")
        new.genes.append(gene)
        return new

    target = proposal.target_gene_id
    idx = next((i for i, g in enumerate(new.genes) if g.id == target), None)
    if idx is None:
        raise GeneNotFoundError(f"gene not found: {target!r}")
    gene = new.genes[idx]

    if proposal.mutation_type is MutationType.EDIT_GENE:
        if gene.safety_locked:
            raise SafetyLockError(f"gene {gene.id!r} is safety-locked; unlock before editing")
        gene.content = proposal.after  # type: ignore[assignment]
    elif proposal.mutation_type is MutationType.REMOVE_GENE:
        if gene.safety_locked:
            raise SafetyLockError(f"gene {gene.id!r} is safety-locked; unlock before removing")
        del new.genes[idx]
    elif proposal.mutation_type is MutationType.LOCK_GENE:
        gene.safety_locked = True

    return new
