"""Non-negotiable mutation guards (SECURITY.md, IMPLEMENTATION §13).

Two properties enforced here, deliberately conservative and load-bearing:
- a single mutation set may touch at most `MAX_GENES_PER_MUTATION` genes (keeps
  attribution possible — risk-register 17.3);
- a mutation may never raise a budget cap on its own.
"""

from __future__ import annotations

from loopgepa.core.errors import BudgetIncreaseError, TooManyGenesError
from loopgepa.core.models import GeneType, LoopGenome, MutationProposal, MutationType

# v0 caps mutations to 3 genes so a result stays attributable to a specific change.
MAX_GENES_PER_MUTATION = 3


def _affected_gene_id(proposal: MutationProposal) -> str | None:
    if proposal.mutation_type is MutationType.ADD_GENE and isinstance(proposal.after, dict):
        gid = proposal.after.get("id")
        return gid if isinstance(gid, str) else None
    return proposal.target_gene_id


def check_mutation_set(proposals: list[MutationProposal]) -> None:
    """Reject a variant whose proposals touch more than the gene cap."""
    genes = {gid for p in proposals if (gid := _affected_gene_id(p)) is not None}
    if len(genes) > MAX_GENES_PER_MUTATION:
        raise TooManyGenesError(
            f"mutation touches {len(genes)} genes; cap is {MAX_GENES_PER_MUTATION}"
        )


def _assert_no_numeric_increase(before: dict[str, object], after: dict[str, object]) -> None:
    for key, new_val in after.items():
        old_val = before.get(key)
        if not (isinstance(new_val, int | float) and isinstance(old_val, int | float)):
            continue
        if new_val > old_val:
            raise BudgetIncreaseError(
                f"budget field {key!r} raised {old_val} -> {new_val}; unlock to allow"
            )


def assert_safe(genome: LoopGenome, proposal: MutationProposal) -> None:
    """Guard a single proposal against silently weakening a budget cap."""
    if proposal.mutation_type is not MutationType.EDIT_GENE:
        return
    gene = next((g for g in genome.genes if g.id == proposal.target_gene_id), None)
    if gene is None or gene.type is not GeneType.BUDGET_POLICY:
        return
    if isinstance(gene.content, dict) and isinstance(proposal.after, dict):
        _assert_no_numeric_increase(gene.content, proposal.after)
