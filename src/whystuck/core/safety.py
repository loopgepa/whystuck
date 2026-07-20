"""Non-negotiable mutation guards (SECURITY.md, IMPLEMENTATION §13).

Two properties enforced here, deliberately conservative and load-bearing:
- a single mutation set may touch at most `MAX_GENES_PER_MUTATION` genes (keeps
  attribution possible — IMPLEMENTATION §13);
- a mutation may never raise, drop, or remove a budget cap on its own.
"""

from __future__ import annotations

from whystuck.core.errors import BudgetIncreaseError, TooManyGenesError
from whystuck.core.models import GeneType, LoopGenome, MutationProposal, MutationType

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


def _assert_no_budget_weakening(before: dict[str, object], after: dict[str, object]) -> None:
    # A numeric cap present in `before` must still be present and not larger.
    # Removing it or raising it weakens the budget. Adding a brand-new cap is fine.
    for key, old_val in before.items():
        if not isinstance(old_val, int | float):
            continue
        if key not in after:
            raise BudgetIncreaseError(f"budget field {key!r} removed; unlock to allow")
        new_val = after[key]
        if isinstance(new_val, int | float) and new_val > old_val:
            raise BudgetIncreaseError(
                f"budget field {key!r} raised {old_val} -> {new_val}; unlock to allow"
            )


def assert_safe(genome: LoopGenome, proposal: MutationProposal) -> None:
    """Guard a single proposal against silently weakening a budget cap.

    Covers raising a cap, dropping a cap key, replacing the content with a
    non-dict, and removing the budget gene entirely.
    """
    gene = next((g for g in genome.genes if g.id == proposal.target_gene_id), None)
    if gene is None or gene.type is not GeneType.BUDGET_POLICY:
        return
    if proposal.mutation_type is MutationType.REMOVE_GENE:
        raise BudgetIncreaseError("removing a budget_policy gene weakens the cap; unlock to allow")
    if proposal.mutation_type is MutationType.EDIT_GENE:
        if not isinstance(proposal.after, dict):
            raise BudgetIncreaseError("budget_policy content must remain a dict")
        before = gene.content if isinstance(gene.content, dict) else {}
        _assert_no_budget_weakening(before, proposal.after)
