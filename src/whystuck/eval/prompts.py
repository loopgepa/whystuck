"""Prompt fragments shared by the eval workers (llm_worker + repo_worker).

Both build a prompt from the genome's text genes; the per-gene block format lives
here so a change to how a gene is rendered updates every worker at once.
"""

from __future__ import annotations

from whystuck.core.models import LoopGenome


def render_text_genes(genome: LoopGenome) -> list[str]:
    """One ``# <type>: <name>\\n<content>`` block per string-valued gene."""
    return [
        f"# {gene.type}: {gene.name}\n{gene.content}"
        for gene in genome.genes
        if isinstance(gene.content, str)
    ]
