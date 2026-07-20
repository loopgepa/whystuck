"""Render human-readable unified diffs for gene content changes."""

from __future__ import annotations

import difflib


def render_gene_diff(gene_name: str, before: str, after: str) -> str:
    """Unified diff between a gene's before/after content. Empty string if equal."""
    if before == after:
        return ""
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"{gene_name} (before)",
        tofile=f"{gene_name} (after)",
        lineterm="",
    )
    return "\n".join(lines)
