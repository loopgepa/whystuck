"""Promotion: turn an approved variant into real file changes (M7).

Dry-run (`plan_promotion`) computes diffs against the LIVE files (ADR-004.5) and
touches nothing. `apply_promotion` writes the file-backed genes atomically and
returns a PromotionRecord. Rollback is variant-based: re-promote the previous
variant (its genome still holds the old contents), so no file bodies are stored.

Only file-backed, evolvable genes with a path are written here; runner-backed
genes (retry/stop/etc.) take effect by persisting the variant's genome as the
loop (the caller does that via the store).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from loopgepa.core.diff import render_gene_diff
from loopgepa.core.models import LoopVariant, PromotionRecord
from loopgepa.core.storage import atomic_write_text


class FileChange(BaseModel):
    path: str
    diff: str
    after: str


class PromotionPlan(BaseModel):
    loop_id: str
    variant_id: str
    changes: list[FileChange] = Field(default_factory=list)


def _promotable_files(variant: LoopVariant) -> Iterator[tuple[str, str]]:
    for gene in variant.genome.genes:
        if not (gene.file_backed and gene.evolvable):
            continue
        if gene.path is None or not isinstance(gene.content, str):
            continue
        yield gene.path, gene.content


def plan_promotion(variant: LoopVariant, base_dir: Path) -> PromotionPlan:
    """Dry run: diff each file-backed gene against the live file. No writes."""
    changes: list[FileChange] = []
    for rel, after in _promotable_files(variant):
        target = base_dir / rel
        before = target.read_text(encoding="utf-8") if target.exists() else ""
        if before != after:
            changes.append(
                FileChange(path=rel, diff=render_gene_diff(rel, before, after), after=after)
            )
    return PromotionPlan(loop_id=variant.loop_id, variant_id=variant.id, changes=changes)


def apply_promotion(
    variant: LoopVariant,
    base_dir: Path,
    *,
    approved_by: str | None,
    from_variant_id: str | None,
    timestamp: str,
) -> PromotionRecord:
    """Write the variant's file-backed genes and return a promotion record.

    Call only after explicit approval; the record enables variant-based rollback.
    """
    plan = plan_promotion(variant, base_dir)
    for change in plan.changes:
        atomic_write_text(base_dir / change.path, change.after)
    return PromotionRecord(
        timestamp=timestamp,
        loop_id=variant.loop_id,
        from_variant_id=from_variant_id,
        to_variant_id=variant.id,
        files_changed=[c.path for c in plan.changes],
        approved_by=approved_by,
    )
