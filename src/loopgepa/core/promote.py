"""Promotion: turn an approved variant into real file changes (M7).

Dry-run (`plan_promotion`) computes diffs against the LIVE files (ADR-004.5) and
touches nothing. `apply_promotion` writes the file-backed genes in two phases —
stage every file to a temp, then swap them all — so a mid-way failure leaves the
working tree unchanged (no half-applied promotion). Every target path is checked
to stay inside `base_dir` (no traversal / absolute escape). Rollback is
variant-based: re-promote the previous variant, so no file bodies are stored.

Only file-backed, evolvable genes with a path are written here; runner-backed
genes (retry/stop/etc.) take effect by persisting the variant's genome as the
loop (the caller does that via the store).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from loopgepa.core.diff import render_gene_diff
from loopgepa.core.models import LoopVariant, PromotionRecord
from loopgepa.core.storage import _fsync_dir, _write_all


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


def _safe_target(base_root: Path, rel: str) -> Path:
    """Resolve `rel` under `base_root`, refusing absolute paths or `..` escapes."""
    dest = (base_root / rel).resolve()
    if not dest.is_relative_to(base_root):
        raise ValueError(f"gene path escapes base_dir: {rel!r}")
    return dest


def plan_promotion(variant: LoopVariant, base_dir: Path) -> PromotionPlan:
    """Dry run: diff each file-backed gene against the live file. No writes.

    Raises ValueError if any gene path would escape base_dir."""
    base_root = base_dir.resolve()
    changes: list[FileChange] = []
    for rel, after in _promotable_files(variant):
        target = _safe_target(base_root, rel)
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
    """Write the variant's file-backed genes atomically (all-or-nothing) and
    return a promotion record. Call only after explicit approval."""
    base_root = base_dir.resolve()
    plan = plan_promotion(variant, base_dir)

    # Phase 1 — stage every change to a temp file (fail here → nothing swapped).
    staged: list[tuple[Path, Path]] = []
    try:
        for change in plan.changes:
            dest = _safe_target(base_root, change.path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
            try:
                _write_all(fd, change.after.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            staged.append((dest, Path(tmp_name)))
    except BaseException:
        for _, tmp in staged:
            tmp.unlink(missing_ok=True)
        raise

    # Phase 2 — swap them all into place.
    for dest, tmp in staged:
        os.replace(tmp, dest)
    for dest, _ in staged:
        _fsync_dir(dest.parent)

    return PromotionRecord(
        timestamp=timestamp,
        loop_id=variant.loop_id,
        from_variant_id=from_variant_id,
        to_variant_id=variant.id,
        files_changed=[c.path for c in plan.changes],
        approved_by=approved_by,
    )
