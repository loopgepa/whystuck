"""Promotion: turn an approved variant into real file changes (M7).

Dry-run (`plan_promotion`) computes diffs against the LIVE files (ADR-004.5) and
touches nothing. `apply_promotion` writes the file-backed genes in two phases —
stage every file to a temp, then swap them all into place. Phase 1 (staging) is
all-or-nothing: a failure there touches no live file. Each swap in phase 2
(`os.replace`) is atomic *per file*, but true atomicity across several files is
not possible on POSIX — a failure mid-swap can leave a partial promotion (some
dests swapped, some not). On that failure the not-yet-swapped staged temps are
unlinked and the error names exactly which dests were swapped vs not, so the
caller can reason about (and correct, e.g. by re-promoting) the partial state.
Every target path is checked to stay inside `base_dir` (no traversal / absolute
escape). Rollback is variant-based: re-promote the previous variant, so no file
bodies are stored.

Only file-backed, evolvable genes with a path are written here; runner-backed
genes (retry/stop/etc.) take effect by persisting the variant's genome as the
loop (the caller does that via the store).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from whystuck.core.diff import render_gene_diff
from whystuck.core.errors import PathEscapeError, PromotionSwapError, UnpromotableGeneError
from whystuck.core.models import LoopVariant, PromotionRecord
from whystuck.core.storage import _fsync_dir, _stage_temp


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
            # A file-backed, evolvable gene is meant to be written to disk; one
            # with no path or non-str content can never actually be promoted.
            # Silently skipping it would let `promote` report success while
            # never writing the intended change — raise instead.
            raise UnpromotableGeneError(
                f"gene {gene.id!r} is file-backed and evolvable but has "
                f"path={gene.path!r} content_type={type(gene.content).__name__}; "
                "it cannot be promoted"
            )
        yield gene.path, gene.content


def _safe_target(base_root: Path, rel: str) -> Path:
    """Resolve `rel` under `base_root`, refusing absolute paths or `..` escapes."""
    dest = (base_root / rel).resolve()
    if not dest.is_relative_to(base_root):
        raise PathEscapeError(f"gene path escapes base_dir: {rel!r}")
    return dest


def plan_promotion(variant: LoopVariant, base_dir: Path) -> PromotionPlan:
    """Dry run: diff each file-backed gene against the live file. No writes.

    Raises PathEscapeError if any gene path would escape base_dir."""
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
            staged.append((dest, _stage_temp(dest, change.after.encode("utf-8"))))
    except BaseException:
        for _, tmp in staged:
            tmp.unlink(missing_ok=True)
        raise

    # Phase 2 — swap them all into place. Each os.replace is atomic on its own,
    # but the loop across files is not (see module docstring); on failure,
    # unlink the temps that never got swapped (they'd otherwise leak) and name
    # exactly which dests were swapped vs not so the caller can reason about
    # the partial state.
    swapped: list[Path] = []
    try:
        for dest, tmp in staged:
            os.replace(tmp, dest)
            swapped.append(dest)
    except OSError as exc:
        not_swapped = [dest for dest, tmp in staged if dest not in swapped]
        for dest, tmp in staged:
            if dest not in swapped:
                tmp.unlink(missing_ok=True)
        raise PromotionSwapError(
            f"promotion swap failed after {len(swapped)}/{len(staged)} files — "
            f"swapped={[str(d) for d in swapped]!r} "
            f"not_swapped={[str(d) for d in not_swapped]!r}: {exc}"
        ) from exc
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
