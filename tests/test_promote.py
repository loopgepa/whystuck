"""Tests for whystuck.core.promote — dry-run + reversible file promotion (M7).

Diffs are computed against the LIVE file, never a stored snapshot (ADR-004.5).
Dry-run must never touch disk.
"""

from pathlib import Path

import pytest
from conftest import FIXED_TS

from whystuck.core.errors import PromotionSwapError, UnpromotableGeneError
from whystuck.core.models import GeneType, LoopGene, LoopGenome, LoopVariant
from whystuck.core.promote import apply_promotion, plan_promotion


def _variant(*genes: LoopGene, variant_id: str = "v1") -> LoopVariant:
    return LoopVariant(
        id=variant_id,
        loop_id="bugfix",
        genome=LoopGenome(id="bugfix", name="Bugfix", genes=list(genes)),
        created_at=FIXED_TS,
    )


def _skill(content: str) -> LoopGene:
    return LoopGene(
        id="bugfix_skill",
        type=GeneType.SKILL,
        name="Bugfix Skill",
        path="skills/bugfix/SKILL.md",
        content=content,
    )


class TestPlanPromotion:
    def test_reports_change_against_live_file(self, tmp_path: Path):
        live = tmp_path / "skills" / "bugfix" / "SKILL.md"
        live.parent.mkdir(parents=True)
        live.write_text("old instructions")
        plan = plan_promotion(_variant(_skill("new instructions")), tmp_path)
        assert len(plan.changes) == 1
        assert plan.changes[0].path == "skills/bugfix/SKILL.md"
        assert "-old instructions" in plan.changes[0].diff
        assert "+new instructions" in plan.changes[0].diff

    def test_no_change_when_content_matches_live(self, tmp_path: Path):
        live = tmp_path / "skills" / "bugfix" / "SKILL.md"
        live.parent.mkdir(parents=True)
        live.write_text("same")
        plan = plan_promotion(_variant(_skill("same")), tmp_path)
        assert plan.changes == []

    def test_ignores_runner_backed_genes(self, tmp_path: Path):
        retry = LoopGene(id="retry", type=GeneType.RETRY_POLICY, name="Retry", content="max: 2")
        plan = plan_promotion(_variant(retry), tmp_path)
        assert plan.changes == []

    def test_dry_run_does_not_write(self, tmp_path: Path):
        plan_promotion(_variant(_skill("brand new")), tmp_path)
        assert not (tmp_path / "skills" / "bugfix" / "SKILL.md").exists()

    def test_dict_content_on_file_backed_gene_raises_not_silently_skipped(self, tmp_path: Path):
        # A file-backed + evolvable gene with non-str content can never be
        # written to disk — that must be a loud error, not a silently empty
        # diff (review finding: promote reports success while doing nothing).
        bad = LoopGene(
            id="bugfix_skill",
            type=GeneType.SKILL,
            name="Bugfix Skill",
            path="skills/bugfix/SKILL.md",
            content={"not": "a string"},
        )
        with pytest.raises(UnpromotableGeneError, match="bugfix_skill"):
            plan_promotion(_variant(bad), tmp_path)


class TestApplyPromotion:
    def test_writes_file_content(self, tmp_path: Path):
        apply_promotion(
            _variant(_skill("promoted content")),
            tmp_path,
            approved_by="human",
            from_variant_id="baseline",
            timestamp=FIXED_TS,
        )
        written = tmp_path / "skills" / "bugfix" / "SKILL.md"
        assert written.read_text() == "promoted content"

    def test_returns_promotion_record(self, tmp_path: Path):
        record = apply_promotion(
            _variant(_skill("x")),
            tmp_path,
            approved_by="human",
            from_variant_id="baseline",
            timestamp=FIXED_TS,
        )
        assert record.to_variant_id == "v1"
        assert record.from_variant_id == "baseline"
        assert record.approved_by == "human"
        assert record.files_changed == ["skills/bugfix/SKILL.md"]

    def test_dict_content_on_file_backed_gene_raises_real_promotion_too(self, tmp_path: Path):
        bad = LoopGene(
            id="bugfix_skill",
            type=GeneType.SKILL,
            name="Bugfix Skill",
            path="skills/bugfix/SKILL.md",
            content={"not": "a string"},
        )
        with pytest.raises(UnpromotableGeneError, match="bugfix_skill"):
            apply_promotion(
                _variant(bad),
                tmp_path,
                approved_by="human",
                from_variant_id="baseline",
                timestamp=FIXED_TS,
            )

    def test_swap_failure_unlinks_pending_temps_and_names_swapped_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # os.replace is atomic per file, but the swap loop across several
        # files is not — a failure partway through must not leak the
        # not-yet-swapped staged temp files, and must say exactly which dests
        # made it and which did not (module docstring's honesty fix).
        import os as os_module

        second = _skill("second")
        second = second.model_copy(update={"id": "second_skill", "path": "skills/bugfix/SECOND.md"})
        variant = _variant(_skill("first"), second)

        real_replace = os_module.replace
        calls = {"n": 0}

        def flaky_replace(src, dst):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk fell over")
            return real_replace(src, dst)

        monkeypatch.setattr("whystuck.core.promote.os.replace", flaky_replace)

        with pytest.raises(PromotionSwapError) as excinfo:
            apply_promotion(
                variant,
                tmp_path,
                approved_by="human",
                from_variant_id="baseline",
                timestamp=FIXED_TS,
            )

        message = str(excinfo.value)
        assert "skills/bugfix/SKILL.md" in message  # swapped
        assert "skills/bugfix/SECOND.md" in message  # not swapped

        # The first dest was really swapped; no leftover temp remains for
        # either target directory.
        assert (tmp_path / "skills" / "bugfix" / "SKILL.md").read_text() == "first"
        assert not (tmp_path / "skills" / "bugfix" / "SECOND.md").exists()
        leftover_temps = list((tmp_path / "skills" / "bugfix").glob(".*.tmp"))
        assert leftover_temps == []
