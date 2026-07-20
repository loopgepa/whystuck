"""Tests for loopgepa.core.promote — dry-run + reversible file promotion (M7).

Diffs are computed against the LIVE file, never a stored snapshot (ADR-004.5).
Dry-run must never touch disk.
"""

from pathlib import Path

from loopgepa.core.models import GeneType, LoopGene, LoopGenome, LoopVariant
from loopgepa.core.promote import apply_promotion, plan_promotion


def _variant(*genes: LoopGene, variant_id: str = "v1") -> LoopVariant:
    return LoopVariant(
        id=variant_id,
        loop_id="bugfix",
        genome=LoopGenome(id="bugfix", name="Bugfix", genes=list(genes)),
        created_at="2026-07-20T00:00:00Z",
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


class TestApplyPromotion:
    def test_writes_file_content(self, tmp_path: Path):
        apply_promotion(
            _variant(_skill("promoted content")),
            tmp_path,
            approved_by="human",
            from_variant_id="baseline",
            timestamp="2026-07-20T00:00:00Z",
        )
        written = tmp_path / "skills" / "bugfix" / "SKILL.md"
        assert written.read_text() == "promoted content"

    def test_returns_promotion_record(self, tmp_path: Path):
        record = apply_promotion(
            _variant(_skill("x")),
            tmp_path,
            approved_by="human",
            from_variant_id="baseline",
            timestamp="2026-07-20T00:00:00Z",
        )
        assert record.to_variant_id == "v1"
        assert record.from_variant_id == "baseline"
        assert record.approved_by == "human"
        assert record.files_changed == ["skills/bugfix/SKILL.md"]
