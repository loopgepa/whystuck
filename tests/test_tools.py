"""Tests for whystuck MCP tool handlers + server registration (M5).

Handlers are tested directly against a temp store (fast, deterministic); the
server is smoke-tested via FastMCP's in-memory Client — no subprocess, no network
(IMPLEMENTATION §4, S4).
"""

import asyncio
from pathlib import Path

import pytest
from fastmcp import Client

from whystuck.core.errors import TooManyGenesError
from whystuck.core.models import GeneType, LoopGene, LoopGenome, LoopVariant
from whystuck.core.storage import WhyStuckStore
from whystuck.tools import (
    build_server,
    diagnose,
    promote_variant,
    propose_mutations,
    record_trace,
    run_loop,
    save_loop,
)

EXPECTED_TOOLS = {
    "whystuck_diagnose",
    "whystuck_run_loop",
    "whystuck_propose_fix",
    "whystuck_promote",
    "whystuck_save_loop",
    "whystuck_record_trace",
}


def _trace(tid: str, outcome: str, summary: str | None = None) -> dict:
    return {
        "id": tid,
        "loop_id": "bugfix",
        "created_at": "2026-07-20T00:00:00Z",
        "final_outcome": outcome,
        "failure_summary": summary,
    }


def _rubric_genome() -> LoopGenome:
    return LoopGenome(
        id="bugfix",
        name="Bugfix",
        genes=[LoopGene(id="rubric", type=GeneType.VERIFIER_RUBRIC, name="R", content="check")],
    )


def _edit_proposal(target: str, after: str, mid: str) -> dict:
    return {
        "id": mid,
        "loop_id": "bugfix",
        "mutation_type": "edit_gene",
        "target_gene_id": target,
        "title": f"edit {target}",
        "rationale": "evidence says so",
        "evidence_trace_ids": ["t1"],
        "after": after,
    }


class TestTraceTools:
    def test_record_increments_count(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        out = record_trace(store, _trace("t1", "failure", "boom"))
        assert out["trace_count"] == 1

    def test_run_loop_records_and_reports_outcome(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        out = run_loop(
            store,
            loop_id="bugfix",
            worker=["w"],
            verifier=["v"],
            max_retries=0,
            execute=lambda cmd: (0, "ok"),
        )
        assert out["final_outcome"] == "success"
        assert out["trace_count"] == 1

    def test_summarize_and_infer(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        for i in range(6):
            record_trace(store, _trace(f"t{i}", "failure", "repair repeats mistake"))
        assert diagnose(store, "bugfix")["trace_count"] == 6
        modes = diagnose(store, "bugfix")["failure_modes"]
        assert modes[0]["title"] == "repair repeats mistake"


class TestMutationTools:
    def test_propose_saves_unpromoted_draft(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        store.save_loop(_rubric_genome())
        out = propose_mutations(store, "bugfix", [_edit_proposal("rubric", "check harder", "m1")])
        assert out["variant_id"]
        variants = store.list_variants("bugfix")
        assert len(variants) == 1
        assert variants[0].promoted is False

    def test_propose_over_gene_cap_is_rejected(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        store.save_loop(_rubric_genome())
        proposals = [_edit_proposal(f"g{i}", "x", f"m{i}") for i in range(4)]
        with pytest.raises(TooManyGenesError):
            propose_mutations(store, "bugfix", proposals)


class TestPromoteTool:
    def _seed_variant(self, store: WhyStuckStore) -> None:
        skill = LoopGene(
            id="s",
            type=GeneType.SKILL,
            name="Skill",
            path="skills/bugfix/SKILL.md",
            content="new content",
        )
        store.save_loop(_rubric_genome())
        store.save_variant(
            LoopVariant(
                id="v1",
                loop_id="bugfix",
                genome=LoopGenome(id="bugfix", name="Bugfix", genes=[skill]),
                created_at="2026-07-20T00:00:00Z",
            )
        )

    def test_dry_run_reports_without_writing(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path / "store")
        self._seed_variant(store)
        proj = tmp_path / "proj"
        (proj / "skills" / "bugfix").mkdir(parents=True)
        (proj / "skills" / "bugfix" / "SKILL.md").write_text("old content")

        out = promote_variant(store, "bugfix", "v1", base_dir=str(proj), dry_run=True)
        assert out["dry_run"] is True
        assert out["files_to_change"] == ["skills/bugfix/SKILL.md"]
        assert (proj / "skills" / "bugfix" / "SKILL.md").read_text() == "old content"

    def test_apply_writes_and_records(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path / "store")
        self._seed_variant(store)
        proj = tmp_path / "proj"
        (proj / "skills" / "bugfix").mkdir(parents=True)
        (proj / "skills" / "bugfix" / "SKILL.md").write_text("old content")

        out = promote_variant(
            store, "bugfix", "v1", base_dir=str(proj), approved_by="human", dry_run=False
        )
        assert out["promoted"] is True
        assert (proj / "skills" / "bugfix" / "SKILL.md").read_text() == "new content"
        assert len(store.list_promotions("bugfix")) == 1


class TestSaveLoop:
    def test_save_loop(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        out = save_loop(store, _rubric_genome().model_dump(mode="json"))
        assert out["loop_id"] == "bugfix"
        assert store.load_loop("bugfix").id == "bugfix"


class TestServerRegistration:
    def _tools(self, store: WhyStuckStore):
        server = build_server(store)

        async def go():
            async with Client(server) as client:
                return await client.list_tools()

        return asyncio.run(go())

    def test_all_tools_registered(self, tmp_path: Path):
        names = {t.name for t in self._tools(WhyStuckStore(tmp_path))}
        assert names >= EXPECTED_TOOLS

    def test_no_tool_claims_optimized(self, tmp_path: Path):
        # S9: v0.1 must never claim "optimized" (that word is earned in v0.2).
        for tool in self._tools(WhyStuckStore(tmp_path)):
            assert "optimized" not in (tool.description or "").lower()
