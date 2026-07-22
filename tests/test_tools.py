"""Tests for whystuck MCP tool handlers + server registration (M5).

Handlers are tested directly against a temp store (fast, deterministic); the
server is smoke-tested via FastMCP's in-memory Client — no subprocess, no network
(IMPLEMENTATION §4, S4).
"""

import asyncio
from pathlib import Path

import pytest
from conftest import FIXED_TS, rubric_genome, trace_dict
from fastmcp import Client

from whystuck.core.errors import TooManyGenesError
from whystuck.core.models import GeneType, LoopGene, LoopGenome, LoopVariant
from whystuck.core.storage import WhyStuckStore
from whystuck.tools import (
    build_server,
    diagnose,
    promote,
    propose_fix,
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
        out = record_trace(store, trace_dict("t1", "failure", "boom"))
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
            record_trace(store, trace_dict(f"t{i}", "failure", "repair repeats mistake"))
        assert diagnose(store, "bugfix")["trace_count"] == 6
        modes = diagnose(store, "bugfix")["failure_modes"]
        assert modes[0]["title"] == "repair repeats mistake"


class TestMutationTools:
    def test_propose_saves_unpromoted_draft(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        store.save_loop(rubric_genome())
        out = propose_fix(store, "bugfix", [_edit_proposal("rubric", "check harder", "m1")])
        assert out["variant_id"]
        variants = store.list_variants("bugfix")
        assert len(variants) == 1
        assert variants[0].promoted is False

    def test_propose_returns_server_computed_confidence(self, tmp_path: Path):
        # The recomputed confidence must be reachable by the caller, not just
        # persisted on the variant (review finding #4).
        store = WhyStuckStore(tmp_path)
        store.save_loop(rubric_genome())
        for i in range(6):
            record_trace(store, trace_dict(f"t{i}", "failure", "boom"))
        proposal = _edit_proposal("rubric", "check harder", "m1")
        proposal["evidence_trace_ids"] = ["t0", "t1", "t2"]
        out = propose_fix(store, "bugfix", [proposal])
        assert out["proposals"][0]["confidence"] is not None
        assert out["proposals"][0]["confidence"]["band"] in {"low", "medium", "high"}

    def test_propose_over_gene_cap_is_rejected(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        store.save_loop(rubric_genome())
        proposals = [_edit_proposal(f"g{i}", "x", f"m{i}") for i in range(4)]
        with pytest.raises(TooManyGenesError):
            propose_fix(store, "bugfix", proposals)


class TestPromoteTool:
    def _seed_variant(self, store: WhyStuckStore) -> None:
        skill = LoopGene(
            id="s",
            type=GeneType.SKILL,
            name="Skill",
            path="skills/bugfix/SKILL.md",
            content="new content",
        )
        store.save_loop(rubric_genome())
        store.save_variant(
            LoopVariant(
                id="v1",
                loop_id="bugfix",
                genome=LoopGenome(id="bugfix", name="Bugfix", genes=[skill]),
                created_at=FIXED_TS,
            )
        )

    def test_dry_run_reports_without_writing(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path / "store")
        self._seed_variant(store)
        proj = tmp_path / "proj"
        (proj / "skills" / "bugfix").mkdir(parents=True)
        (proj / "skills" / "bugfix" / "SKILL.md").write_text("old content")

        out = promote(store, "bugfix", "v1", base_dir=str(proj), dry_run=True)
        assert out["dry_run"] is True
        assert out["files_to_change"] == ["skills/bugfix/SKILL.md"]
        assert (proj / "skills" / "bugfix" / "SKILL.md").read_text() == "old content"

    def test_apply_writes_and_records(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path / "store")
        self._seed_variant(store)
        proj = tmp_path / "proj"
        (proj / "skills" / "bugfix").mkdir(parents=True)
        (proj / "skills" / "bugfix" / "SKILL.md").write_text("old content")

        out = promote(store, "bugfix", "v1", base_dir=str(proj), approved_by="human", dry_run=False)
        assert out["promoted"] is True
        assert (proj / "skills" / "bugfix" / "SKILL.md").read_text() == "new content"
        assert len(store.list_promotions("bugfix")) == 1


class TestSaveLoop:
    def test_save_loop(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        out = save_loop(store, rubric_genome().model_dump(mode="json"))
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
        # S9: core tools must never claim "optimized".
        for tool in self._tools(WhyStuckStore(tmp_path)):
            assert "optimized" not in (tool.description or "").lower()


class TestEvalGating:
    def _names(self, store: WhyStuckStore, with_eval: bool) -> set[str]:
        server = build_server(store, with_eval=with_eval)

        async def go():
            async with Client(server) as client:
                return {t.name for t in await client.list_tools()}

        return asyncio.run(go())

    def test_eval_tools_absent_by_default(self, tmp_path: Path):
        assert self._names(WhyStuckStore(tmp_path), with_eval=False) == EXPECTED_TOOLS

    def test_eval_tools_registered_when_enabled(self, tmp_path: Path):
        names = self._names(WhyStuckStore(tmp_path), with_eval=True)
        assert {
            "whystuck_optimize_loop",
            "whystuck_optimize_loop_llm",
            "whystuck_get_frontier",
        } <= names
        assert len(names) == len(EXPECTED_TOOLS) + 3

    def test_get_frontier_empty(self, tmp_path: Path):
        from whystuck.tools import get_frontier

        assert get_frontier(WhyStuckStore(tmp_path), "bugfix") == {"frontier": []}
