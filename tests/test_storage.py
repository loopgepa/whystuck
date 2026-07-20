"""Tests for whystuck.core.storage — local file store (ADR-004)."""

from pathlib import Path

import pytest
from conftest import FIXED_TS

from whystuck.core.errors import NotFoundError
from whystuck.core.models import (
    GeneType,
    LoopGene,
    LoopGenome,
    LoopTrace,
    LoopVariant,
    PromotionRecord,
)
from whystuck.core.storage import WhyStuckStore, resolve_store_root


def _genome(loop_id: str = "bugfix") -> LoopGenome:
    return LoopGenome(
        id=loop_id,
        name="Bugfix Loop",
        genes=[LoopGene(id="verifier", type=GeneType.VERIFIER_RUBRIC, name="V", content="check")],
    )


def _trace(trace_id: str, loop_id: str = "bugfix", outcome: str = "failure") -> LoopTrace:
    return LoopTrace(
        id=trace_id, loop_id=loop_id, created_at=FIXED_TS, final_outcome=outcome
    )


def _variant(variant_id: str, loop_id: str = "bugfix") -> LoopVariant:
    return LoopVariant(
        id=variant_id, loop_id=loop_id, genome=_genome(loop_id), created_at=FIXED_TS
    )


class TestResolveStoreRoot:
    """ADR-004.2: store lives outside the repo — WHYSTUCK_HOME > XDG > ~/.local/share."""

    def test_whystuck_home_wins(self):
        env = {"WHYSTUCK_HOME": "/custom/lg", "XDG_DATA_HOME": "/x"}
        assert resolve_store_root(env) == Path("/custom/lg")

    def test_falls_back_to_xdg_data_home(self):
        env = {"XDG_DATA_HOME": "/x"}
        assert resolve_store_root(env) == Path("/x/whystuck")

    def test_falls_back_to_local_share(self):
        env = {"HOME": "/home/u"}
        assert resolve_store_root(env) == Path("/home/u/.local/share/whystuck")


class TestLoopRoundTrip:
    def test_save_then_load_genome(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        store.save_loop(_genome())
        loaded = store.load_loop("bugfix")
        assert loaded.id == "bugfix"
        assert loaded.genes[0].type is GeneType.VERIFIER_RUBRIC

    def test_load_missing_loop_raises(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        with pytest.raises(NotFoundError):
            store.load_loop("nope")

    def test_list_loops(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        store.save_loop(_genome("a"))
        store.save_loop(_genome("b"))
        assert {g.id for g in store.list_loops()} == {"a", "b"}

    def test_save_loop_is_idempotent_overwrite(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        store.save_loop(_genome())
        g2 = _genome()
        g2.description = "updated"
        store.save_loop(g2)
        assert store.load_loop("bugfix").description == "updated"
        assert len(store.list_loops()) == 1


class TestTraces:
    def test_append_then_list_in_order(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        store.append_trace(_trace("t1"))
        store.append_trace(_trace("t2"))
        ids = [t.id for t in store.list_traces("bugfix")]
        assert ids == ["t1", "t2"]

    def test_traces_are_jsonl_one_line_each(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        store.append_trace(_trace("t1"))
        store.append_trace(_trace("t2"))
        path = tmp_path / "traces" / "bugfix.jsonl"
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_list_traces_limit_returns_most_recent(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        for i in range(5):
            store.append_trace(_trace(f"t{i}"))
        recent = store.list_traces("bugfix", limit=2)
        assert [t.id for t in recent] == ["t3", "t4"]

    def test_list_traces_empty_for_unknown_loop(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        assert store.list_traces("nope") == []


class TestVariants:
    def test_save_load_list_variant(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        store.save_variant(_variant("v1"))
        assert store.load_variant("bugfix", "v1").id == "v1"
        assert [v.id for v in store.list_variants("bugfix")] == ["v1"]


class TestPromotions:
    def test_record_and_list_promotion(self, tmp_path: Path):
        store = WhyStuckStore(tmp_path)
        rec = PromotionRecord(
            timestamp=FIXED_TS,
            loop_id="bugfix",
            from_variant_id="baseline",
            to_variant_id="v1",
            files_changed=["skills/bugfix/SKILL.md"],
            approved_by="human",
        )
        store.record_promotion(rec)
        listed = store.list_promotions("bugfix")
        assert len(listed) == 1
        assert listed[0].to_variant_id == "v1"
