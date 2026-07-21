"""Regression tests for issues found across the multi-agent code reviews.

Each test pins a specific finding (path traversal, limit handling, promotion
lineage + first-promotion rollback, server-side confidence, budget-guard bypass,
runner robustness, validators)."""

from pathlib import Path

import pytest
from conftest import FIXED_TS, edit_proposal, rubric_genome, trace_dict
from pydantic import ValidationError

from whystuck.core.errors import (
    BudgetIncreaseError,
    NotFoundError,
    PathEscapeError,
    SafetyLockError,
)
from whystuck.core.models import (
    Confidence,
    ConfidenceBand,
    GeneType,
    LoopGene,
    LoopGenome,
    LoopVariant,
    MutationProposal,
    MutationType,
)
from whystuck.core.runner import run_command_loop
from whystuck.core.storage import WhyStuckStore
from whystuck.tools import promote, propose_fix, record_trace, save_loop

# ---------------------------------------------------------------- storage bugs


def test_list_traces_limit_zero_returns_none(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    for i in range(3):
        record_trace(store, trace_dict(f"t{i}"))
    assert store.list_traces("bugfix", limit=0) == []


def test_list_traces_negative_limit_raises(tmp_path: Path):
    # A negative limit is a caller bug, not "all" and not "none" — fail loud.
    store = WhyStuckStore(tmp_path)
    record_trace(store, trace_dict("t0"))
    with pytest.raises(ValueError):
        store.list_traces("bugfix", limit=-1)


def test_load_variant_missing_raises_notfound(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    with pytest.raises(NotFoundError):
        store.load_variant("bugfix", "nope")


def test_load_loop_missing_raises_notfound(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    with pytest.raises(NotFoundError):
        store.load_loop("nope")


def test_unicode_trace_round_trips(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    record_trace(store, trace_dict("t1", summary="café ☕ 上手 — naïve"))
    assert store.list_traces("bugfix")[0].failure_summary == "café ☕ 上手 — naïve"


# ---------------------------------------------------------------- model validators


def test_confidence_rejects_score_above_band_cap():
    with pytest.raises(ValidationError):
        Confidence(band=ConfidenceBand.LOW, score=0.9)  # LOW cap is 0.4


def test_confidence_rejects_out_of_range_score():
    with pytest.raises(ValidationError):
        Confidence(band=ConfidenceBand.HIGH, score=1.5)


def test_genome_rejects_duplicate_gene_ids():
    with pytest.raises(ValidationError):
        LoopGenome(
            id="bugfix",
            name="B",
            genes=[
                LoopGene(id="g", type=GeneType.SKILL, name="a", content="x"),
                LoopGene(id="g", type=GeneType.PROMPT, name="b", content="y"),
            ],
        )


def test_budget_gene_requires_dict_content():
    with pytest.raises(ValidationError):
        LoopGene(id="b", type=GeneType.BUDGET_POLICY, name="Budget", content="max_usd: 1")


def test_edit_proposal_requires_after():
    with pytest.raises(ValidationError):
        MutationProposal(
            id="m",
            loop_id="l",
            mutation_type=MutationType.EDIT_GENE,
            target_gene_id="g",
            title="t",
            rationale="r",
            evidence_trace_ids=["t1"],
        )


def test_add_proposal_requires_dict_after():
    with pytest.raises(ValidationError):
        MutationProposal(
            id="m",
            loop_id="l",
            mutation_type=MutationType.ADD_GENE,
            title="t",
            rationale="r",
            evidence_trace_ids=[],
            after="not a dict",
        )


# ---------------------------------------------------------------- safety guard


def _budget_genome(budget: dict) -> LoopGenome:
    return LoopGenome(
        id="bugfix",
        name="B",
        genes=[LoopGene(id="budget", type=GeneType.BUDGET_POLICY, name="Budget", content=budget)],
    )


def test_budget_guard_rejects_dropping_the_cap_key():
    from whystuck.core.safety import assert_safe

    genome = _budget_genome({"max_usd": 1.0})
    with pytest.raises(BudgetIncreaseError):
        assert_safe(genome, edit_proposal("budget", {}))  # drops max_usd entirely


def test_budget_guard_rejects_removing_the_budget_gene():
    from whystuck.core.safety import assert_safe

    genome = _budget_genome({"max_usd": 1.0})
    remove = MutationProposal(
        id="m",
        loop_id="bugfix",
        mutation_type=MutationType.REMOVE_GENE,
        target_gene_id="budget",
        title="drop budget",
        rationale="r",
        evidence_trace_ids=["t1"],
    )
    with pytest.raises(BudgetIncreaseError):
        assert_safe(genome, remove)


# ---------------------------------------------------------------- mutation


def test_add_gene_rejects_duplicate_id():
    from whystuck.core.mutation import apply_mutation

    genome = rubric_genome()
    dup = MutationProposal(
        id="m",
        loop_id="bugfix",
        mutation_type=MutationType.ADD_GENE,
        title="add dup",
        rationale="r",
        evidence_trace_ids=[],
        after={"id": "rubric", "type": "verifier_rubric", "name": "R2", "content": "x"},
    )
    with pytest.raises(ValueError):
        apply_mutation(genome, dup)


# ---------------------------------------------------------------- runner


def test_worker_failure_but_verifier_pass_is_success():
    def execute(cmd: list[str]) -> tuple[int, str]:
        return (7, "worker blew up") if cmd == ["work"] else (0, "verified ok")

    trace = run_command_loop(
        loop_id="l",
        trace_id="t",
        created_at=FIXED_TS,
        worker=["work"],
        verifier=["check"],
        max_retries=0,
        execute=execute,
    )
    assert trace.final_outcome == "success"
    worker_step = next(s for s in trace.steps if s.role == "worker")
    assert worker_step.status.value == "failure"


def test_missing_command_becomes_failure_trace_not_crash():
    def execute(cmd: list[str]) -> tuple[int, str]:
        raise FileNotFoundError(cmd[0])

    trace = run_command_loop(
        loop_id="l",
        trace_id="t",
        created_at=FIXED_TS,
        worker=["does-not-exist"],
        verifier=["check"],
        max_retries=0,
        execute=execute,
    )
    assert trace.final_outcome == "failure"
    assert trace.failure_summary is not None


# -------------------------------------------------- tools: confidence + lineage + traversal


def test_propose_recomputes_confidence_and_ignores_client_value(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    store.save_loop(rubric_genome())
    proposal = {
        "id": "m1",
        "loop_id": "bugfix",
        "mutation_type": "edit_gene",
        "target_gene_id": "rubric",
        "title": "edit",
        "rationale": "r",
        "evidence_trace_ids": [],  # no evidence...
        "after": "check harder",
        "confidence": {"band": "high", "score": 0.99},  # ...client lies
    }
    out = propose_fix(store, "bugfix", [proposal])
    variant = store.load_variant("bugfix", out["variant_id"])
    # Empty evidence must force LOW confidence, not the client's HIGH.
    assert variant.mutations[0].confidence.band is ConfidenceBand.LOW


def test_confidence_low_when_evidence_cites_no_real_trace(tmp_path: Path):
    # Fabricated trace ids are no evidence — force LOW even when raw trace volume
    # alone would qualify for MEDIUM/HIGH (review finding #8).
    store = WhyStuckStore(tmp_path)
    store.save_loop(rubric_genome())
    for i in range(20):
        record_trace(store, trace_dict(f"t{i}", "failure", "boom"))
    proposal = {
        "id": "m1",
        "loop_id": "bugfix",
        "mutation_type": "edit_gene",
        "target_gene_id": "rubric",
        "title": "edit",
        "rationale": "r",
        "evidence_trace_ids": ["ghost-1", "ghost-2", "ghost-3"],  # none exist
        "after": "check harder",
    }
    out = propose_fix(store, "bugfix", [proposal])
    assert out["proposals"][0]["confidence"]["band"] == "low"
    variant = store.load_variant("bugfix", out["variant_id"])
    assert variant.mutations[0].confidence.band is ConfidenceBand.LOW


def test_confidence_duplicate_evidence_id_does_not_inflate_to_high():
    # Citing the SAME real trace id 15 times must not count as 15 distinct
    # pieces of evidence — dedupe before computing cluster_share, or a client
    # can manufacture HIGH confidence out of a single failing trace.
    from whystuck.tools import _confidence_for_evidence

    store_traces = [f"t{i}" for i in range(15)]

    class _FakeStore:
        def list_traces(self, loop_id: str) -> list:
            from whystuck.core.models import LoopTrace

            return [
                LoopTrace(id=tid, loop_id="bugfix", created_at=FIXED_TS, final_outcome="failure")
                for tid in store_traces
            ]

    confidence = _confidence_for_evidence(_FakeStore(), "bugfix", ["t0"] * 15)
    assert confidence.band is not ConfidenceBand.HIGH


def test_confidence_evidence_citing_only_successful_traces_is_low(tmp_path: Path):
    # Citing real trace ids that never failed is not evidence of the failure
    # mode — it must still force LOW, not raw-volume-driven MEDIUM/HIGH.
    store = WhyStuckStore(tmp_path)
    store.save_loop(rubric_genome())
    success_ids = []
    for i in range(10):
        record_trace(store, trace_dict(f"fail-{i}", "failure", "boom"))
    for i in range(10):
        tid = f"ok-{i}"
        record_trace(store, trace_dict(tid, "success", None))
        success_ids.append(tid)
    proposal = {
        "id": "m1",
        "loop_id": "bugfix",
        "mutation_type": "edit_gene",
        "target_gene_id": "rubric",
        "title": "edit",
        "rationale": "r",
        "evidence_trace_ids": success_ids,  # real ids, but none failing
        "after": "check harder",
    }
    out = propose_fix(store, "bugfix", [proposal])
    assert out["proposals"][0]["confidence"]["band"] == "low"


def test_confidence_cluster_share_is_distinct_failing_ids_over_all_failing():
    # K distinct cited failing ids of N failing traces → cluster_share = K/N.
    from whystuck.tools import _confidence_for_evidence

    class _FakeStore:
        def list_traces(self, loop_id: str) -> list:
            from whystuck.core.models import LoopTrace

            return [
                LoopTrace(
                    id=f"t{i}", loop_id="bugfix", created_at=FIXED_TS, final_outcome="failure"
                )
                for i in range(16)
            ]

    confidence = _confidence_for_evidence(_FakeStore(), "bugfix", [f"t{i}" for i in range(8)])
    assert confidence.score == 0.5  # 8 distinct / 16 failing
    assert confidence.band is ConfidenceBand.HIGH


def test_propose_on_missing_loop_raises_notfound(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    with pytest.raises(NotFoundError):
        propose_fix(store, "ghost", [])


def _skill_variant(vid: str, content: str) -> LoopVariant:
    return LoopVariant(
        id=vid,
        loop_id="bugfix",
        genome=LoopGenome(
            id="bugfix",
            name="B",
            genes=[
                LoopGene(
                    id="s",
                    type=GeneType.SKILL,
                    name="S",
                    path="skills/x.md",
                    content=content,
                )
            ],
        ),
        created_at=FIXED_TS,
    )


def _skill_loop(content: str) -> dict:
    return {
        "id": "bugfix",
        "name": "B",
        "genes": [
            {"id": "s", "type": "skill", "name": "S", "path": "skills/x.md", "content": content}
        ],
    }


def test_promotion_lineage_uses_most_recent_not_alphabetical(tmp_path: Path):
    store = WhyStuckStore(tmp_path / "store")
    proj = tmp_path / "proj"
    proj.mkdir()
    store.save_loop(rubric_genome())
    # ids chosen so alphabetical != chronological
    store.save_variant(_skill_variant("variant-zzz", "first"))
    store.save_variant(_skill_variant("variant-aaa", "second"))

    promote(store, "bugfix", "variant-zzz", base_dir=str(proj), dry_run=False)
    out = promote(store, "bugfix", "variant-aaa", base_dir=str(proj), dry_run=False)

    assert out["from_variant_id"] == "variant-zzz"
    promoted = [v.id for v in store.list_variants("bugfix") if v.promoted]
    assert promoted == ["variant-aaa"]  # exactly one live variant


def test_first_promotion_is_rollbackable_via_genesis(tmp_path: Path):
    # The first promotion must name a real prior state so it can be rolled back
    # by re-promoting it (review finding #3).
    store = WhyStuckStore(tmp_path / "store")
    proj = tmp_path / "proj"
    (proj / "skills").mkdir(parents=True)
    (proj / "skills" / "x.md").write_text("original")

    save_loop(store, _skill_loop("original"))
    variant_id = propose_fix(
        store,
        "bugfix",
        [
            {
                "id": "m1",
                "loop_id": "bugfix",
                "mutation_type": "edit_gene",
                "target_gene_id": "s",
                "title": "edit skill",
                "rationale": "r",
                "evidence_trace_ids": [],
                "after": "improved",
            }
        ],
    )["variant_id"]

    out = promote(store, "bugfix", variant_id, base_dir=str(proj), dry_run=False)
    assert (proj / "skills" / "x.md").read_text() == "improved"
    genesis_id = out["from_variant_id"]
    assert genesis_id is not None  # first promotion now names a real prior state

    # Roll back = re-promote the genesis variant → original content restored.
    promote(store, "bugfix", genesis_id, base_dir=str(proj), dry_run=False)
    assert (proj / "skills" / "x.md").read_text() == "original"


def test_promote_rejects_path_traversal(tmp_path: Path):
    store = WhyStuckStore(tmp_path / "store")
    proj = tmp_path / "proj"
    proj.mkdir()
    store.save_loop(rubric_genome())
    store.save_variant(
        LoopVariant(
            id="evil",
            loop_id="bugfix",
            genome=LoopGenome(
                id="bugfix",
                name="B",
                genes=[
                    LoopGene(
                        id="s",
                        type=GeneType.SKILL,
                        name="S",
                        path="../../etc/evil.md",
                        content="pwned",
                    )
                ],
            ),
            created_at=FIXED_TS,
        )
    )
    with pytest.raises(PathEscapeError):
        promote(store, "bugfix", "evil", base_dir=str(proj), dry_run=True)


def test_promote_absolute_path_rejected(tmp_path: Path):
    store = WhyStuckStore(tmp_path / "store")
    proj = tmp_path / "proj"
    proj.mkdir()
    store.save_loop(rubric_genome())
    store.save_variant(
        LoopVariant(
            id="abs",
            loop_id="bugfix",
            genome=LoopGenome(
                id="bugfix",
                name="B",
                genes=[
                    LoopGene(
                        id="s",
                        type=GeneType.SKILL,
                        name="S",
                        path="/tmp/evil.md",
                        content="pwned",
                    )
                ],
            ),
            created_at=FIXED_TS,
        )
    )
    with pytest.raises(PathEscapeError):
        promote(store, "bugfix", "abs", base_dir=str(proj), dry_run=True)


def test_lock_then_edit_in_one_set_is_refused(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    store.save_loop(rubric_genome())
    lock = {
        "id": "m1",
        "loop_id": "bugfix",
        "mutation_type": "lock_gene",
        "target_gene_id": "rubric",
        "title": "lock",
        "rationale": "safety",
        "evidence_trace_ids": [],
    }
    edit = {
        "id": "m2",
        "loop_id": "bugfix",
        "mutation_type": "edit_gene",
        "target_gene_id": "rubric",
        "title": "edit",
        "rationale": "r",
        "evidence_trace_ids": ["t1"],
        "after": "sneaky",
    }
    with pytest.raises(SafetyLockError):
        propose_fix(store, "bugfix", [lock, edit])


def test_save_loop_smoke(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    save_loop(store, rubric_genome().model_dump(mode="json"))
    assert store.load_loop("bugfix").id == "bugfix"
