"""Regression tests for issues found in the multi-agent code review.

Each test pins a specific finding (path traversal, limit=0, promotion lineage,
server-side confidence, budget-guard bypass, runner robustness, validators).
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from whystuck.core.errors import (
    BudgetIncreaseError,
    NotFoundError,
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
from whystuck.tools import promote_variant, propose_mutations, record_trace, save_loop

# ---------------------------------------------------------------- storage bugs


def _trace(tid: str, outcome: str = "failure", summary: str | None = "boom") -> dict:
    return {
        "id": tid,
        "loop_id": "bugfix",
        "created_at": "2026-07-20T00:00:00Z",
        "final_outcome": outcome,
        "failure_summary": summary,
    }


def test_list_traces_limit_zero_returns_none(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    for i in range(3):
        record_trace(store, _trace(f"t{i}"))
    assert store.list_traces("bugfix", limit=0) == []


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
    record_trace(store, _trace("t1", summary="café ☕ 上手 — naïve"))
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


def _edit(target: str, after, mid: str = "m1") -> MutationProposal:
    return MutationProposal(
        id=mid,
        loop_id="bugfix",
        mutation_type=MutationType.EDIT_GENE,
        target_gene_id=target,
        title="edit",
        rationale="r",
        evidence_trace_ids=["t1"],
        after=after,
    )


def test_budget_guard_rejects_dropping_the_cap_key():
    from whystuck.core.safety import assert_safe

    genome = _budget_genome({"max_usd": 1.0})
    with pytest.raises(BudgetIncreaseError):
        assert_safe(genome, _edit("budget", {}))  # drops max_usd entirely


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

    genome = LoopGenome(
        id="bugfix",
        name="B",
        genes=[LoopGene(id="rubric", type=GeneType.VERIFIER_RUBRIC, name="R", content="check")],
    )
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
        created_at="2026-07-20T00:00:00Z",
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
        created_at="2026-07-20T00:00:00Z",
        worker=["does-not-exist"],
        verifier=["check"],
        max_retries=0,
        execute=execute,
    )
    assert trace.final_outcome == "failure"
    assert trace.failure_summary is not None


# -------------------------------------------------- tools: confidence + lineage + traversal


def _rubric_genome() -> LoopGenome:
    return LoopGenome(
        id="bugfix",
        name="B",
        genes=[LoopGene(id="rubric", type=GeneType.VERIFIER_RUBRIC, name="R", content="check")],
    )


def test_propose_recomputes_confidence_and_ignores_client_value(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    store.save_loop(_rubric_genome())
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
    out = propose_mutations(store, "bugfix", [proposal])
    variant = store.load_variant("bugfix", out["variant_id"])
    # Empty evidence must force LOW confidence, not the client's HIGH.
    assert variant.mutations[0].confidence.band is ConfidenceBand.LOW


def test_propose_on_missing_loop_raises_notfound(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    with pytest.raises(NotFoundError):
        propose_mutations(store, "ghost", [])


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
        created_at="2026-07-20T00:00:00Z",
    )


def test_promotion_lineage_uses_most_recent_not_alphabetical(tmp_path: Path):
    store = WhyStuckStore(tmp_path / "store")
    proj = tmp_path / "proj"
    proj.mkdir()
    store.save_loop(_rubric_genome())
    # ids chosen so alphabetical != chronological
    store.save_variant(_skill_variant("variant-zzz", "first"))
    store.save_variant(_skill_variant("variant-aaa", "second"))

    promote_variant(store, "bugfix", "variant-zzz", base_dir=str(proj), dry_run=False)
    out = promote_variant(store, "bugfix", "variant-aaa", base_dir=str(proj), dry_run=False)

    assert out["from_variant_id"] == "variant-zzz"
    promoted = [v.id for v in store.list_variants("bugfix") if v.promoted]
    assert promoted == ["variant-aaa"]  # exactly one live variant


def test_promote_rejects_path_traversal(tmp_path: Path):
    store = WhyStuckStore(tmp_path / "store")
    proj = tmp_path / "proj"
    proj.mkdir()
    store.save_loop(_rubric_genome())
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
            created_at="2026-07-20T00:00:00Z",
        )
    )
    with pytest.raises(ValueError):
        promote_variant(store, "bugfix", "evil", base_dir=str(proj), dry_run=True)


def test_promote_absolute_path_rejected(tmp_path: Path):
    store = WhyStuckStore(tmp_path / "store")
    proj = tmp_path / "proj"
    proj.mkdir()
    store.save_loop(_rubric_genome())
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
            created_at="2026-07-20T00:00:00Z",
        )
    )
    with pytest.raises(ValueError):
        promote_variant(store, "bugfix", "abs", base_dir=str(proj), dry_run=True)


def test_lock_then_edit_in_one_set_is_refused(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    store.save_loop(_rubric_genome())
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
        propose_mutations(store, "bugfix", [lock, edit])


def test_save_loop_smoke(tmp_path: Path):
    store = WhyStuckStore(tmp_path)
    save_loop(store, _rubric_genome().model_dump(mode="json"))
    assert store.load_loop("bugfix").id == "bugfix"
