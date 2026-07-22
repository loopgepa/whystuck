"""Tests for whystuck.eval.study — the 4-arm GEPA validation study orchestration.

Fully offline: a fake `complete` (keyed on an "OPTIMIZED" sentinel), a fake
`_optimize` (injects the sentinel into the evolvable genes, standing in for
GEPA's search), and small IFBench fixtures (single-constraint no_whitespace, plus
a mixed no_whitespace/sub-bullets set for fractional scoring) so scoring is
deterministic. No gepa, no network, no real LLM.

Scope: this pins the *wiring* of the arms (which path each takes, how many genes
each evolves, seed-vs-optimized held-out scoring). Arm C's genome-blind
*flatness* is a real-GEPA behavior an injected fake can't reproduce — it is
asserted at the E2E/live steps and by the existing BUG-2 unit test, not here.
"""

from __future__ import annotations

import json

import pytest

from whystuck.core.storage import WhyStuckStore
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.ifbench import load_ifbench
from whystuck.eval.study import (
    GENERATE_GENE,
    SEED_VERIFY,
    VERIFY_GENE,
    _heldout_score,
    run_study,
    seed_genome,
)


def _ifbench_task(constraint, key):
    return BenchmarkTask(id=key, payload={"goal": f"do {key}", "instruction_id_list": [constraint]})


SENTINEL = "OPTIMIZED"


def _fake_complete(prompt: str) -> str:
    # An "optimized" genome carries the sentinel in its prompt → constraint-
    # satisfying output ("solid" has no whitespace); otherwise a failing one.
    return "solid" if SENTINEL in prompt else "x x x"


def _fake_optimize(**kw):
    # Stand-in for GEPA: prepend the sentinel to every evolvable gene.
    seed = kw["seed_candidate"]
    return {"best_candidate": {gid: f"{SENTINEL} {text}" for gid, text in seed.items()}}


@pytest.fixture
def tasks(tmp_path):
    fixture = tmp_path / "nw.jsonl"
    fixture.write_text(
        "\n".join(
            json.dumps(
                {"key": str(i), "prompt": f"p{i}", "instruction_id_list": ["format:no_whitespace"]}
            )
            for i in range(4)
        ),
        encoding="utf-8",
    )
    train = load_ifbench(split="train", path=fixture)
    heldout = load_ifbench(split="heldout", path=fixture)
    return train, heldout


def test_seed_genome_has_two_genes_all_arms_share_content():
    a = seed_genome("x", evolvable=set())
    b = seed_genome("y", evolvable={GENERATE_GENE})
    ids = [g.id for g in a.genes]
    assert ids == [GENERATE_GENE, VERIFY_GENE]
    # identical content across arms → identical seed prompt (fair parity)
    assert [g.content for g in a.genes] == [g.content for g in b.genes]


def test_arm_a_is_seed_baseline_no_variant(tmp_path, tasks):
    train, heldout = tasks
    store = WhyStuckStore(tmp_path / "store")
    res = run_study(
        store,
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("A",),
        _optimize=_fake_optimize,
    )
    a = res["A"]
    assert a.variant_id is None
    assert a.evolvable_genes == ()
    assert a.seed_heldout == a.opt_heldout == 0.0  # seed prompt fails no_whitespace
    assert a.delta == 0.0


def test_arm_b_evolves_one_gene_and_improves(tmp_path, tasks):
    train, heldout = tasks
    store = WhyStuckStore(tmp_path / "store")
    res = run_study(
        store,
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("B",),
        _optimize=_fake_optimize,
    )
    b = res["B"]
    assert b.evolvable_genes == (GENERATE_GENE,)  # prompt-only: exactly one component
    assert b.variant_id is not None
    assert b.seed_heldout == 0.0 and b.opt_heldout == 1.0 and b.delta == 1.0


def test_arm_d_evolves_both_genes(tmp_path, tasks):
    train, heldout = tasks
    store = WhyStuckStore(tmp_path / "store")
    res = run_study(
        store,
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("D",),
        _optimize=_fake_optimize,
    )
    d = res["D"]
    assert d.evolvable_genes == (GENERATE_GENE, VERIFY_GENE)  # loop-level: whole genome
    assert d.variant_id is not None
    assert d.opt_heldout == 1.0


def test_arm_c_runs_command_path_and_returns_result(tmp_path, tasks):
    train, heldout = tasks
    store = WhyStuckStore(tmp_path / "store")
    res = run_study(
        store,
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("C",),
        worker_cmd=("true",),
        verifier_cmd=("true",),
        _optimize=_fake_optimize,
    )
    c = res["C"]
    assert c.variant_id is not None  # command path optimized + produced a variant
    # held-out is LLM-graded (via `complete`) for C too, not the command outcome:
    assert c.evolvable_genes == (GENERATE_GENE, VERIFY_GENE)  # same evolvable set as D
    assert c.seed_heldout == 0.0
    assert c.opt_heldout == 1.0
    assert c.delta == 1.0


def test_delta_is_fractional_on_mixed_constraints(tmp_path):
    # Explicit held-out (not a split) so the fraction is shuffle-independent:
    # "x x x" fails no_whitespace but trivially passes sub-bullets (no '*'),
    # "solid" passes both. seed = 2/4 = 0.5, opt = 4/4 = 1.0, delta = 0.5.
    heldout = [
        _ifbench_task("format:no_whitespace", "n1"),
        _ifbench_task("format:no_whitespace", "n2"),
        _ifbench_task("format:sub-bullets", "s1"),
        _ifbench_task("format:sub-bullets", "s2"),
    ]
    store = WhyStuckStore(tmp_path / "store")
    res = run_study(
        store,
        complete=_fake_complete,
        reflection_lm="fake",
        train=heldout,
        heldout=heldout,
        arms=("B",),
        _optimize=_fake_optimize,
    )
    b = res["B"]
    assert b.seed_heldout == 0.5
    assert b.opt_heldout == 1.0
    assert b.delta == 0.5


def test_heldout_score_of_empty_tasks_is_zero():
    # Empty held-out must return 0.0, never a vacuous 1.0 or a ZeroDivisionError.
    assert _heldout_score(seed_genome("x", evolvable=set()), [], lambda p: "anything") == 0.0


def test_b_leaves_verify_gene_at_seed_while_d_evolves_it(tmp_path, tasks):
    train, heldout = tasks
    store = WhyStuckStore(tmp_path / "store")
    res = run_study(
        store,
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("B", "D"),
        _optimize=_fake_optimize,
    )
    b_verify = next(
        g
        for g in store.load_variant("study-B", res["B"].variant_id).genome.genes
        if g.id == VERIFY_GENE
    ).content
    d_verify = next(
        g
        for g in store.load_variant("study-D", res["D"].variant_id).genome.genes
        if g.id == VERIFY_GENE
    ).content
    assert b_verify == SEED_VERIFY  # prompt-only never touches the verifier gene
    assert d_verify != SEED_VERIFY  # loop-level evolves it
    assert set(res["B"].evolvable_genes) < set(res["D"].evolvable_genes)  # strict subset


def test_all_four_arms_run_together(tmp_path, tasks):
    train, heldout = tasks
    store = WhyStuckStore(tmp_path / "store")
    res = run_study(
        store,
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        worker_cmd=("true",),
        verifier_cmd=("true",),
        _optimize=_fake_optimize,
    )
    assert set(res) == {"A", "B", "C", "D"}
    # Fair-parity: identical seed content → identical seed held-out for every arm.
    assert (
        res["A"].seed_heldout
        == res["B"].seed_heldout
        == res["C"].seed_heldout
        == res["D"].seed_heldout
    )
    # A never optimizes; B and D improve on held-out via the LLM worker.
    assert res["A"].opt_heldout == 0.0
    assert res["B"].opt_heldout == 1.0
    assert res["D"].opt_heldout == 1.0
