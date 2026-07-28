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

from whystuck.core.models import LoopMetrics, LoopTrace
from whystuck.core.storage import WhyStuckStore
from whystuck.core.util import mean, utc_now_iso
from whystuck.eval.benchmark import BenchmarkTask
from whystuck.eval.ifbench import load_ifbench
from whystuck.eval.study import (
    ACTUATOR_GENE,
    GENERATE_GENE,
    SEED_GENERATE,
    SEED_VERIFY,
    VERIFY_GENE,
    _heldout_by_task,
    run_study,
    seed_genome,
)


def _ifbench_task(constraint, key):
    return BenchmarkTask(id=key, payload={"goal": f"do {key}", "instruction_id_list": [constraint]})


SENTINEL = "OPTIMIZED"


def _fake_run_task(genome, task):
    """A minimal run_task that records a programmatic score, as the real worker does."""
    return LoopTrace(
        id=f"r-{task.id}",
        loop_id=genome.id,
        created_at=utc_now_iso(),
        task_id=task.id,
        final_outcome="success",
        metrics=LoopMetrics(success=True),
        metadata={"eval_score": 1.0},
    )


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


def test_arm_c_runs_the_genome_blind_control_and_returns_result(tmp_path, tasks):
    train, heldout = tasks
    store = WhyStuckStore(tmp_path / "store")
    res = run_study(
        store,
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("C",),
        _optimize=_fake_optimize,
    )
    c = res["C"]
    assert c.variant_id is not None  # command path optimized + produced a variant
    # held-out is LLM-graded (via `complete`) for C too, not the command outcome:
    assert c.evolvable_genes == (GENERATE_GENE, VERIFY_GENE)  # same evolvable set as D
    assert c.seed_heldout == 0.0
    assert c.opt_heldout == 1.0
    assert c.delta == 1.0


def test_run_study_forwards_proposal_cap_to_optimizer(tmp_path, tasks):
    # The study must pass the reflection-call cap down to GEPA (as
    # max_candidate_proposals) so the cost guard's reflection budget is enforced,
    # not just estimated. Capture what the optimizer actually receives.
    train, heldout = tasks
    store = WhyStuckStore(tmp_path / "store")
    seen: list = []

    def capturing_optimize(**kw):
        seen.append(kw.get("max_candidate_proposals"))
        return _fake_optimize(**kw)

    run_study(
        store,
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("B", "D"),
        max_candidate_proposals=15,
        _optimize=capturing_optimize,
    )
    # One optimize call per LLM-optimizing arm, each told the same hard cap.
    assert seen == [15, 15]


def test_study_arms_bd_use_fractional_ifbench_signal_not_binary(tmp_path):
    # Guards the exact production wiring at study.py's make_run_task call: arms
    # B/D must train on fractional `ifbench_score` + `ifbench_feedback`, NOT the
    # old binary `ifbench_pass`. The fake `_optimize` never calls run_task, so we
    # capture the adapter GEPA is handed and exercise its run_task directly with a
    # partial-satisfaction output ("a\nb" passes format:newline but fails
    # format:no_whitespace → score 0.5). A revert to `ifbench_pass` would collapse
    # this to 0.0 and drop the constraint-naming feedback, failing here.
    store = WhyStuckStore(tmp_path / "store")
    task = BenchmarkTask(
        id="mix",
        payload={"goal": "g", "instruction_id_list": ["format:no_whitespace", "format:newline"]},
    )
    captured: dict = {}

    def capturing_optimize(**kw):
        captured["adapter"] = kw["adapter"]
        return _fake_optimize(**kw)

    run_study(
        store,
        complete=lambda prompt: "a\nb",
        reflection_lm="fake",
        train=[task],
        heldout=[task],
        arms=("B",),
        _optimize=capturing_optimize,
    )
    adapter = captured["adapter"]
    trace = adapter.run_task(seed_genome("study-B", evolvable={GENERATE_GENE}), task)
    assert trace.metadata["eval_score"] == 0.5  # fractional, not binary 0/1
    assert "format:no_whitespace" in trace.metadata["feedback"]  # names the failed constraint


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


def test_heldout_score_of_empty_tasks_raises_not_silent_zero():
    # An empty held-out list is a misconfiguration (e.g. n_heldout=0): returning a
    # silent 0.0 would report every arm as catastrophic in the paid report,
    # indistinguishable from a genuine all-fail. Fail loud instead.
    with pytest.raises(ValueError, match="empty"):
        _heldout_by_task(seed_genome("x", evolvable=set()), [], lambda g, t: "anything")


def test_heldout_score_raises_when_trace_lacks_eval_score():
    # A run_task whose trace omits metadata['eval_score'] is misconfigured. Defaulting
    # to 0.0 would silently report every genome as catastrophic (looks like "GEPA made
    # it worse") and write that into the report. Fail loudly instead.
    def bad_run_task(genome, task):
        return LoopTrace(
            id="r",
            loop_id=genome.id,
            created_at=utc_now_iso(),
            final_outcome="failure",
            metrics=LoopMetrics(success=False),
            metadata={},  # no eval_score
        )

    with pytest.raises(ValueError, match="eval_score"):
        _heldout_by_task(seed_genome("x", evolvable=set()), [BenchmarkTask(id="t")], bad_run_task)


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


def test_run_study_wires_multi_constraint_scorer(tmp_path):
    # The multi-constraint task set must flow through run_study via score_fn/
    # feedback_fn: all 4 arms run, seed parity holds, and the injected scorer
    # produces fractional held-out scores (the mc path is wired, not just ifbench).
    from whystuck.eval.multi_constraint import generate_multi_tasks, mc_feedback, mc_score

    pool = generate_multi_tasks(8, seed=0)
    train, heldout = pool[:4], pool[4:]

    def mc_worker(prompt: str) -> str:
        # genome-blind, generic answer satisfying ~half the constraints
        return "- idea system\n- result signal that is all."

    res = run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=mc_worker,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        _optimize=_fake_optimize,
        score_fn=mc_score,
        feedback_fn=mc_feedback,
    )
    assert set(res) == {"A", "B", "C", "D"}
    assert res["A"].seed_heldout == res["B"].seed_heldout == res["D"].seed_heldout
    # fractional signal from mc_score (not the ifbench no_whitespace 0/1 pattern)
    assert all(0.0 <= res[a].opt_heldout <= 1.0 for a in ("A", "B", "C", "D"))
    assert 0.0 < res["A"].seed_heldout < 1.0


def test_run_study_forwards_exec_feedback_to_the_worker_loop(tmp_path):
    # The v2-code execution signal must reach the loop's critique via run_study. Arm A's
    # held-out scoring runs the same self-refine worker, so a spy over it sees the block.
    calls = []

    def spy(prompt: str) -> str:
        calls.append(prompt)
        return "PASS" if "# Draft answer" in prompt else "draft"

    run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=spy,
        reflection_lm="fake",
        train=[BenchmarkTask(id="t0", payload={"goal": "g"})],
        heldout=[BenchmarkTask(id="t1", payload={"goal": "g"})],
        arms=("A",),
        _optimize=_fake_optimize,
        score_fn=lambda o, t: 0.0,
        exec_feedback=lambda draft, task: f"EXEC[{task.id}]",
    )
    critique_prompts = [c for c in calls if "# Draft answer" in c]
    assert critique_prompts and all("EXEC[t1]" in c for c in critique_prompts)


# --- run-4/run-5 levers: feedback stats + a third `plan` gene -------------------


def test_seed_genome_gets_a_plan_gene_when_requested():
    from whystuck.eval.study import PLAN_GENE

    g = seed_genome("x", evolvable=set(), include_plan=True)
    assert [gene.id for gene in g.genes] == [GENERATE_GENE, VERIFY_GENE, PLAN_GENE]
    # default stays two-gene (existing runs unchanged)
    assert len(seed_genome("x", evolvable=set()).genes) == 2


def test_arm_evolvable_extends_bd_to_the_plan_gene():
    from whystuck.eval.study import PLAN_GENE, arm_evolvable

    assert arm_evolvable("A", include_plan=True) == set()
    assert arm_evolvable("B", include_plan=True) == {GENERATE_GENE}  # prompt-only stays one gene
    assert arm_evolvable("D", include_plan=True) == {GENERATE_GENE, VERIFY_GENE, PLAN_GENE}
    assert PLAN_GENE not in arm_evolvable("D", include_plan=False)  # off by default


def test_seed_genome_gets_an_actuator_gene_when_requested():
    from whystuck.eval.study import ACTUATOR_GENE

    g = seed_genome("x", evolvable=set(), include_actuator=True)
    assert [gene.id for gene in g.genes] == [GENERATE_GENE, VERIFY_GENE, ACTUATOR_GENE]
    assert len(seed_genome("x", evolvable=set()).genes) == 2  # default unchanged


def test_actuator_mode_evolves_generate_and_actuator_but_holds_the_detector_fixed():
    # The whole point of the actuator experiment: D optimizes the fixer's feedback
    # (generate + actuator), NOT the detector (ensure_correct), which stays at seed.
    from whystuck.eval.study import ACTUATOR_GENE, arm_evolvable

    assert arm_evolvable("A", actuator=True) == set()
    assert arm_evolvable("B", actuator=True) == {GENERATE_GENE}  # prompt-only baseline
    assert arm_evolvable("D", actuator=True) == {GENERATE_GENE, ACTUATOR_GENE}
    assert VERIFY_GENE not in arm_evolvable("D", actuator=True)  # detector held fixed
    assert ACTUATOR_GENE not in arm_evolvable("D")  # off by default


def test_run_study_with_plan_gene_lets_only_d_evolve_the_plan(tmp_path, tasks):
    from whystuck.eval.study import PLAN_GENE

    train, heldout = tasks
    res = run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        _optimize=_fake_optimize,
        use_plan_gene=True,
    )
    assert PLAN_GENE in res["D"].evolvable_genes
    assert PLAN_GENE not in res["B"].evolvable_genes
    assert res["B"].evolvable_genes == (GENERATE_GENE,)


def test_run_study_feeds_stats_into_the_critique(tmp_path, tasks):
    train, heldout = tasks
    seen = []

    def spy(prompt: str) -> str:
        seen.append(prompt)
        return _fake_complete(prompt)

    run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=spy,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("A",),  # held-out scoring alone exercises the loop
        _optimize=_fake_optimize,
        stats_fn=lambda _out: "STATSMARK words=3",
    )
    critiques = [p for p in seen if "# Draft answer" in p]
    assert critiques and all("STATSMARK words=3" in p for p in critiques)


def test_run_study_routes_critique_to_a_separate_critic_model(tmp_path, tasks):
    train, heldout = tasks
    worker_seen, critic_seen = [], []

    def worker(p: str) -> str:
        worker_seen.append(p)
        return _fake_complete(p)

    def critic(p: str) -> str:
        critic_seen.append(p)
        return "FAIL: fix it"

    run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=worker,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("A",),
        _optimize=_fake_optimize,
        critique_complete=critic,
    )
    assert critic_seen and all("# Draft answer" in p for p in critic_seen)  # critique -> critic
    assert not any("# Draft answer" in p for p in worker_seen)  # never on the worker


def test_run_study_oracle_critique_makes_no_critique_llm_call(tmp_path, tasks):
    train, heldout = tasks
    seen = []

    def spy(p: str) -> str:
        seen.append(p)
        return _fake_complete(p)

    run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=spy,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("A",),
        _optimize=_fake_optimize,
        oracle_critique=True,
    )
    assert seen  # the loop still generated
    assert not any("# Draft answer" in p for p in seen)  # oracle gate: no critique LLM call


def test_run_study_threads_oracle_feedback_into_the_refine_step(tmp_path, tasks):
    # D2' feedback-actionability lever: run_study must forward oracle_feedback to the
    # loop worker so it shapes the repair text handed to the refine step (arm A's
    # held-out scoring alone drives the loop; the fake worker always drafts "x x x"
    # which fails no_whitespace, so the oracle gate fails and a refine runs).
    train, heldout = tasks
    seen = []

    def spy(prompt: str) -> str:
        seen.append(prompt)
        return _fake_complete(prompt)

    run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=spy,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("A",),
        _optimize=_fake_optimize,
        oracle_critique=True,
        oracle_feedback=lambda o, t: "ACTIONABLE_MARKER",
    )
    refine = [p for p in seen if "# Problems to fix" in p]
    assert refine and all("ACTIONABLE_MARKER" in p for p in refine)
    assert not any("# Draft answer" in p for p in seen)  # still no critique LLM call


def _noop_optimize(**kw):
    """Stand-in for a GEPA run that accepts nothing: the winner IS the seed candidate.

    This is the empirically common case in the released study — on most seeds GEPA
    rejected every proposal for at least one gene — so the reporting must be able to
    say so out loud.
    """
    return {"best_candidate": dict(kw["seed_candidate"])}


def test_arm_result_reports_which_genes_actually_changed(tmp_path, tasks):
    # The load-bearing provenance datum: Δ(D−B) is only evidence about loop-level
    # optimization if the loop-level gene actually CHANGED. When GEPA accepts nothing,
    # genes_changed must be empty so a reader can tell "treatment did nothing" apart
    # from "treatment was never applied" — which a bare Δ=0 silently conflates.
    train, heldout = tasks
    res = run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("D",),
        _optimize=_noop_optimize,
    )
    assert res["D"].genes_changed == ()

    res2 = run_study(
        WhyStuckStore(tmp_path / "store2"),
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("D",),
        _optimize=_fake_optimize,
    )
    # The fake optimizer rewrites every evolvable gene, so both must be reported.
    assert set(res2["D"].genes_changed) == {GENERATE_GENE, VERIFY_GENE}


def test_arm_result_reports_train_aggregate_and_final_genes(tmp_path, tasks):
    # The paper attributes the null to train-split over-fitting, but the train aggregate
    # was computed and discarded, making that mechanism unfalsifiable from the release.
    # Both the train score and the final gene text must reach the result.
    train, heldout = tasks
    res = run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("B", "D"),
        _optimize=_fake_optimize,
    )
    for arm in ("B", "D"):
        assert res[arm].train_score is not None
        assert res[arm].final_genes is not None
        assert SENTINEL in res[arm].final_genes[GENERATE_GENE]
    # B may not touch the verify gene, D may.
    assert VERIFY_GENE not in (res["B"].final_genes or {})
    assert VERIFY_GENE in (res["D"].final_genes or {})


def test_run_study_forwards_module_selector(tmp_path, tasks):
    # Arm fairness knob must reach the optimizer, and default to GEPA's own default so
    # the committed cassettes keep replaying byte-for-byte.
    train, heldout = tasks
    seen = {}

    def capturing_optimize(**kw):
        seen.update(kw)
        return _fake_optimize(**kw)

    run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("D",),
        _optimize=capturing_optimize,
    )
    assert seen["module_selector"] == "round_robin"

    run_study(
        WhyStuckStore(tmp_path / "store2"),
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("D",),
        module_selector="all",
        _optimize=capturing_optimize,
    )
    assert seen["module_selector"] == "all"


def test_fixed_draft_control_is_flat_but_not_saturated(tmp_path, tasks):
    # The replacement control: genome-blind during SEARCH (so the training signal carries
    # no information about the genome) but scored by the REAL metric on a deliberately
    # imperfect constant draft, so there is real headroom and GEPA actually searches.
    #
    # This is what makes arm C a control rather than a tautology: it selects a genome using
    # a signal that cannot see the genome, then scores that genome with the real worker. If
    # blindly-selected genomes reliably improved held-out, the whole comparison would be
    # measuring harness noise. The released command control could never show that, because
    # `scalarize` handed it a perfect 1.0 and GEPA terminated without proposing anything.
    train, heldout = tasks
    res = run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("C",),
        _optimize=_noop_optimize,
    )
    c = res["C"]
    assert c.train_score is not None
    assert c.train_score < 1.0  # NOT saturated: there is headroom to search
    assert c.genes_changed == ()  # this optimizer accepted nothing...
    assert c.delta == 0.0  # ...so held-out cannot move either


def test_arm_a_reports_a_real_second_measurement_not_a_copy(tmp_path, tasks):
    # Arm A used to hardcode delta=0.0 by copying its own seed score, which made "a non-flat
    # A would invalidate the run" unfalsifiable. It now scores held-out a second time through
    # the same path. The two agree whenever that path is deterministic, so this is a
    # DETERMINISM check, not a flatness control — labelling it honestly is the point.
    train, heldout = tasks
    res = run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("A",),
        _optimize=_fake_optimize,
    )
    a = res["A"]
    assert a.opt_heldout == a.seed_heldout  # deterministic path agrees
    assert a.genes_changed == ()  # nothing evolved, and it is reported as such


# --- hand-written gene overrides (the verifier-ablation condition) -------------
#
# C1 asks a question no optimizer run can answer: does a BETTER verify gene move
# held-out at all? If a hand-written verifier cannot, there is no signal for GEPA
# to find and the loop-level null is a property of the loop, not of the search.
# That needs a way to seed a gene with author-supplied text, which is what these
# pin down.

HAND_VERIFY = "Count the words with a counter. Reply PASS or the exact edit needed."


def test_seed_genome_accepts_a_hand_written_gene_override():
    g = seed_genome("x", evolvable=set(), overrides={VERIFY_GENE: HAND_VERIFY})
    content = {gene.id: gene.content for gene in g.genes}
    assert content[VERIFY_GENE] == HAND_VERIFY
    assert content[GENERATE_GENE] == SEED_GENERATE  # every other gene untouched


def test_seed_genome_override_keeps_the_evolvable_flags():
    # The override changes the STARTING text, not what GEPA may rewrite: an ablation
    # arm and an optimized arm must still be distinguishable.
    g = seed_genome("x", evolvable={VERIFY_GENE}, overrides={VERIFY_GENE: HAND_VERIFY})
    assert next(gene for gene in g.genes if gene.id == VERIFY_GENE).evolvable


def test_seed_genome_rejects_an_override_for_a_gene_it_does_not_have():
    # Silently dropping it would run the UN-ablated condition and bill for it.
    with pytest.raises(KeyError, match=ACTUATOR_GENE):
        seed_genome("x", evolvable=set(), overrides={ACTUATOR_GENE: "text"})


def test_seed_genome_rejects_a_blank_override():
    # A blank verify gene does not disable the critique — it sends the critique LLM a
    # prompt with no rubric, which scores like a bad rubric. Fail loud instead.
    with pytest.raises(ValueError, match="blank"):
        seed_genome("x", evolvable=set(), overrides={VERIFY_GENE: "  \n "})


def test_gene_override_reaches_the_critique_prompt(tmp_path, tasks):
    """E2E: the injected text must actually be the head of the critique prompt.

    This is the claim C1's dollars rest on. `make_self_refine_run_task` builds the
    critique as `f"{ver}\\n\\n# Task..."`, so if the override does not arrive here it
    would buy a re-measurement of the seed condition under a different filename.
    """
    train, heldout = tasks
    prompts: list[str] = []

    def recording_complete(prompt: str) -> str:
        prompts.append(prompt)
        return "x x x"  # never PASS, so the critique always runs

    run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=recording_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("A",),
        gene_overrides={VERIFY_GENE: HAND_VERIFY},
        _optimize=_fake_optimize,
    )
    critiques = [p for p in prompts if "# Draft answer" in p]
    assert critiques, "no critique prompt was built"
    assert all(p.startswith(HAND_VERIFY) for p in critiques)
    assert not any(SEED_VERIFY in p for p in prompts)  # the seed rubric is gone entirely


def test_gene_override_is_the_baseline_for_genes_changed(tmp_path, tasks):
    # Provenance must compare against the OVERRIDDEN seed, not the module constant,
    # or an ablation arm would report its own starting text as an optimizer change.
    train, heldout = tasks
    res = run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("D",),
        gene_overrides={VERIFY_GENE: HAND_VERIFY},
        _optimize=lambda **kw: {"best_candidate": dict(kw["seed_candidate"])},  # accepts nothing
    )
    assert res["D"].genes_changed == ()


# --- per-task held-out scores -------------------------------------------------
#
# The aggregate mean cannot support the analysis these runs are for. IFBench seeds
# re-partition ONE 62-row pool, so any two seeds share ~50% of their held-out tasks
# (measured: 15.4 of 31) — per-seed results are correlated pseudo-replicates, and a
# sign test over seeds would assume an independence that does not hold. The unit that
# IS independent is the task, so the per-task scores have to survive into the output.


def test_heldout_by_task_returns_one_score_per_task(tmp_path, tasks):
    _, heldout = tasks
    got = _heldout_by_task(seed_genome("x", evolvable=set()), heldout, _fake_run_task)
    assert set(got) == {t.id for t in heldout}
    assert all(0.0 <= v <= 1.0 for v in got.values())


def test_heldout_by_task_rejects_duplicate_task_ids(tmp_path):
    # A duplicate id would silently collapse two tasks into one dict entry and change the
    # reported mean, with nothing in the output showing that fewer tasks were scored.
    dup = [
        BenchmarkTask(id="same", payload={"goal": "a"}),
        BenchmarkTask(id="same", payload={"goal": "b"}),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        _heldout_by_task(seed_genome("x", evolvable=set()), dup, _fake_run_task)


def test_arm_result_carries_per_task_heldout_scores(tmp_path, tasks):
    train, heldout = tasks
    res = run_study(
        WhyStuckStore(tmp_path / "store"),
        complete=_fake_complete,
        reflection_lm="fake",
        train=train,
        heldout=heldout,
        arms=("A", "D"),
        _optimize=_fake_optimize,
    )
    ids = {t.id for t in heldout}
    for arm in ("A", "D"):
        r = res[arm]
        assert set(r.seed_heldout_by_task) == ids
        assert set(r.opt_heldout_by_task) == ids
        # The reported aggregate must be exactly the mean of the released per-task scores,
        # or the two halves of the artifact disagree.
        assert r.seed_heldout == pytest.approx(mean(list(r.seed_heldout_by_task.values())))
        assert r.opt_heldout == pytest.approx(mean(list(r.opt_heldout_by_task.values())))
